"""Python side of cuda/cuda_gemma_llm: PyTorch reference model and weight export."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from model import VLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda" / "cuda_gemma_llm"))
from export_weights import decoder_tensors, export_for_cuda  # noqa: E402
from text_model import TextGenerator, generate_greedy, load_checkpoint  # noqa: E402

from test_vlm import FakeVisionModel, FakeVisionProcessor  # noqa: E402


def save_vlm_checkpoint(directory: Path, fine_tuning: bool) -> tuple[Path, VLM]:
    """A train.py-style checkpoint whose embeddings (12 wide) are projected to 16."""
    torch.manual_seed(3)
    embeddings_path = directory / "embeddings.pt"
    torch.save(torch.randn(23, 12).to(torch.bfloat16), embeddings_path)
    model_config = dict(
        num_layer=2, max_context_length=8, word_embeddings_tensor=str(embeddings_path),
        projection_dim=16, expansion_factor=2, head_dim=4, q_head=4, kv_head=2,
        dropout_ratio=0.0, theta=10000, use_moe=False, fine_tuning=fine_tuning,
        lora_rank=2, lora_alpha=4,
    )
    with (
        patch("model.CLIPVisionModel.from_pretrained", return_value=FakeVisionModel()),
        patch("model.CLIPImageProcessor.from_pretrained", return_value=FakeVisionProcessor()),
    ):
        vlm = VLM(**model_config, device="cpu").eval()
    checkpoint_path = directory / "checkpoint.pt"
    torch.save({"model_config": model_config, "model_state_dict": vlm.state_dict()},
               checkpoint_path)
    return checkpoint_path, vlm


def decoder_logits(model: TextGenerator, token_ids: torch.Tensor) -> torch.Tensor:
    return model.llm(model.embed_tokens(token_ids), fine_tuning=model.fine_tuning)[0]


@pytest.mark.parametrize("fine_tuning", [False, True])
@torch.inference_mode()
def test_checkpoint_matches_vlm_text_only_path(tmp_path, fine_tuning):
    checkpoint_path, vlm = save_vlm_checkpoint(tmp_path, fine_tuning)
    model, architecture = load_checkpoint(checkpoint_path)
    token_ids = torch.tensor([[2, 5, 7, 1, 9]])

    expected, _ = vlm(token_ids)
    torch.testing.assert_close(decoder_logits(model, token_ids), expected)
    assert architecture["fine_tuning"] == fine_tuning


@torch.inference_mode()
def test_exported_lora_merge_reproduces_fine_tuned_logits(tmp_path):
    checkpoint_path, _ = save_vlm_checkpoint(tmp_path, fine_tuning=True)
    fine_tuned, architecture = load_checkpoint(checkpoint_path)
    merged = TextGenerator(**{**architecture, "fine_tuning": False},
                           word_embeddings=fine_tuned.word_embeddings_tensor,
                           device="cpu").eval()
    missing, unexpected = merged.load_state_dict(decoder_tensors(fine_tuned), strict=False)
    assert not unexpected
    assert all("lora" in name or "pos_emb" in name for name in missing)

    token_ids = torch.tensor([[2, 11, 4, 4, 19, 3]])
    torch.testing.assert_close(decoder_logits(merged, token_ids),
                               decoder_logits(fine_tuned, token_ids))


def test_export_writes_cuda_layout(tmp_path):
    checkpoint_path, _ = save_vlm_checkpoint(tmp_path, fine_tuning=False)
    model, _ = load_checkpoint(checkpoint_path)
    tokens, logits = generate_greedy(model, [2, 5], 6, return_first_logits=True)
    export_dir = tmp_path / "export"
    export_for_cuda(model, export_dir, reference=(2, tokens, logits))

    config = dict(line.split(": ") for line in (export_dir / "config.yaml").read_text().splitlines())
    assert config == {
        "format": "cuda_gemma_llm_v1", "num_layer": "2", "vocab_size": "23",
        "text_dim": "12", "projection_dim": "16", "expansion_factor": "2",
        "head_dim": "4", "q_head": "4", "kv_head": "2", "theta": "10000",
        "max_context_length": "8", "norm_eps": "1e-05", "embedding_dtype": "bfloat16",
    }
    embeddings = np.fromfile(export_dir / "embeddings.bin", dtype=np.int16)
    np.testing.assert_array_equal(
        embeddings, model.word_embeddings_tensor.view(torch.int16).flatten().numpy())
    for name, tensor in decoder_tensors(model).items():
        values = np.fromfile(export_dir / f"{name}.bin", dtype=np.float32)
        np.testing.assert_array_equal(values, tensor.detach().flatten().numpy())
    assert (export_dir / "reference_tokens.txt").read_text().split() == [
        "2", *map(str, tokens)]
    np.testing.assert_array_equal(
        np.fromfile(export_dir / "reference_logits.bin", dtype=np.float32), logits.numpy())
