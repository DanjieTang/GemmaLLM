from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from generate import generate
from lazy_dataloader import AnnotationDataset, collate_annotations, prepare_annotation_dataset
from model import VLM
from train import parse_args, run_epoch, save_training_checkpoint
from test_vlm import FakeVisionModel, FakeVisionProcessor


def test_training_defaults_to_paired_data_and_accepts_explicit_legacy_paths():
    assert parse_args([]).data_root == "data"
    assert parse_args(["--data_root", "data/data"]).data_root == "data/data"
    args = parse_args(["--train_path", "train.npy", "--val_path", "val.npy"])
    assert args.data_root is None
    assert args.train_path == "train.npy"


@pytest.mark.parametrize("flags", [
    ["--train_path", "train.npy"],
    ["--val_path", "val.npy"],
    ["--data_root", "data", "--train_path", "train.npy", "--val_path", "val.npy"],
    ["--train_image_paths", "images.npy"],
    ["--q_head", "0"],
    ["--kv_head", "0"],
    ["--head_dim", "0"],
])
def test_invalid_training_options_fail_before_loading_model(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)


def make_pair(root, name, ids, annotation_folder="annotations"):
    for folder, suffix in (("images", ".png"), (annotation_folder, ".txt"),
                           ("input_ids", ".npy")):
        path = root / folder / (name + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if folder == "images":
            Image.new("RGB", (2, 2), "red").save(path)
        elif folder == "input_ids":
            np.save(path, np.array(ids, dtype=np.int32))
        else:
            path.write_text("annotation")


def tiny_model(tmp_path, use_moe=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    embeddings = tmp_path / "embeddings.pt"
    torch.save(torch.randn(16, 6, dtype=torch.bfloat16), embeddings)
    config = dict(num_layer=1, max_context_length=6,
                  word_embeddings_tensor=str(embeddings), projection_dim=4,
                  expansion_factor=2, head_dim=2, q_head=2, kv_head=1,
                  dropout_ratio=0.0, use_moe=use_moe, num_experts=2)
    with patch("model.CLIPVisionModel.from_pretrained", return_value=FakeVisionModel()), \
         patch("model.CLIPImageProcessor.from_pretrained", return_value=FakeVisionProcessor()):
        model = VLM(**config, device="cpu")
    return model, config


def test_nested_pairs_shift_padding_and_truncation(tmp_path):
    make_pair(tmp_path / "train", "nested/one.v1", [2, 4, 5, 6, 7, 1])
    make_pair(tmp_path / "other_train", "nested/one.v1", [2, 8, 1], "images_annotation")
    dataset = AnnotationDataset([tmp_path / "train", tmp_path / "other_train"],
                                3, 16, 2, 1)
    assert len(dataset) == 2
    assert dataset[0][0].tolist() == [2, 4, 5, 1]
    batch = collate_annotations([dataset[0], dataset[1]], 0)
    assert batch["input_ids"].tolist() == [[2, 4, 5], [2, 8, 0]]
    assert batch["labels"].tolist() == [[4, 5, 1], [8, 1, -100]]
    assert batch["attention_mask"].tolist() == [[True]*3, [True, True, False]]
    assert batch["image_paths"][0] != batch["image_paths"][1]


@pytest.mark.parametrize("ids", [[2, 99, 1], [2, 4, 3], [4, 5, 1]])
def test_invalid_token_arrays_fail_with_filename(tmp_path, ids):
    make_pair(tmp_path, "one", ids)
    dataset = AnnotationDataset([tmp_path], 3, 16, 2, 1)
    with pytest.raises(ValueError, match="one.npy"):
        dataset[0]


def test_missing_pairs_are_counted_and_split_overlap_rejected(tmp_path):
    make_pair(tmp_path / "train", "one", [2, 1])
    make_pair(tmp_path / "train", "two", [2, 1])
    (tmp_path / "train/input_ids/two.npy").unlink()
    dataset = AnnotationDataset([tmp_path / "train"], 3, 16, 2, 1)
    assert len(dataset) == 1
    assert dataset.missing_tokens == 1
    with pytest.raises(ValueError, match="disjoint"):
        prepare_annotation_dataset(str(tmp_path), ["train"], ["train"],
                                   2, 3, 16, 2, 1, 0)


@pytest.mark.parametrize("use_moe", [False, True])
def test_padded_forward_matches_unpadded_and_backpropagates(tmp_path, use_moe):
    model, _ = tiny_model(tmp_path, use_moe)
    make_pair(tmp_path, "one", [2, 4, 5, 1])
    path = str(tmp_path / "images/one.png")
    model.eval()
    ids = torch.tensor([[2, 4, 5], [2, 8, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).bool()
    logits, aux = model(ids, [path, path], attention_mask=mask)
    single, _ = model(ids[1:2, :2], [path])
    torch.testing.assert_close(logits[1, :2], single[0], atol=1e-6, rtol=1e-5)
    labels = torch.tensor([[4, 5, 1], [8, 1, -100]])
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten()) + aux
    loss.backward()
    assert torch.isfinite(loss)
    assert model.seperation_token.grad.abs().sum() > 0
    assert model.text_token_projection.weight.grad.abs().sum() > 0
    assert model.llm.classifier.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.vision_model.parameters())
    assert model.vocabulary_size == 16


def test_causal_text_predictions_do_not_see_future(tmp_path):
    model, _ = tiny_model(tmp_path)
    model.eval()
    first, _ = model(torch.tensor([[2, 4, 5]]))
    second, _ = model(torch.tensor([[2, 4, 9]]))
    torch.testing.assert_close(first[:, :2], second[:, :2])


@pytest.mark.parametrize("use_cache", [False, True])
def test_generation_encodes_once_and_stops_at_eos(tmp_path, use_cache):
    model, _ = tiny_model(tmp_path)
    with patch.object(model, "_encode_images", return_value=torch.zeros(1, 1, 4)) as encode:
        next_ids = iter([4, 1])
        def next_logits(token_ids, **kwargs):
            logits = torch.full((1, token_ids.shape[1], 16), -100.0)
            logits[0, -1, next(next_ids)] = 100
            if kwargs.get("use_cache"):
                return logits, torch.tensor(0.0), object()
            return logits, torch.tensor(0.0)
        with patch.object(model, "forward", side_effect=next_logits) as forward:
            assert generate(model, "image.png", 2, 1, pad_token_id=0,
                            use_cache=use_cache) == [4]
            assert forward.call_count == 2
        encode.assert_called_once()
    assert model.training
    assert not model.vision_model.training


def test_training_checkpoint_roundtrip(tmp_path):
    model, config = tiny_model(tmp_path)
    make_pair(tmp_path / "data", "one", [2, 4, 5, 1])
    dataset = AnnotationDataset([tmp_path / "data"], 6, 16, 2, 1)
    batch = collate_annotations([dataset[0]], 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loss = run_epoch(model, [batch], "cpu", optimizer, scheduler)
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, model, config, "tokenizer", {}, optimizer,
                             scheduler, 1, loss)
    saved = torch.load(path, weights_only=True)
    restored, _ = tiny_model(tmp_path / "restore")
    # Frozen embeddings are external artifacts and intentionally not in state_dict.
    restored.word_embeddings_tensor.copy_(model.word_embeddings_tensor)
    restored.load_state_dict(saved["model_state_dict"])
    model.eval()
    restored.eval()
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    token_ids = inputs.pop("input_ids")
    torch.testing.assert_close(model(token_ids, **inputs)[0],
                               restored(token_ids, **inputs)[0])
