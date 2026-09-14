from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import train_mtp
from test_annotation_training import make_pair, tiny_model
from train import prepare_batch


@pytest.fixture(scope="module", autouse=True)
def single_threaded_tensors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_trainer(tmp_path, use_moe=False, fine_tuning=False):
    base, config = tiny_model(tmp_path, use_moe)
    if fine_tuning:
        base.begin_fine_tunning()
    return train_mtp.MTPTrainingModel(base, config, 2, 0.3, 1), config


@pytest.mark.parametrize("flags", [
    ["--mtp_depth", "0"], ["--mtp_depth", "-1"],
    ["--mtp_depth", "6", "--max_context_length", "6"],
    ["--mtp_loss_weight", "0"], ["--mtp_loss_weight", "-0.1"],
    ["--mtp_loss_weight", "nan"], ["--mtp_loss_weight", "inf"],
    ["--q_head", "0"], ["--train_path", "train.npy"],
])
def test_invalid_options_fail_early(flags):
    with pytest.raises(SystemExit):
        train_mtp.parse_args(flags)


@pytest.mark.parametrize("use_moe", [False, True])
def test_joint_losses_align_text_positions_padding_and_final_eos(tmp_path, use_moe):
    model, _ = make_trainer(tmp_path, use_moe)
    captured = []
    handles = [module.register_forward_hook(
        lambda module, args, output: captured.append((args, output))
    ) for module in model.mtp_modules]
    ids = torch.tensor([[2, 4, 5, 6], [2, 8, 0, 0]])
    labels = torch.tensor([[4, 5, 6, 1], [8, 1, -100, -100]])
    mask = torch.tensor([[True]*4, [True, True, False, False]])
    try:
        with patch.object(model.base_model, "_encode_images",
                          return_value=torch.randn(1, 1, 4)):
            losses = model(ids, labels, mask, image_paths=["image.png", None])
    finally:
        for handle in handles:
            handle.remove()
    assert losses.main_count == 6
    assert losses.mtp_counts == [4, 2]
    for k, (args, output) in enumerate(captured, start=1):
        assert args[0].shape == (2, 4 - k, 4)
        torch.testing.assert_close(args[1], model.base_model.embed_tokens(ids)[:, k:])
        expected = F.cross_entropy(output[0].flatten(0, 1), labels[:, k:].flatten())
        torch.testing.assert_close(losses.mtp[k - 1], expected)
    torch.testing.assert_close(captured[1][0][0], captured[0][1][2][:, :-1])
    # A draft-only loss trains the backbone and both new blocks.
    losses.mtp[-1].backward()
    for weight in (
        model.base_model.text_token_projection.weight,
        model.base_model.llm.transformer[0].mqa.qkv.weight,
        model.base_model.llm.classifier.weight,
        model.base_model.seperation_token,
        *(module.input_projection.weight for module in model.mtp_modules),
    ):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.base_model.vision_model.parameters())


def test_short_sequences_and_document_boundaries(tmp_path):
    model, _ = make_trainer(tmp_path)
    short = model(**prepare_batch(torch.tensor([[2, 1]]), "cpu"))
    assert short.mtp_counts == [0, 0]
    assert torch.isfinite(short.objective)
    # EOS can be predicted but cannot be an intervening conditioning token.
    batch = prepare_batch(torch.tensor([[2, 4, 1, 2, 5, 1]]), "cpu")
    losses = model(**batch)
    assert losses.mtp_counts == [2, 0]
    losses.objective.backward()
    with pytest.raises(ValueError, match="no target"):
        model(torch.tensor([[2]]), torch.tensor([[-100]]))


def test_lora_mode_keeps_new_draft_blocks_trainable(tmp_path):
    model, _ = make_trainer(tmp_path, fine_tuning=True)
    assert not model.base_model.llm.classifier.weight.requires_grad
    assert not model.base_model.llm.transformer[0].mqa.qkv.weight.requires_grad
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    base_weight = model.base_model.llm.transformer[0].mqa.qkv.weight.detach().clone()
    draft_weight = model.mtp_modules[0].input_projection.weight.detach().clone()
    metrics = train_mtp.run_epoch(model, [torch.tensor([[2, 4, 5, 1]])], "cpu", optimizer)
    torch.testing.assert_close(model.base_model.llm.transformer[0].mqa.qkv.weight, base_weight)
    assert not torch.equal(model.mtp_modules[0].input_projection.weight, draft_weight)
    assert model.base_model.llm.transformer[0].mqa.lora_qkv_a.weight.grad is not None
    assert metrics["mtp_1_tokens"] == 2
    parameters = optimizer.param_groups[0]["params"]
    assert len(parameters) == len({id(p) for p in parameters})


@pytest.mark.parametrize("with_image", [False, True])
@torch.inference_mode()
def test_vlm_hidden_states_align_in_cached_generation(tmp_path, with_image):
    base, _ = tiny_model(tmp_path)
    base.eval()
    ids = torch.tensor([[2, 4, 5, 6]])
    image_kwargs = {"image_tokens": torch.randn(1, 1, 4)} if with_image else {}
    logits, _, expected = base(ids, output_hidden_states=True, **image_kwargs)
    assert expected.shape == (1, 4, 4)
    torch.testing.assert_close(logits, base.llm.classifier(base.llm.output_norm(expected)))
    _, _, cache = base(ids[:, :2], use_cache=True, **image_kwargs)
    actual_logits, _, hidden, cache = base(
        ids[:, 2:], past_key_values=cache, use_cache=True, output_hidden_states=True,
    )
    torch.testing.assert_close(actual_logits, logits[:, 2:])
    torch.testing.assert_close(hidden, expected[:, 2:])
    assert cache.text_length == 4


@pytest.mark.parametrize("paired", [False, True])
def test_main_trains_saves_and_initializes_from_both_checkpoint_types(tmp_path, paired):
    base, config = tiny_model(tmp_path / "model")
    flags = [
        "--device", "cpu", "--embeddings_path", config["word_embeddings_tensor"],
        "--output_dir", str(tmp_path / "output"), "--num_layer", "1",
        "--max_context_length", "6", "--projection_dim", "4", "--head_dim", "2",
        "--q_head", "2", "--kv_head", "1", "--expansion_factor", "2",
        "--dropout_ratio", "0", "--mtp_depth", "2", "--max_steps", "1",
        "--inference_every", "1", "--epochs", "2",
    ]
    if paired:
        for folder in ("train", "val"):
            make_pair(tmp_path / "data" / folder, "one", [2, 4, 5, 1])
        flags += ["--data_root", str(tmp_path / "data"), "--train_folders", "train",
                  "--val_folders", "val"]
    else:
        for split in ("train", "val"):
            np.save(tmp_path / f"{split}.npy", np.array([[2, 4, 5, 1]]))
        flags += ["--train_path", str(tmp_path / "train.npy"),
                  "--val_path", str(tmp_path / "val.npy")]
    args = train_mtp.parse_args(flags)
    special_ids = dict(bos_token_id=2, eos_token_id=1, pad_token_id=0)
    tokenizer = SimpleNamespace(**special_ids, get_vocab=lambda: {str(i): i for i in range(16)})
    before = base.llm.classifier.weight.detach().clone()
    with (
        patch("train_mtp.parse_args", return_value=args),
        patch("train_mtp.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("train_mtp.VLM", return_value=base),
        patch("train_mtp.print_image_inference") as preview,
    ):
        train_mtp.main()
    assert preview.call_count == (2 if paired else 0)
    assert not torch.equal(base.llm.classifier.weight, before)
    checkpoint_path = tmp_path / "output/latest.pt"
    saved = torch.load(checkpoint_path, weights_only=True)
    assert saved["mtp_config"] == {"mtp_depth": 2, "mtp_loss_weight": 0.3}
    assert saved["epoch"] == 2
    assert saved["metrics"]["mtp_2_tokens"] == 1
    assert saved["optimizer_state_dict"]["state"]
    assert saved["scheduler_state_dict"]["last_epoch"] == 2
    restored, _ = make_trainer(tmp_path / "restored")
    restored.base_model.word_embeddings_tensor.copy_(base.word_embeddings_tensor)
    train_mtp.load_initial_checkpoint(checkpoint_path, restored, special_ids)
    for key, value in restored.mtp_modules.state_dict().items():
        torch.testing.assert_close(value, saved["mtp_state_dict"][key])
    assert restored.mtp_modules[0].classifier is restored.base_model.llm.classifier
    # generate.py loads exactly these base keys, ignoring the draft extras.
    base.load_state_dict(saved["model_state_dict"], strict=True)
    base.eval()
    restored.eval()
    ids = torch.tensor([[2, 4, 5]])
    torch.testing.assert_close(base(ids)[0], restored.base_model(ids)[0])
    base_path = tmp_path / "base.pt"
    torch.save({key: saved[key] for key in ("model_state_dict", "special_token_ids")}, base_path)
    train_mtp.load_initial_checkpoint(base_path, restored, special_ids)
    restored.mtp_depth = 1
    with pytest.raises(ValueError, match="mtp_depth"):
        train_mtp.load_initial_checkpoint(checkpoint_path, restored, special_ids)


def test_validation_does_not_update_parameters_or_call_preview(tmp_path):
    model, _ = make_trainer(tmp_path)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    callback = Mock()
    metrics = train_mtp.run_epoch(model, [torch.tensor([[2, 4, 5, 1]])], "cpu",
                                   inference_every=1, inference_callback=callback)
    callback.assert_not_called()
    assert metrics["mtp_2_tokens"] == 1
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before[name])
        assert param.grad is None
