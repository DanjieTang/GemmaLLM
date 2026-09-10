import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from model import VLM
from train import main, parse_args
from test_vlm import FakeVisionModel, FakeVisionProcessor


@pytest.mark.parametrize("flags", [
    ["--dropout_ratio", "-0.1"],
    ["--dropout_ratio", "1.1"],
    ["--dropout_ratio", "nan"],
    ["--theta", "0"],
    ["--num_experts", "0"],
    ["--use_moe", "true", "--num_experts", "1"],
    ["--load_balancing_loss_weight", "-1"],
    ["--load_balancing_loss_weight", "inf"],
    ["--lora_rank", "0"],
    ["--lora_alpha", "-1"],
    ["--fine_tuning", "maybe"],
])
def test_invalid_model_options_fail_early(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)


def test_boolean_flags_accept_bare_and_explicit_values():
    assert parse_args(["--use_moe", "--fine_tuning"]).use_moe
    assert parse_args(["--fine_tuning"]).fine_tuning
    args = parse_args(["--use_moe", "false", "--fine_tuning", "False"])
    assert not args.use_moe
    assert not args.fine_tuning


@pytest.mark.parametrize("use_moe", [False, True])
@pytest.mark.parametrize("fine_tuning", [False, True])
def test_main_trains_and_saves_all_model_settings(tmp_path, use_moe, fine_tuning):
    embeddings = tmp_path / "embeddings.pt"
    torch.save(torch.randn(16, 6), embeddings)
    flags = [
        "--device", "cpu", "--embeddings_path", str(embeddings),
        "--output_dir", str(tmp_path / "output"),
        "--num_layer", "1", "--max_context_length", "4",
        "--projection_dim", "4", "--head_dim", "2", "--q_head", "2",
        "--kv_head", "1", "--expansion_factor", "2",
        "--dropout_ratio", "0.2", "--theta", "1234",
        "--use_moe", str(use_moe), "--num_experts", "2",
        "--load_balancing_loss_weight", "0.07",
        "--fine_tuning", str(fine_tuning), "--lora_rank", "2",
        "--lora_alpha", "6", "--max_steps", "1",
    ]
    batch = {
        "input_ids": torch.tensor([[2, 4, 5], [2, 6, 0]]),
        "labels": torch.tensor([[4, 5, 1], [6, 1, -100]]),
        "attention_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "image_paths": [None, None],
    }
    tokenizer = SimpleNamespace(bos_token_id=2, eos_token_id=1, pad_token_id=0,
                                get_vocab=lambda: {str(i): i for i in range(16)})
    models = []
    initial_states = []

    def make_model(**kwargs):
        model = VLM(**kwargs)
        models.append(model)
        initial_states.append({name: value.clone()
                               for name, value in model.state_dict().items()})
        return model

    with patch("model.CLIPVisionModel.from_pretrained", return_value=FakeVisionModel()), \
         patch("model.CLIPImageProcessor.from_pretrained", return_value=FakeVisionProcessor()), \
         patch("train.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("train.prepare_annotation_dataset", return_value=([batch], [batch])), \
         patch("train.VLM", side_effect=make_model), \
         patch("train.parse_args", return_value=parse_args(flags)):
        main()

        saved = torch.load(tmp_path / "output/latest.pt", weights_only=True)
        config = saved["model_config"]
        # Check every constructor setting is persisted, except the runtime device.
        assert set(config) == set(inspect.signature(VLM).parameters) - {"device"}
        args = parse_args(flags)
        for name, value in config.items():
            expected = str(embeddings) if name == "word_embeddings_tensor" else getattr(args, name)
            assert value == expected
        assert torch.isfinite(torch.tensor(saved["loss"]))
        model = models[0]
        layer = model.llm.transformer[0]
        assert layer.mqa.lora_scale == 3
        experts = layer.moe.experts if use_moe else [layer.ffn]
        assert all(expert.lora_scale == 3 for expert in experts)
        assert all(expert.dropout.p == 0.2 for expert in experts)
        changed = {name for name, value in model.state_dict().items()
                   if not torch.equal(value, initial_states[0][name])}
        assert changed
        if fine_tuning:
            assert all("lora" in name for name in changed)
            assert all(parameter.requires_grad == ("lora" in name)
                       for name, parameter in model.named_parameters())
        else:
            assert "llm.classifier.weight" in changed

        # Initialize a second run from the checkpoint; a zero LR makes exact
        # equality after its forward/backward pass a check that loading worked.
        reload_flags = flags + ["--init_checkpoint", str(tmp_path / "output/latest.pt"),
                                "--output_dir", str(tmp_path / "reload"), "--lr", "0"]
        with patch("train.parse_args", return_value=parse_args(reload_flags)):
            main()
        for name, value in models[1].state_dict().items():
            torch.testing.assert_close(value, saved["model_state_dict"][name])
