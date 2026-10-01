import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from model import VLM
from generate import generate
from lazy_dataloader import AnnotationDataset
from train import (
    iter_validation_texts, main, parse_args, print_image_inference, repeat_forever,
    run_epoch, run_inference_previews, split_text_prompt,
)
from test_annotation_training import make_pair, tiny_model
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
    ["--image_inference_every", "-1"],
    ["--text_inference_every", "-1"],
    ["--inference_max_new_tokens", "0"],
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


def test_inference_defaults_and_disable():
    assert parse_args([]).image_inference_every == 0
    assert parse_args([]).inference_max_new_tokens == 256
    assert parse_args(["--image_inference_every", "1000"]).image_inference_every == 1000
    assert parse_args([]).text_inference_every == 0
    assert parse_args(["--text_inference_every", "500"]).text_inference_every == 500


class PieceTokenizer:
    """Greedy longest-match tokenizer over fixed text pieces, for prompt splitting."""

    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0
    pieces = ["<pad>", "<eos>", "<bos>", "### Instruction\n", "add", "\n\n",
              "### Response\n", "x+y"]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids if not (skip_special_tokens and i < 3))

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            index = max((i for i in range(3, len(self.pieces))
                         if text.startswith(self.pieces[i])),
                        key=lambda i: len(self.pieces[i]))
            ids.append(index)
            text = text[len(self.pieces[index]):]
        return ids


def test_coding_prompt_ends_at_response_marker_and_text_uses_first_half():
    tokenizer = PieceTokenizer()
    coding = torch.tensor([2, 3, 4, 5, 6, 7, 1])
    assert split_text_prompt(coding, tokenizer, 64) == ([2, 3, 4, 5, 6], "x+y")
    # Without BOS and the marker (e.g. a Wikipedia window), prompt with the first half.
    text = torch.tensor([4, 5, 4, 5, 7])
    assert split_text_prompt(text, tokenizer, 64) == ([4, 5], "add\n\nx+y")
    assert split_text_prompt(text, tokenizer, 1) == ([4], "\n\nadd\n\nx+y")


def test_validation_texts_skip_images_and_repeat_without_caching(tmp_path):
    annotations = AnnotationDataset.__new__(AnnotationDataset)
    annotations.samples = [("tokens.npy", "image.png")]
    legacy = [(torch.tensor([2, 4]), "image.png"), (torch.tensor([2, 5]), "")]
    plain = [torch.tensor([2, 6])]
    texts = [ids.tolist() for ids in iter_validation_texts(
        torch.utils.data.ConcatDataset([annotations, legacy, plain]))]
    assert texts == [[2, 5], [2, 6]]
    repeated = repeat_forever(lambda: iter([1, 2]))
    assert [next(repeated) for _ in range(5)] == [1, 2, 1, 2, 1]
    assert next(repeat_forever(lambda: iter([])), None) is None


def test_previews_run_on_their_own_intervals():
    image, text = Mock(), Mock()
    for step in range(1, 7):
        run_inference_previews(step, 2, image, 3, text)
    assert [call.args[0] for call in image.call_args_list] == [2, 4, 6]
    assert [call.args[0] for call in text.call_args_list] == [3, 6]


@torch.inference_mode()
def test_text_generation_continues_prompt_without_images(tmp_path):
    model, _ = tiny_model(tmp_path)
    model.llm.classifier.bias[1] = -100  # Never stop early at EOS.
    lengths = []
    hook = model.llm.register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1]))
    try:
        with patch.object(model, "_encode_images") as encode:
            tokens = generate(model, None, 2, 1, max_new_tokens=10, prompt_ids=[2, 4, 5, 6])
            uncached = generate(model, None, 2, 1, max_new_tokens=10,
                                prompt_ids=[2, 4, 5, 6], use_cache=False)
        encode.assert_not_called()
    finally:
        hook.remove()
    # Text only: no image prefix, and the 6-token context allows three new tokens.
    assert lengths == [4, 1, 1, 4, 5, 6]
    assert len(tokens) == 3 and tokens == uncached
    with pytest.raises(ValueError, match="prompt_ids"):
        generate(model, None, 2, 1, prompt_ids=[2] * 7)


def test_main_previews_validation_images_across_epochs(tmp_path, capsys):
    model, _ = tiny_model(tmp_path / "model")
    for name in ("one", "two"):
        make_pair(tmp_path / "data/train", name, [2, 4, 1])
        make_pair(tmp_path / "data/val", name, [2, 5, 1])
    tokenizer = SimpleNamespace(
        bos_token_id=2, eos_token_id=1, pad_token_id=0,
        get_vocab=lambda: {str(i): i for i in range(16)},
        decode=Mock(return_value="A generated annotation."),
    )
    args = parse_args([
        "--data_root", str(tmp_path / "data"),
        "--train_folders", "train", "--val_folders", "val",
        "--output_dir", str(tmp_path / "output"), "--device", "cpu",
        "--epochs", "5", "--batch_size", "1", "--max_steps", "2",
        "--max_context_length", "6", "--image_inference_every", "3",
        "--inference_max_new_tokens", "2",
    ])
    preview_modes = []
    training_modes = []

    def record_forward(module, inputs, kwargs):
        if kwargs.get("use_cache"):
            preview_modes.append((module.training, torch.is_grad_enabled()))
            # The first decoding input must contain BOS only, without annotation tokens.
            if kwargs.get("past_key_values") is None:
                assert inputs[0].tolist() == [[2]]
        elif torch.is_grad_enabled():
            training_modes.append(module.training)

    handle = model.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        with patch("train.parse_args", return_value=args), \
             patch("train.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("train.VLM", return_value=model), \
             patch("train.print_image_inference", wraps=print_image_inference) as preview:
            main()
    finally:
        handle.remove()
    assert [call.args[3] for call in preview.call_args_list] == [3, 6, 9]
    paths = [call.args[2] for call in preview.call_args_list]
    assert set(paths) == {str(tmp_path / f"data/val/images/{name}.png")
                          for name in ("one", "two")}
    assert paths[0] == paths[2]
    assert all(call.args[4] == 2 for call in preview.call_args_list)
    assert preview_modes and all(mode == (False, False) for mode in preview_modes)
    assert training_modes == [True] * 10
    assert tokenizer.decode.call_count == 3
    assert all(len(call.args[0]) <= 2 for call in tokenizer.decode.call_args_list)
    output = capsys.readouterr().out
    for step, path in zip((3, 6, 9), paths):
        assert f"[Inference | iteration {step}] Image: {path}" in output
    assert output.count("Generated annotation: A generated annotation.") == 3


@pytest.mark.parametrize("training,image_inference_every", [(True, 0), (False, 1)])
def test_disabled_previews_and_validation_do_not_generate(tmp_path, training, image_inference_every):
    model, _ = tiny_model(tmp_path)
    batch = torch.tensor([[2, 4, 1]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if training else None
    callback = Mock()
    run_epoch(model, [batch], "cpu", optimizer, image_inference_every=image_inference_every,
              image_inference_callback=callback)
    callback.assert_not_called()


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
        "--image_inference_every", "0",
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
