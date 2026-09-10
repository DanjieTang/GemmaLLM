from unittest.mock import patch

import pytest
import torch

from generate import generate
from test_annotation_training import tiny_model


@pytest.fixture(scope="module", autouse=True)
def single_threaded_tensors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("fine_tuning", [False, True])
@pytest.mark.parametrize("use_moe", [False, True])
@pytest.mark.parametrize("chunks", [(3, 1, 1, 1), (2, 2, 2)])
@torch.inference_mode()
def test_vlm_cache_matches_full_forward(tmp_path, images, fine_tuning, use_moe, chunks):
    torch.manual_seed(7)
    model, _ = tiny_model(tmp_path, use_moe)
    model.eval()
    model.fine_tuning = fine_tuning
    ids = torch.tensor([[2, 4, 5, 6, 7, 8], [2, 8, 7, 6, 5, 4]])
    image_tokens = torch.randn(2, 1, 4) if images else None
    expected, _ = model(ids, image_tokens=image_tokens)
    cache = None
    position = 0
    outputs = []
    for length in chunks:
        previous_cache = cache
        logits, _, cache = model(
            ids[:, position:position + length],
            image_tokens=image_tokens if cache is None else None,
            past_key_values=cache, use_cache=True,
        )
        outputs.append(logits)
        position += length
        assert cache.text_length == position
        for key, value in cache.past_key_values:
            assert key.shape == value.shape == (2, 1, position + (2 if images else 0), 2)
        if previous_cache is not None:
            for (old_key, old_value), (key, value) in zip(
                previous_cache.past_key_values, cache.past_key_values,
            ):
                torch.testing.assert_close(key[:, :, :-length], old_key)
                torch.testing.assert_close(value[:, :, :-length], old_value)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="max_context_length"):
        model(ids[:, :1], past_key_values=cache, use_cache=True)


@torch.inference_mode()
def test_generation_reuses_image_and_text_cache(tmp_path):
    torch.manual_seed(7)
    model, _ = tiny_model(tmp_path)
    # Ensure greedy decoding runs all the way to the text context boundary.
    model.llm.classifier.bias[1] = -100
    lengths = []
    hook = model.llm.register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    try:
        with patch.object(model, "_encode_images", return_value=torch.zeros(1, 1, 4)) as encode:
            expected = generate(model, "image.png", 2, 1, max_new_tokens=10, use_cache=False)
            assert lengths == [3, 4, 5, 6, 7, 8]
            lengths.clear()
            for _ in range(2):
                actual = generate(model, "image.png", 2, 1, max_new_tokens=10)
                assert actual == expected
            assert lengths == [3, 1, 1, 1, 1, 1] * 2
            assert encode.call_count == 3
    finally:
        hook.remove()
    assert model.training
    assert not model.vision_model.training


@torch.inference_mode()
def test_image_paths_are_encoded_only_on_prefill(tmp_path):
    model, _ = tiny_model(tmp_path)
    model.eval()
    ids = torch.tensor([[2, 4, 5]])
    with patch.object(model, "_encode_images", return_value=torch.zeros(1, 1, 4)) as encode:
        _, _, cache = model(ids[:, :2], image_paths=["image.png"], use_cache=True)
        actual, _, _ = model(ids[:, 2:], past_key_values=cache, use_cache=True)
        encode.assert_called_once()
    expected, _ = model(ids, image_tokens=torch.zeros(1, 1, 4))
    torch.testing.assert_close(actual, expected[:, 2:])


@torch.inference_mode()
def test_vlm_cache_supports_autocast(tmp_path):
    model, _ = tiny_model(tmp_path)
    model.eval()
    ids = torch.tensor([[2, 4, 5, 6]])
    image_tokens = torch.randn(1, 1, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = model(ids, image_tokens=image_tokens)
        _, _, cache = model(ids[:, :2], image_tokens=image_tokens, use_cache=True)
        actual, _, _ = model(ids[:, 2:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual, expected[:, 2:], atol=0.02, rtol=0.02)


@pytest.mark.parametrize("invalid", ["disabled", "raw_cache", "image_paths", "image_tokens", "batch"])
@torch.inference_mode()
def test_invalid_vlm_cache_usage_is_rejected(tmp_path, invalid):
    model, _ = tiny_model(tmp_path)
    model.eval()
    ids = torch.tensor([[2, 4]])
    _, _, cache = model(ids, use_cache=True)
    kwargs = dict(past_key_values=cache, use_cache=True)
    if invalid == "disabled":
        kwargs["use_cache"] = False
    elif invalid == "raw_cache":
        kwargs["past_key_values"] = cache.past_key_values
    elif invalid == "image_paths":
        kwargs["image_paths"] = ["image.png"]
    elif invalid == "image_tokens":
        kwargs["image_tokens"] = torch.zeros(1, 1, 4)
    elif invalid == "batch":
        ids = ids.expand(2, -1)
    with pytest.raises(ValueError):
        model(ids, **kwargs)


@torch.inference_mode()
def test_cache_rejects_padding_and_mixed_image_batches(tmp_path):
    model, _ = tiny_model(tmp_path)
    ids = torch.tensor([[2, 4], [2, 0]])
    with pytest.raises(ValueError, match="unpadded"):
        model(ids, attention_mask=ids != 0, use_cache=True)
    with pytest.raises(ValueError, match="all samples"):
        model(ids, image_paths=["image.png", None], use_cache=True)
