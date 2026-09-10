import pytest
import torch

from model import LLM
from test import generate


@pytest.fixture(scope="module", autouse=True)
def single_threaded_tensors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_llm(kv_head=2, use_moe=False):
    torch.manual_seed(7)
    return LLM(
        num_layer=2, vocabulary_size=23, max_context_length=8,
        hidden_dim=16, head_dim=4, q_head=4, kv_head=kv_head,
        expansion_factor=2, dropout_ratio=0.0, lora_rank=2,
        use_moe=use_moe, num_experts=2, device="cpu",
    ).eval()


@pytest.mark.parametrize("kv_head", [1, 2, 4])
@pytest.mark.parametrize("fine_tuning", [False, True])
@pytest.mark.parametrize("use_moe", [False, True])
@pytest.mark.parametrize("chunks", [(3, 1, 1, 1), (2, 2, 2)])
@torch.inference_mode()
def test_cached_logits_match_full_forward(kv_head, fine_tuning, use_moe, chunks):
    llm = make_llm(kv_head, use_moe)
    embeddings = torch.randn(2, 6, 16)
    mask = torch.full((6, 6), float("-inf")).triu(1)
    expected, _ = llm(embeddings, mask, fine_tuning)

    cache = None
    outputs = []
    position = 0
    for length in chunks:
        previous_cache = cache
        logits, _, cache = llm(
            embeddings[:, position:position + length],
            fine_tuning=fine_tuning, past_key_values=cache, use_cache=True,
        )
        outputs.append(logits)
        position += length
        assert len(cache) == llm.num_layer
        for key, value in cache:
            assert key.shape == value.shape == (2, kv_head, position, 4)
        if previous_cache is not None:
            for (old_key, old_value), (key, value) in zip(previous_cache, cache):
                assert old_key.shape[2] == position - length
                torch.testing.assert_close(key[:, :, :-length], old_key)
                torch.testing.assert_close(value[:, :, :-length], old_value)

    torch.testing.assert_close(torch.cat(outputs, dim=1), expected,
                               atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_generation_matches_baseline_and_only_processes_new_tokens():
    llm = make_llm()
    embeddings = torch.nn.Embedding(23, 16).eval()
    prompt = torch.tensor([[1, 2, 3]])
    mask = torch.full((8, 8), float("-inf")).triu(1)
    expected = generate(llm, embeddings, prompt, 5, mask, use_cache=False)
    lengths = []
    hook = llm.register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1])
    )
    try:
        actual = generate(llm, embeddings, prompt, 5, mask)
        # A separate generation must prefill a fresh cache.
        repeated = generate(llm, embeddings, prompt, 5, mask)
    finally:
        hook.remove()
    assert lengths == [3, 1, 1, 1, 1] * 2
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)


def test_uncached_forward_preserves_training_gradients():
    llm = make_llm().train()
    embeddings = torch.randn(2, 4, 16, requires_grad=True)
    logits, loss = llm(embeddings, torch.full((4, 4), float("-inf")).triu(1), False)
    (logits.square().mean() + loss).backward()
    for gradient in (embeddings.grad, llm.transformer[0].mqa.qkv.weight.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


@torch.inference_mode()
def test_cache_supports_autocast():
    llm = make_llm()
    embeddings = torch.randn(1, 6, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = llm(embeddings)
        _, _, cache = llm(embeddings[:, :4], use_cache=True)
        actual, _, _ = llm(embeddings[:, 4:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(actual, expected[:, 4:], atol=0.02, rtol=0.02)


@torch.inference_mode()
def test_cache_context_boundary_and_explicit_chunk_mask():
    llm = make_llm()
    embeddings = torch.randn(1, 8, 16)
    mask = torch.full((8, 8), float("-inf")).triu(1)
    expected, _ = llm(embeddings, mask)
    _, _, cache = llm(embeddings[:, :6], use_cache=True)
    logits, _, cache = llm(embeddings[:, 6:], mask[6:, :],
                           past_key_values=cache, use_cache=True)
    torch.testing.assert_close(logits, expected[:, 6:])
    with pytest.raises(ValueError, match="max_context_length"):
        llm(embeddings[:, :1], past_key_values=cache, use_cache=True)


@pytest.mark.parametrize("invalid", ["disabled", "layers", "length", "batch", "dtype", "mask"])
@torch.inference_mode()
def test_invalid_cache_is_rejected(invalid):
    llm = make_llm()
    embeddings = torch.randn(1, 3, 16)
    _, _, cache = llm(embeddings, use_cache=True)
    kwargs = {"use_cache": True}
    if invalid == "disabled":
        kwargs["use_cache"] = False
    elif invalid == "layers":
        cache = cache[:1]
    elif invalid == "length":
        cache = (tuple(t[:, :, :1] for t in cache[0]), cache[1])
    elif invalid == "batch":
        cache = tuple(tuple(t.expand(2, -1, -1, -1) for t in pair) for pair in cache)
    elif invalid == "dtype":
        cache = tuple(tuple(t.double() for t in pair) for pair in cache)
    elif invalid == "mask":
        kwargs["causal_mask"] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        llm(embeddings[:, :1], past_key_values=cache, **kwargs)
