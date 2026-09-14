import pytest
import torch

from model import LLM


@pytest.fixture(scope="module", autouse=True)
def single_threaded_tensors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_llm():
    torch.manual_seed(7)
    return LLM(
        num_layer=2, vocabulary_size=23, max_context_length=8,
        hidden_dim=16, head_dim=4, q_head=4, kv_head=2,
        expansion_factor=2, dropout_ratio=0.0, lora_rank=2, device="cpu",
    )


@pytest.mark.parametrize("use_cache", [False, True])
def test_optional_hidden_states_preserve_existing_outputs_and_gradients(use_cache):
    llm = make_llm()
    embeddings = torch.randn(2, 4, 16, requires_grad=True)
    with torch.no_grad():
        expected = llm(embeddings, use_cache=use_cache)
    assert len(expected) == (3 if use_cache else 2)
    final_outputs = []
    hook = llm.transformer[-1].register_forward_hook(
        lambda module, args, output: final_outputs.append(output[0])
    )
    try:
        actual = llm(embeddings, use_cache=use_cache, output_hidden_states=True)
    finally:
        hook.remove()

    assert len(actual) == (4 if use_cache else 3)
    hidden = actual[2]
    assert hidden.shape == (2, 4, 16)
    assert hidden is final_outputs[0]
    torch.testing.assert_close(actual[:2], expected[:2])
    if use_cache:
        torch.testing.assert_close(actual[-1], expected[-1])

    hidden.square().mean().backward()
    for grad in (embeddings.grad, llm.transformer[0].mqa.qkv.weight.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    assert llm.output_norm.weight.grad is None
    assert llm.classifier.weight.grad is None


@torch.inference_mode()
def test_cached_hidden_states_cover_only_new_tokens():
    llm = make_llm().eval()
    embeddings = torch.randn(2, 6, 16)
    expected_logits, _, expected_hidden = llm(embeddings, output_hidden_states=True)
    # A cache created without requesting hidden states can be reused with them.
    _, _, cache = llm(embeddings[:, :3], use_cache=True)
    logits, _, hidden, cache = llm(
        embeddings[:, 3:5], past_key_values=cache,
        use_cache=True, output_hidden_states=True,
    )
    assert hidden.shape == (2, 2, 16)
    torch.testing.assert_close(hidden, expected_hidden[:, 3:5])
    torch.testing.assert_close(logits, expected_logits[:, 3:5])
    # The flag can also be disabled again without altering the cache contract.
    logits, _, cache = llm(embeddings[:, 5:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(logits, expected_logits[:, 5:])
    assert all(key.shape[2] == 6 for key, _ in cache)
