import io

import pytest
import torch
import torch.nn.functional as F

from model import LLM, MTPModule


@pytest.fixture(scope="module", autouse=True)
def single_threaded_tensors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(use_moe=False):
    torch.manual_seed(19)
    config = dict(
        hidden_dim=16, max_context_length=8, head_dim=4,
        q_head=4, kv_head=2, expansion_factor=2, dropout_ratio=0.0,
        use_moe=use_moe, num_experts=2, lora_rank=2, device="cpu",
    )
    llm = LLM(num_layer=2, vocabulary_size=23, **config)
    return torch.nn.ModuleDict({
        "llm": llm,
        "token_embedding": torch.nn.Embedding(23, 16),
        "mtp": torch.nn.ModuleList([
            MTPModule(output_norm=llm.output_norm, classifier=llm.classifier,
                      **config)
            for _ in range(2)
        ]),
    })


@pytest.mark.parametrize("use_moe", [False, True])
def test_two_depths_train_backbone_embeddings_and_shared_head(use_moe):
    model = make_model(use_moe)
    tokens = torch.randint(23, (2, 6))
    embeddings = model["token_embedding"](tokens)
    _, _, hidden = model["llm"](embeddings, output_hidden_states=True)
    for depth, mtp in enumerate(model["mtp"], start=1):
        assert mtp.classifier is model["llm"].classifier
        assert mtp.output_norm is model["llm"].output_norm
        # At depth k, position i uses Emb(t_(i+k)) to predict t_(i+k+1).
        length = tokens.shape[1] - depth - 1
        logits, auxiliary_loss, hidden = mtp(
            hidden[:, :length], embeddings[:, depth:-1],
        )
        assert logits.shape == (2, length, 23)
        assert hidden.shape == (2, length, 16)
        assert torch.isfinite(auxiliary_loss)
    # Only the deepest prediction supplies a loss: its gradient must traverse
    # the first MTP module to reach the backbone and shared token embeddings.
    loss = F.cross_entropy(logits.reshape(-1, 23), tokens[:, 3:].reshape(-1))
    loss.backward()
    weights = [
        model["token_embedding"].weight,
        model["llm"].transformer[0].mqa.qkv.weight,
        model["llm"].classifier.weight,
        model["llm"].output_norm.weight,
    ]
    for mtp in model["mtp"]:
        weights.extend([
            mtp.hidden_norm.weight, mtp.token_norm.weight,
            mtp.input_projection.weight, mtp.transformer.mqa.qkv.weight,
        ])
    for weight in weights:
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert weight.grad.abs().sum() > 0
    assert model["mtp"][0].transformer is not model["mtp"][1].transformer
    params = list(model.parameters())
    assert sum(p is model["llm"].classifier.weight for p in params) == 1


@pytest.mark.parametrize("fine_tuning", [False, True])
@pytest.mark.parametrize("use_moe", [False, True])
@torch.inference_mode()
def test_cached_mtp_matches_full_forward_and_is_causal(use_moe, fine_tuning):
    mtp = make_model(use_moe)["mtp"][0].eval()
    hidden, embeddings = torch.randn(2, 6, 16), torch.randn(2, 6, 16)
    expected, _, expected_hidden = mtp(hidden, embeddings, fine_tuning=fine_tuning)
    cache = None
    logits_chunks, hidden_chunks = [], []
    position = 0
    for length in (2, 1, 3):
        stop = position + length
        logits, _, states, cache = mtp(
            hidden[:, position:stop], embeddings[:, position:stop],
            past_key_value=cache, use_cache=True, fine_tuning=fine_tuning,
        )
        assert cache[0].shape == cache[1].shape == (2, 2, stop, 4)
        assert states.shape == (2, length, 16)
        logits_chunks.append(logits)
        hidden_chunks.append(states)
        position = stop
    torch.testing.assert_close(torch.cat(logits_chunks, dim=1), expected)
    torch.testing.assert_close(torch.cat(hidden_chunks, dim=1), expected_hidden)

    changed_hidden, changed_embeddings = hidden.clone(), embeddings.clone()
    changed_hidden[:, 3:] = torch.randn_like(changed_hidden[:, 3:])
    changed_embeddings[:, 3:] = torch.randn_like(changed_embeddings[:, 3:])
    actual, _, _ = mtp(changed_hidden, changed_embeddings, fine_tuning=fine_tuning)
    torch.testing.assert_close(actual[:, :3], expected[:, :3])


@torch.inference_mode()
def test_padding_is_excluded_from_moe_routing_statistics():
    mtp = make_model(use_moe=True)["mtp"][0].eval()
    hidden, embeddings = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    expected, expected_loss, _ = mtp(hidden[:, :3], embeddings[:, :3])
    mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
    actual, loss, _ = mtp(hidden, embeddings, valid_token_mask=mask)
    torch.testing.assert_close(actual[:, :3], expected)
    torch.testing.assert_close(loss, expected_loss)


@torch.inference_mode()
def test_checkpoint_round_trip_preserves_outputs_and_sharing():
    model = make_model().eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    hidden, embeddings = torch.randn(1, 3, 16), torch.randn(1, 3, 16)
    expected = model["mtp"][0](hidden, embeddings)[0]
    actual = restored["mtp"][0](hidden, embeddings)[0]
    torch.testing.assert_close(actual, expected)
    assert restored["mtp"][0].classifier is restored["llm"].classifier


@torch.inference_mode()
def test_invalid_inputs_and_cache_usage_are_rejected():
    mtp = make_model()["mtp"][0].eval()
    hidden = torch.randn(1, 3, 16)
    with pytest.raises(ValueError, match="matching"):
        mtp(hidden, hidden[:, :2])
    with pytest.raises(ValueError, match="nonempty"):
        mtp(hidden[:, :0], hidden[:, :0])
    with pytest.raises(ValueError, match="right-padded"):
        mtp(hidden, hidden, valid_token_mask=torch.tensor([[True, False, True]]))
    with pytest.raises(ValueError, match="nonempty"):
        mtp(hidden, hidden, valid_token_mask=torch.zeros(1, 3, dtype=torch.bool))
    _, _, _, cache = mtp(hidden, hidden, use_cache=True)
    with pytest.raises(ValueError, match="use_cache"):
        mtp(hidden, hidden, past_key_value=cache)
    too_long = torch.randn(1, 6, 16)
    with pytest.raises(ValueError, match="max_context_length"):
        mtp(too_long, too_long, past_key_value=cache, use_cache=True)
