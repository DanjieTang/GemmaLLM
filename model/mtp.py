"""DeepSeek-style sequential multi-token prediction using this repo's decoder."""

import torch
import torch.nn as nn

from .cache import KVCache
from .llm_layer import LLMLayer
from .rope import ROPEEmbedding


class MTPModule(nn.Module):
    """One prediction depth with its own fusion layers and transformer block.

    Pass the main LLM's output_norm and classifier objects to share its output
    head. Token lookup stays with the caller: next_token_embeddings must come
    from the same embedding/projection path used to feed the main LLM.
    This implements the MTP structure from DeepSeek-V3 section 2.2 while using
    this repository's attention/FFN block, rather than DeepSeek's MLA block.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_context_length: int,
        *,
        output_norm: nn.Module,
        classifier: nn.Linear,
        expansion_factor: int = 4,
        head_dim: int = 64,
        q_head: int | None = None,
        kv_head: int | None = None,
        dropout_ratio: float = 0.1,
        theta: int = 10000,
        use_moe: bool = False,
        num_experts: int = 8,
        load_balancing_loss_weight: float = 1e-2,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        norm_eps: float = 1e-6,
        device: str = "cpu",
    ):
        super().__init__()
        if hidden_dim <= 0 or head_dim <= 0 or head_dim % 2:
            raise ValueError("hidden_dim must be positive and head_dim positive and even.")
        q_head = hidden_dim // head_dim if q_head is None else q_head
        kv_head = q_head if kv_head is None else kv_head
        if (q_head <= 0 or kv_head <= 0 or hidden_dim != q_head * head_dim
                or q_head % kv_head):
            raise ValueError(
                "Require hidden_dim == q_head * head_dim "
                "and q_head divisible by kv_head."
            )
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive.")
        if classifier.in_features != hidden_dim:
            raise ValueError("Shared classifier input size must equal hidden_dim.")
        if use_moe and num_experts < 2:
            raise ValueError("The top-two MoE block requires at least two experts.")

        self.hidden_dim = hidden_dim
        self.max_context_length = max_context_length
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.hidden_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.token_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.input_projection = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.transformer = LLMLayer(
            hidden_dim, head_dim, q_head, kv_head,
            ROPEEmbedding(max_context_length, head_dim=head_dim, theta=theta),
            expansion_factor=expansion_factor, dropout_ratio=dropout_ratio,
            use_moe=use_moe, num_experts=num_experts,
            lora_rank=lora_rank, lora_alpha=lora_alpha, device=device,
        )
        # These are references to the main model's modules, not copies.
        self.output_norm = output_norm
        self.classifier = classifier

    def forward(
        self,
        hidden_states: torch.Tensor,
        next_token_embeddings: torch.Tensor,
        *,
        fine_tuning: bool = False,
        valid_token_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, KVCache]
    ):
        """Return logits, weighted MoE loss, hidden states, and optionally cache.

        Both inputs have shape [batch, new_tokens, hidden_dim]. At depth k,
        pair h_i^(k-1) with Emb(t_(i+k)) to predict t_(i+k+1). The caller
        aligns/slices these inputs; this method does not shift them internally.
        Output logits have shape [batch, new_tokens, vocabulary_size]. Raw
        output hidden states retain gradients and can feed the next MTP depth.

        valid_token_mask marks nonempty, right-padded batches for MoE routing;
        exclude padded targets separately when calculating prediction loss.
        Each MTP module owns a separate KV cache, never the main LLM's cache.
        Cached calls must contain only new, unpadded input pairs.
        """
        if (hidden_states.ndim != 3
                or hidden_states.shape != next_token_embeddings.shape
                or hidden_states.shape[-1] != self.hidden_dim):
            raise ValueError(
                "Both inputs must have matching [batch, new_tokens, hidden_dim] shapes."
            )
        if hidden_states.device != next_token_embeddings.device:
            raise ValueError("Hidden states and token embeddings must be on the same device.")
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size < 1 or seq_len < 1:
            raise ValueError("MTP requires a nonempty batch and sequence.")
        if past_key_value is not None and not use_cache:
            raise ValueError("past_key_value requires use_cache=True.")
        past_length = 0
        if past_key_value is not None:
            key, value = past_key_value
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError("Cached keys and values must have matching 4D shapes.")
            past_length = key.shape[2]
        total_length = past_length + seq_len
        if total_length > self.max_context_length:
            raise ValueError("Sequence including cached tokens exceeds max_context_length.")

        if valid_token_mask is not None:
            if valid_token_mask.shape != (batch_size, seq_len):
                raise ValueError("valid_token_mask must match [batch, new_tokens].")
            valid_token_mask = valid_token_mask.to(
                device=hidden_states.device, dtype=torch.bool,
            )
            if not valid_token_mask.any() or (
                valid_token_mask[:, 1:] & ~valid_token_mask[:, :-1]
            ).any():
                raise ValueError("valid_token_mask must describe a nonempty, right-padded batch.")
            if use_cache and not valid_token_mask.all():
                raise ValueError("KV caching requires unpadded input pairs.")

        # [B, T, H] + [B, T, H] -> [B, T, 2H] -> [B, T, H].
        tensor = self.input_projection(torch.cat((
            self.hidden_norm(hidden_states),
            self.token_norm(next_token_embeddings),
        ), dim=-1))
        causal_mask = torch.full(
            (seq_len, total_length), float("-inf"),
            device=tensor.device, dtype=tensor.dtype,
        ).triu(diagonal=past_length + 1)
        layer_output = self.transformer(
            tensor, attention_mask=causal_mask, fine_tuning=fine_tuning,
            valid_token_mask=valid_token_mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        hidden_states, auxiliary_loss = layer_output[:2]
        auxiliary_loss = auxiliary_loss * self.load_balancing_loss_weight
        logits = self.classifier(self.output_norm(hidden_states))
        if use_cache:
            return logits, auxiliary_loss, hidden_states, layer_output[2]
        return logits, auxiliary_loss, hidden_states
