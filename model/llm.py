"""Language model decoder and vocabulary classifier."""

import torch
import torch.nn as nn

from .cache import PastKeyValues
from .llm_layer import LLMLayer
from .rope import ROPEEmbedding


class LLM(nn.Module):
    def __init__(self,
                 num_layer: int,
                 vocabulary_size: int,
                 max_context_length: int,
                 hidden_dim: int,
                 expansion_factor: int = 4,
                 head_dim: int = 64,
                 q_head: int = None,
                 kv_head: int = None,
                 dropout_ratio: float = 0.1,
                 theta: int = 10000,
                 use_moe: bool = False,
                 num_experts=8,
                 load_balancing_loss_weight: float = 1e-2,
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 device: str = "mps"):
        super().__init__()
        self.embedding = ROPEEmbedding(max_context_length, head_dim=head_dim, theta=theta)
        self.num_layer = num_layer
        self.load_balancing_loss_weight = load_balancing_loss_weight

        if q_head == None:
            q_head = (hidden_dim // head_dim)

        if kv_head == None:
            kv_head = (hidden_dim // head_dim)

        if hidden_dim % (head_dim * q_head) != 0 or hidden_dim % (head_dim * kv_head):
            raise ValueError("Error: hidden_dim or projection_dim (if specified) must be divisible by the product of the number of q or kv heads and the head dimension.")

        self.transformer = nn.ModuleList()
        for _ in range(self.num_layer):
            self.transformer.append(LLMLayer(hidden_dim, head_dim, q_head, kv_head, self.embedding, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio, use_moe=use_moe, num_experts=num_experts, lora_rank=lora_rank, lora_alpha=lora_alpha, device=device))
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, vocabulary_size)
        self.device = device

    def forward(
        self,
        tensor: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        fine_tuning: bool = False,
        valid_token_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, PastKeyValues]:
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True.")
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != self.num_layer:
                raise ValueError("past_key_values must contain one entry per layer.")
            for key, value in past_key_values:
                if len(key.shape) != 4 or key.shape != value.shape:
                    raise ValueError("Cached keys and values must have matching 4D shapes.")
            past_length = past_key_values[0][0].shape[2]
            if any(key.shape[2] != past_length for key, _ in past_key_values): # Check every layer have same amount of cached token
                raise ValueError("All layers must have the same cached sequence length.")
        seq_len = tensor.shape[1]
        total_length = past_length + seq_len
        if seq_len < 1 or total_length > self.embedding.pos_emb.shape[0]:
            raise ValueError("Sequence including cached tokens must be within max_context_length.")
        if causal_mask is None:
            # Example: past_length=2, seq_len=3 -> shape (3, 5), diagonal=3.
            # Columns:  cached tokens | new tokens
            #                 C0  C1  |  N0    N1    N2
            # Row N0:          0   0  |   0  -inf  -inf
            # Row N1:          0   0  |   0     0  -inf
            # Row N2:          0   0  |   0     0     0
            # 0 allows attention; -inf blocks attention to future tokens.
            causal_mask = torch.full(
                (seq_len, total_length), float("-inf"),
                device=tensor.device, dtype=tensor.dtype,
            ).triu(diagonal=past_length + 1)
        elif causal_mask.shape[-2:] != (seq_len, total_length):
            # For the example above, the supplied mask must end in shape (3, 5).
            raise ValueError("causal_mask must end in [new_tokens, past + new_tokens].")

        # Track load-balancing across layers (only if MoE is used)
        load_balancing_sum = torch.tensor(0.0, device=self.device)

        present_key_values = []
        for index, layer in enumerate(self.transformer):
            layer_output = layer(
                tensor,
                attention_mask=causal_mask,
                fine_tuning=fine_tuning,
                valid_token_mask=valid_token_mask,
                past_key_value=None if past_key_values is None else past_key_values[index],
                use_cache=use_cache,
            )
            if use_cache:
                tensor, load_balancing_loss, present_key_value = layer_output
                present_key_values.append(present_key_value)
            else:
                tensor, load_balancing_loss = layer_output
            load_balancing_sum += load_balancing_loss

        load_balancing_loss = (load_balancing_sum / self.num_layer) * self.load_balancing_loss_weight

        # Classification
        tensor = self.output_norm(tensor)
        tensor = self.classifier(tensor)

        if use_cache:
            return tensor, load_balancing_loss, tuple(present_key_values)
        return tensor, load_balancing_loss
