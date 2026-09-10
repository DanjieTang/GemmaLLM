"""Transformer decoder layer combining attention and feed-forward blocks."""

import torch
import torch.nn as nn

from .attention import Attention
from .cache import KVCache
from .feed_forward import FeedForward
from .moe import MOE
from .rope import ROPEEmbedding


class LLMLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 head_dim: int,
                 q_head: int,
                 kv_head: int,
                 embedding: ROPEEmbedding,
                 expansion_factor: int = 4,
                 dropout_ratio: float = 0.1,
                 use_moe: bool = False,
                 num_experts: int = 8,
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 device: str = "mps"):
        super().__init__()
        self.use_moe = use_moe

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mqa = Attention(hidden_dim, head_dim, q_head, kv_head, embedding, lora_rank=lora_rank, lora_alpha=lora_alpha)

        self.norm2 = nn.LayerNorm(hidden_dim)
        if self.use_moe:
            self.moe = MOE(hidden_dim, num_experts=num_experts, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio, lora_rank=lora_rank, lora_alpha=lora_alpha, device=device)
        else:
            self.ffn = FeedForward(hidden_dim, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.device = device

    def forward(
        self,
        tensor: torch.Tensor,
        attention_mask: torch.Tensor = None,
        fine_tuning: bool = False,
        valid_token_mask: torch.Tensor | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ):
        skip_connection = tensor
        tensor = self.norm1(tensor)
        attention_output = self.mqa(
            tensor, attention_mask=attention_mask, fine_tuning=fine_tuning,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        if use_cache:
            tensor, present_key_value = attention_output
        else:
            tensor = attention_output
        tensor += skip_connection

        skip_connection = tensor
        tensor = self.norm2(tensor)
        if self.use_moe:
            tensor, load_balancing_loss = self.moe(
                tensor, fine_tuning=fine_tuning, valid_token_mask=valid_token_mask
            )
        else:
            tensor = self.ffn(tensor, fine_tuning=fine_tuning)
            load_balancing_loss = torch.tensor(0.0, dtype=tensor.dtype, device=self.device)# If not using MoE, load-balancing loss is zero

        tensor += skip_connection

        if use_cache:
            return tensor, load_balancing_loss, present_key_value
        return tensor, load_balancing_loss
