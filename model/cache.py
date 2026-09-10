"""Shared decoder and vision-language KV cache types."""

from dataclasses import dataclass

import torch


# Each layer stores rotated keys and values as [batch, kv_heads, sequence, head_dim].
KVCache = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[KVCache, ...]


@dataclass(frozen=True)
class VLMCache:
    """Decoder KV tensors and text length, excluding the optional image prefix."""

    past_key_values: PastKeyValues
    text_length: int
