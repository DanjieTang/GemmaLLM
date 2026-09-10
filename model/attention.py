"""Gated attention with grouped KV heads, LoRA, and KV caching."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .rope import ROPEEmbedding


class Attention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 head_dim: int,
                 q_head: int,
                 kv_head: int,
                 embedding: ROPEEmbedding,
                 lora_rank: int = 16,
                 lora_alpha: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.q_head = q_head
        self.kv_head = kv_head
        self.embedding = embedding
        self.qkv = nn.Linear(hidden_dim, (q_head+kv_head*2)*head_dim)
        self.o = nn.Linear(q_head*head_dim, hidden_dim)
        self.scaler = 1/math.sqrt(head_dim)

        # LoRA
        self.lora_scale = lora_alpha / lora_rank
        self.lora_qkv_a = nn.Linear(hidden_dim, lora_rank)
        self.lora_qkv_b = nn.Linear(lora_rank, (q_head+kv_head*2)*head_dim)
        self.lora_o_a = nn.Linear(q_head*head_dim, lora_rank)
        self.lora_o_b = nn.Linear(lora_rank, hidden_dim)

        if q_head != kv_head:
            # If we are using multi query attention
            assert q_head % kv_head == 0
            self.multi_query_attention = True
            self.q_kv_scale = q_head//kv_head
        else:
            self.multi_query_attention = False

        # Gated attention from Qwen paper
        # Each gate gets one gate score as recommended in the paper
        self.gate = nn.Linear(hidden_dim, q_head)

    def forward(
        self,
        tensor: torch.Tensor,
        attention_mask: torch.Tensor = None,
        fine_tuning: bool = False,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch_size, seq_len, hid_dim = tensor.shape
        pre_norm_tensor = tensor # Used to calculate gate score

        qkv_tensor = self.qkv(tensor)
        if fine_tuning:
            lora_tensor = self.lora_qkv_a(tensor)
            lora_tensor = self.lora_qkv_b(lora_tensor)
            lora_tensor = lora_tensor * self.lora_scale
            qkv_tensor = lora_tensor + qkv_tensor
        query, key, value = qkv_tensor.split([self.head_dim*self.q_head, self.head_dim*self.kv_head, self.head_dim*self.kv_head], dim=-1)

        query = query.view(batch_size, seq_len, self.q_head, self.head_dim)
        key = key.view(batch_size, seq_len, self.kv_head, self.head_dim)
        value = value.view(batch_size, seq_len, self.kv_head, self.head_dim)

        # Switch to batch_size, head, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            if not use_cache:
                raise ValueError("past_key_value requires use_cache=True.")
            past_key, past_value = past_key_value
            # Check past key and value tensor KV caches have the correct shape [batch_size, kv_head, sequence_length, head_dim]
            if (len(past_key.shape) != 4 or past_key.shape != past_value.shape
                    or past_key.shape[:2] != (batch_size, self.kv_head)
                    or past_key.shape[-1] != self.head_dim):
                raise ValueError("Cached keys and values have an incompatible shape.")
            past_length = past_key.shape[2]

        # Apply ROPE
        query = self.embedding(query, position_offset=past_length)
        key = self.embedding(key, position_offset=past_length)
        if past_key_value is not None:
            # RoPE can promote keys to float32 under mixed-precision autocast.
            if (past_key.device != key.device or past_value.device != value.device
                    or past_key.dtype != key.dtype or past_value.dtype != value.dtype):
                raise ValueError("Cached keys and values must match the current device and dtype.")
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        present_key_value = (key, value)

        if self.multi_query_attention:
            # Store only the original KV heads, expanding them for attention.
            key = torch.repeat_interleave(key, self.q_kv_scale, dim=1)
            value = torch.repeat_interleave(value, self.q_kv_scale, dim=1)

        # Classic self attention
        attention_raw = torch.matmul(query, key.transpose(2, 3))
        attention_scaled = attention_raw * self.scaler
        if attention_mask != None:
            attention_scaled += attention_mask
        attention_score = torch.softmax(attention_scaled, dim=-1)
        value = torch.matmul(attention_score, value)

        # Reshape back to batch_size, seq_len, hid_dim
        value = value.transpose(1, 2).contiguous()
        value = value.view(batch_size, seq_len, hid_dim)

        # Gated attention from Qwen paper
        gate_score = self.gate(pre_norm_tensor)
        gate_score = F.sigmoid(gate_score)
        gate_score = gate_score.repeat_interleave(self.head_dim, dim=-1)
        value = value * gate_score

        # Output layer
        output = self.o(value)
        if fine_tuning:
            lora_tensor = self.lora_o_a(value)
            lora_tensor = self.lora_o_b(lora_tensor)
            lora_tensor = lora_tensor * self.lora_scale
            output = lora_tensor + output

        if use_cache:
            return output, present_key_value
        return output
