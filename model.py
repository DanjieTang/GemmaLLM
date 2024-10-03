# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import DEVICE_ID

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, wandb: bool = False):
        super().__init__()
        self.sqrt_dim: float = 1 / math.sqrt(dim)
        self.eps: float = eps
        self.wandb: bool = wandb
        if wandb:
            self.scale = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def find_rms_value(self, tensor: torch.Tensor) -> torch.Tensor:
        norm_2 = tensor.norm(2, dim=-1)
        return norm_2 * self.sqrt_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()
        rms = self.find_rms_value(tensor)
        tensor = tensor / (rms.unsqueeze(-1) + self.eps)

        if self.wandb:
            tensor = tensor * self.scale
            tensor = tensor + self.bias

        return tensor

class ROPEEmbedding(nn.Module):
    def __init__(self, max_token: int, dim: int, theta: int):
        super().__init__()
        self.pos_emb = self.create_embedding(max_token, dim, theta)

    def create_embedding(self, max_token: int, dim: int, theta: int) -> torch.Tensor:
        tensor = torch.arange(0, dim // 2)
        tensor = torch.repeat_interleave(tensor, 2)
        tensor = -tensor * 2 / dim
        tensor = torch.pow(theta, tensor)

        index = torch.arange(max_token).float()
        tensor = torch.einsum("i, j -> ij", tensor, index)

        cos_matrix = tensor.cos()
        sin_matrix = tensor.sin()
        sin_matrix[0::2] *= -1

        pos_emb = torch.cat((cos_matrix, sin_matrix), dim=0)
        pos_emb = pos_emb.transpose(1, 0)
        pos_emb = nn.Parameter(pos_emb, requires_grad=False)

        return pos_emb

    def flip_for_sin(self, tensor: torch.Tensor) -> torch.Tensor:
        original_shape = tensor.shape
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1, 2)
        tensor = tensor[..., [1, 0]]
        tensor = tensor.reshape(original_shape)
        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        sequence_length = tensor.shape[2]

        tensor = torch.cat((tensor, self.flip_for_sin(tensor)), dim=-1)
        tensor = tensor * self.pos_emb[:sequence_length, :]
        cos, sin = tensor.chunk(chunks=2, dim=-1)
        tensor = cos + sin
        return tensor

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int, q_head: int, kv_head: int, embedding: ROPEEmbedding):
        super().__init__()
        self.head_dim = head_dim
        self.q_head = q_head
        self.kv_head = kv_head
        self.embedding = embedding
        self.qkv = nn.Linear(hidden_dim, (q_head + kv_head * 2) * head_dim)
        self.o = nn.Linear(q_head * head_dim, hidden_dim)
        self.scaler = 1 / math.sqrt(head_dim)

        if q_head != kv_head:
            assert q_head % kv_head == 0
            self.multi_query_attention = True
            self.q_kv_scale = q_head // kv_head
        else:
            self.multi_query_attention = False

    def forward(self, tensor: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, hid_dim = tensor.shape

        tensor = self.qkv(tensor)
        query, key, value = tensor.split(
            [self.head_dim * self.q_head, self.head_dim * self.kv_head, self.head_dim * self.kv_head], dim=-1
        )

        query = query.view(batch_size, seq_len, self.q_head, self.head_dim)
        key = key.view(batch_size, seq_len, self.kv_head, self.head_dim)
        value = value.view(batch_size, seq_len, self.kv_head, self.head_dim)

        if self.multi_query_attention:
            key = key.repeat_interleave(self.q_kv_scale, dim=-2)
            value = value.repeat_interleave(self.q_kv_scale, dim=-2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.embedding(query)
        key = self.embedding(key)

        attention_raw = torch.matmul(query, key.transpose(2, 3))
        attention_scaled = attention_raw * self.scaler
        if attention_mask is not None:
            attention_scaled += attention_mask
        attention_score = torch.softmax(attention_scaled, dim=-1)
        value = torch.matmul(attention_score, value)

        value = value.transpose(1, 2).contiguous()
        value = value.view(batch_size, seq_len, hid_dim)

        output = self.o(value)

        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, inner_size: int, dropout_ratio: float = 0.5):
        super().__init__()
        self.gate_and_up = nn.Linear(hidden_size, inner_size * 2)
        self.down = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.gate_and_up(tensor)
        gate, up = tensor.chunk(chunks=2, dim=-1)
        gate = F.gelu(gate, approximate="tanh")
        up = self.dropout(up)
        tensor = gate * up
        tensor = self.down(tensor)
        return tensor

class GemmaLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        inner_size: int,
        head_dim: int,
        q_head: int,
        kv_head: int,
        embedding: ROPEEmbedding,
        dropout_ratio: float = 0.5,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.mqa = MultiQueryAttention(hidden_dim, head_dim, q_head, kv_head, embedding)

        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, inner_size, dropout_ratio)

    def forward(self, tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        skip_connection = tensor
        tensor = self.norm1(tensor)
        tensor = self.mqa(tensor, attention_mask)
        tensor += skip_connection

        skip_connection = tensor
        tensor = self.norm2(tensor)
        tensor = self.ffn(tensor)
        tensor += skip_connection

        return tensor

class Gemma(nn.Module):
    def __init__(
        self,
        num_layer: int,
        vocab_size: int,
        max_token: int,
        hidden_dim: int,
        inner_size: int,
        head_dim: int,
        q_head: int = None,
        kv_head: int = None,
        dropout_ratio: float = 0.5,
        theta: int = 10000,
        projection_dim: int = None,
    ):
        super().__init__()
        self.embedding = ROPEEmbedding(max_token, head_dim, theta)
        self.num_layer = num_layer

        if projection_dim is not None:
            self.projection = True
            self.projection_matrix = nn.Linear(hidden_dim, projection_dim)
            hidden_dim = projection_dim
        else:
            self.projection = False

        if q_head is None:
            q_head = hidden_dim // head_dim

        if kv_head is None:
            kv_head = hidden_dim // head_dim

        if hidden_dim % (head_dim * q_head) != 0 or hidden_dim % (head_dim * kv_head) != 0:
            raise ValueError(
                "Error: hidden_dim or projection_dim (if specified) must be divisible by the product of the number of q or kv heads and the head dimension."
            )

        self.transformer = nn.ModuleList()
        for _ in range(self.num_layer):
            self.transformer.append(
                GemmaLayer(hidden_dim, inner_size, head_dim, q_head, kv_head, self.embedding, dropout_ratio)
            )
        self.output_norm = RMSNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.projection:
            tensor = self.projection_matrix(tensor)

        seq_len = tensor.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1).cuda(DEVICE_ID)
        for layer in self.transformer:
            tensor = layer(tensor, causal_mask)

        tensor = self.output_norm(tensor)

        tensor = self.classifier(tensor)
        return tensor
