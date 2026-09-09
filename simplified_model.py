import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROPEEmbedding(nn.Module):
    def __init__(self, max_context_length: int, head_dim: int = 64, theta: int = 10000):
        super().__init__()
        self.pos_emb = self.create_embedding(max_context_length, head_dim=head_dim, theta=theta)

    @staticmethod
    def create_embedding(max_context_length: int, head_dim: int = 64, theta: int = 10000) -> torch.Tensor:
        # Angles
        tensor = torch.arange(0, head_dim // 2)
        tensor = torch.repeat_interleave(tensor, 2)
        tensor = -tensor * 2 / head_dim
        tensor = torch.pow(theta, tensor)

        index = torch.arange(max_context_length).float() # This is the m in the formula
        tensor = torch.einsum("i, j -> ij", tensor, index)

        cos_matrix = tensor.cos()
        sin_matrix = tensor.sin()
        sin_matrix[0::2, :] *= -1 # Flipping sign for 0, 2, 4... row of sin matrix

        pos_emb = torch.cat((cos_matrix, sin_matrix), dim=0)
        pos_emb = pos_emb.transpose(1, 0)
        pos_emb = nn.Parameter(pos_emb, requires_grad=False)

        return pos_emb

    @staticmethod
    def flip_for_sin(tensor: torch.Tensor) -> torch.Tensor:
        B, H, S, D = tensor.shape
        tensor = tensor.view(B, H, S, D//2, 2) # Get to pairs
        tensor = tensor[:, :, :, :, [1, 0]].contiguous() # Swap
        tensor = tensor.view(B, H, S, D) # Get back to original shape
        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        sequence_length = tensor.shape[2] # Assuming we are using batch_size, head, sequence_length and dim

        tensor = torch.cat((tensor, self.flip_for_sin(tensor)), dim=-1)
        tensor = tensor * self.pos_emb[:sequence_length, :]
        cos, sin = tensor.chunk(chunks=2, dim=-1)
        tensor = cos + sin
        return tensor

class Attention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 head_dim: int,
                 num_head: int,
                 embedding: ROPEEmbedding,
                 max_context_length: int):
        super().__init__()
        self.head_dim = head_dim
        self.num_head = num_head
        self.embedding = embedding
        self.qkv = nn.Linear(hidden_dim, num_head*head_dim*3)
        self.o = nn.Linear(num_head*head_dim, hidden_dim)
        self.scaler = 1/math.sqrt(head_dim)

        # Causal mask, built once and sliced per forward pass
        mask = torch.full((max_context_length, max_context_length), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hid_dim = tensor.shape

        qkv_tensor = self.qkv(tensor)
        query, key, value = qkv_tensor.chunk(chunks=3, dim=-1)

        query = query.view(batch_size, seq_len, self.num_head, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_head, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_head, self.head_dim)

        # Switch to batch_size, head, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply ROPE
        query = self.embedding(query)
        key = self.embedding(key)

        # Classic self attention
        attention_raw = torch.matmul(query, key.transpose(2, 3))
        attention_scaled = attention_raw * self.scaler
        attention_scaled += self.causal_mask[:seq_len, :seq_len]
        attention_score = torch.softmax(attention_scaled, dim=-1)
        value = torch.matmul(attention_score, value)

        # Reshape back to batch_size, seq_len, hid_dim
        value = value.transpose(1, 2).contiguous()
        value = value.view(batch_size, seq_len, hid_dim)

        # Output layer
        output = self.o(value)

        return output

class FeedForward(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 expansion_factor: int = 4,
                 dropout_ratio: float = 0.1):
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size * expansion_factor)
        self.down = nn.Linear(hidden_size * expansion_factor, hidden_size)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.up(tensor)
        tensor = F.relu(tensor)
        tensor = self.dropout(tensor)
        tensor = self.down(tensor)
        return tensor

class LLMLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 head_dim: int,
                 num_head: int,
                 embedding: ROPEEmbedding,
                 max_context_length: int,
                 expansion_factor: int = 4,
                 dropout_ratio: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = Attention(hidden_dim, head_dim, num_head, embedding, max_context_length)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        skip_connection = tensor
        tensor = self.norm1(tensor)
        tensor = self.attention(tensor)
        tensor += skip_connection

        skip_connection = tensor
        tensor = self.norm2(tensor)
        tensor = self.ffn(tensor)
        tensor += skip_connection

        return tensor

class LLM(nn.Module):
    def __init__(self,
                 num_layer: int,
                 vocabulary_size: int,
                 max_context_length: int,
                 hidden_dim: int,
                 expansion_factor: int = 4,
                 head_dim: int = 64,
                 num_head: int = None,
                 dropout_ratio: float = 0.1,
                 theta: int = 10000,
                 device: str = "mps"):
        super().__init__()
        self.embedding = ROPEEmbedding(max_context_length, head_dim=head_dim, theta=theta)
        self.token_embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.num_layer = num_layer

        if num_head == None:
            num_head = (hidden_dim // head_dim)

        if hidden_dim % (head_dim * num_head) != 0:
            raise ValueError("Error: hidden_dim must be divisible by the product of the number of heads and the head dimension.")

        self.transformer = nn.ModuleList()
        for _ in range(self.num_layer):
            self.transformer.append(LLMLayer(hidden_dim, head_dim, num_head, self.embedding, max_context_length, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio))
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, vocabulary_size)
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        tensor = self.token_embedding(token_ids)

        for layer in self.transformer:
            tensor = layer(tensor)

        # Classification
        tensor = self.output_norm(tensor)
        tensor = self.classifier(tensor)

        return tensor
