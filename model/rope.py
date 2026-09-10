"""Rotary positional embeddings."""

import torch
import torch.nn as nn


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
        # original_shape = tensor.shape
        tensor = tensor.view(B, H, S, D//2, 2) # Get to pairs
        # tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1, 2) # Get to pairs
        tensor = tensor[:, :, :, :, [1, 0]].contiguous() # Swap
        # tensor = tensor[..., [1, 0]]
        tensor = tensor.view(B, H, S, D) # Get back to original shape
        # tensor = tensor.reshape(original_shape)
        return tensor

    def forward(self, tensor: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        sequence_length = tensor.shape[2] # Assuming we are using batch_size, head, sequence_length and dim
        if position_offset < 0 or position_offset + sequence_length > self.pos_emb.shape[0]: # Check to see the embedding we're rotating fits into max_context_length after image offset(if present)
            raise ValueError("Sequence including cached tokens exceeds the RoPE context length.")

        tensor = torch.cat((tensor, self.flip_for_sin(tensor)), dim=-1)
        tensor = tensor * self.pos_emb[position_offset:position_offset + sequence_length, :] # Image offset/KV cache if applicable.
        cos, sin = tensor.chunk(chunks=2, dim=-1)
        tensor = cos + sin
        return tensor
