"""Gated feed-forward network with optional LoRA adapters."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 expansion_factor: int = 4,
                 dropout_ratio: float = 0.1,
                 lora_rank: int = 16,
                 lora_alpha: int = 32):
        super().__init__()
        self.gate_and_up = nn.Linear(hidden_size, hidden_size * expansion_factor * 2)
        self.down = nn.Linear(hidden_size * expansion_factor, hidden_size)
        self.dropout = nn.Dropout(p=dropout_ratio)

        # LoRA
        self.lora_scale = lora_alpha / lora_rank
        self.lora_gate_and_up_a = nn.Linear(hidden_size, lora_rank)
        self.lora_gate_and_up_b = nn.Linear(lora_rank, hidden_size * expansion_factor * 2)
        self.lora_down_a = nn.Linear(hidden_size * expansion_factor, lora_rank)
        self.lora_down_b = nn.Linear(lora_rank, hidden_size)

    def forward(self, tensor: torch.Tensor, fine_tuning: bool = False) -> torch.Tensor:
        gate_and_up = self.gate_and_up(tensor)
        if fine_tuning:
            lora_tensor = self.lora_gate_and_up_a(tensor)
            lora_tensor = self.lora_gate_and_up_b(lora_tensor)
            lora_tensor = lora_tensor * self.lora_scale
            gate_and_up = gate_and_up + lora_tensor
        gate, up = gate_and_up.chunk(chunks=2, dim=-1)
        gate = F.gelu(gate, approximate="tanh")
        tensor = gate * up
        tensor = self.dropout(tensor)
        down_tensor = self.down(tensor)
        if fine_tuning:
            lora_tensor = self.lora_down_a(tensor)
            lora_tensor = self.lora_down_b(lora_tensor)
            lora_tensor = lora_tensor * self.lora_scale
            down_tensor = down_tensor + lora_tensor
        return down_tensor
