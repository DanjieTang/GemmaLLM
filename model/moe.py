"""Top-two mixture-of-experts routing and load-balancing loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feed_forward import FeedForward


class MOE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int = 8, expansion_factor: int = 4, dropout_ratio: float = 0.1, lora_rank: int = 16, lora_alpha: int = 32, device: str = "mps"):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([FeedForward(hidden_size, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio, lora_rank=lora_rank, lora_alpha=lora_alpha) for _ in range(num_experts)])
        self.device = device

    def forward(
        self,
        tensor: torch.Tensor,
        fine_tuning: bool = False,
        valid_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Flatten for better manipulation, this is ok because tokens are independent at this stage
        batch_size, seq_len, hidden_size = tensor.shape
        flat_tensor = tensor.reshape(batch_size * seq_len, hidden_size)

        # Pass through the gating network and select experts
        tensor = self.gate(flat_tensor)
        tensor = F.softmax(tensor, dim=-1)

        # The output of this step is a tensor of shape [batch_size * seq_len, 2] with element i in the second dimension representing ith expert selected for this token
        value_tensor, index_tensor = tensor.topk(k=2, dim=-1)

        # Find the load balancing loss
        # Exclude trailing batch padding from routing statistics. The mask
        # has shape [batch, seq_len] and does not change the attention mask.
        routing_probabilities = tensor
        first_experts = index_tensor[:, 0]
        if valid_token_mask is not None:
            valid_tokens = valid_token_mask.reshape(-1)
            routing_probabilities = routing_probabilities[valid_tokens]
            first_experts = first_experts[valid_tokens]
        counts = torch.bincount(first_experts, minlength=self.num_experts)
        frequencies = counts.float() / routing_probabilities.shape[0]
        probability = routing_probabilities.mean(0)
        load_balancing_loss = (probability * frequencies).mean() * float(self.num_experts ** 2)

        # Normalize top1 and top2 score
        top_expert_score = value_tensor[:, 0]
        second_expert_score = value_tensor[:, 1]
        total_score = top_expert_score + second_expert_score
        top_expert_score = top_expert_score / total_score
        second_expert_score = second_expert_score / total_score

        # Split into top 2 experts
        split_tensors = torch.split(index_tensor, 1, dim=-1)
        top_expert, second_expert = split_tensors[0], split_tensors[1]
        indices = torch.arange(batch_size * seq_len).unsqueeze(-1).to(self.device)
        top_expert = torch.cat((indices, top_expert), dim=-1)
        second_expert = torch.cat((indices, second_expert), dim=-1)

        # Sort based on expert selection
        top_expert = top_expert[top_expert[:,1].argsort()]
        second_expert = second_expert[second_expert[:,1].argsort()]

        # Count how many tokens goes to each expert
        top_expert_counts = torch.zeros(self.num_experts, dtype=int)
        for i in range(self.num_experts):
            top_expert_counts[i] = (top_expert[:,1] == i).sum()
        top_expert_counts = top_expert_counts.tolist()

        second_expert_counts = torch.zeros(self.num_experts, dtype=int)
        for i in range(self.num_experts):
            second_expert_counts[i] = (second_expert[:,1] == i).sum()
        second_expert_counts = second_expert_counts.tolist()

        # Split input tokens for each expert
        top_expert_tokens = flat_tensor[top_expert[:,0]]
        second_expert_tokens = flat_tensor[second_expert[:,0]]

        # Split into a list of tensors, element i tensor is for ith expert.
        top_expert_tokens = torch.split(top_expert_tokens, top_expert_counts, dim=0)
        second_expert_tokens = torch.split(second_expert_tokens, second_expert_counts, dim=0)

        # Input into each expert and obtain results in a list
        top_expert_outputs = [self.experts[i](top_expert_tokens[i], fine_tuning) if top_expert_counts[i] > 0 else torch.zeros(0, hidden_size, dtype=torch.float16).to(self.device) for i in range(self.num_experts)]
        second_expert_outputs = [self.experts[i](second_expert_tokens[i], fine_tuning) if second_expert_counts[i] > 0 else torch.zeros(0, hidden_size, dtype=torch.float16).to(self.device) for i in range(self.num_experts)]

        # Combine outputs
        top_expert_outputs = torch.cat(top_expert_outputs, dim=0)
        second_expert_outputs = torch.cat(second_expert_outputs, dim=0)

        # Re-index the output back to original token order
        flat_top_expert_tensor = torch.zeros_like(flat_tensor, dtype=torch.float32).to(self.device)
        flat_top_expert_tensor.index_copy_(0, top_expert[:, 0], top_expert_outputs)

        flat_second_expert_tensor = torch.zeros_like(flat_tensor, dtype=torch.float32).to(self.device)
        flat_second_expert_tensor.index_copy_(0, second_expert[:, 0], second_expert_outputs)

        # Find final output tensor based on weight between top and second expert
        final_tensor = top_expert_score.unsqueeze(-1) * flat_top_expert_tensor + second_expert_score.unsqueeze(-1) * flat_second_expert_tensor

        # Reshape to original [batch_size, seq_len, hidden_size]
        final_tensor = final_tensor.reshape(batch_size, seq_len, hidden_size)

        return final_tensor, load_balancing_loss
