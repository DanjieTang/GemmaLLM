import math
from dataclasses import dataclass
from os import PathLike
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


# Each layer stores rotated keys and values as [batch, kv_heads, sequence, head_dim].
KVCache = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[KVCache, ...]


@dataclass(frozen=True)
class VLMCache:
    """Decoder KV tensors and text length, excluding the optional image prefix."""

    past_key_values: PastKeyValues
    text_length: int


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
            self.moe = MOE(hidden_dim, num_experts=num_experts, expansion_factor=expansion_factor, dropout_ratio=dropout_ratio, lora_rank=lora_rank, device=device)
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

class VLM(nn.Module):
    def __init__(self,
                 num_layer: int,
                 max_context_length: int,
                 word_embeddings_tensor: str,
                 expansion_factor: int = 4,
                 head_dim: int = 64,
                 q_head: int = None,
                 kv_head: int = None,
                 dropout_ratio: float = 0.1,
                 theta: int = 10000,
                 projection_dim: int = None,
                 use_moe: bool = False,
                 num_experts=8,
                 load_balancing_loss_weight: float = 1e-2,
                 fine_tuning: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 device: str = "cuda",
                 clip_model_id: str = "openai/clip-vit-large-patch14"):
        super().__init__()

        self.max_context_length = max_context_length
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_id)
        self.vision_processor = CLIPImageProcessor.from_pretrained(clip_model_id)
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()
        image_dim = self.vision_model.config.hidden_size
        self.visual_seq_len = 1  # Use only CLIP's CLS token.
        self.multimodal_prefix_len = self.visual_seq_len + 1

        # Load model token embeddings
        word_embeddings = torch.load(word_embeddings_tensor, map_location="cpu",
                                     weights_only=True, mmap=True)
        if not isinstance(word_embeddings, torch.Tensor) or word_embeddings.ndim != 2:
            raise ValueError("Expected a 2D token embedding tensor.")
        word_embeddings = word_embeddings.to(device)
        self.register_buffer("word_embeddings_tensor", word_embeddings, persistent=False)
        self.vocabulary_size, text_dim = self.word_embeddings_tensor.shape
        self.word_embeddings_tensor.requires_grad = False

        # The token that seperates image tokens from text tokens
        if projection_dim:
            self.seperation_token = nn.Parameter(torch.randn(1, projection_dim))
        else:
            self.seperation_token = nn.Parameter(torch.randn(1, text_dim))

        # Potentially image&text token projection layer
        if projection_dim:
            self.image_projection = image_dim != projection_dim
            self.text_projection = text_dim != projection_dim
        else: # If no explicit projection_dim, use text token_dim
            self.image_projection = image_dim != text_dim
            self.text_projection = False
        if projection_dim:
            if self.image_projection:
                self.image_token_projection = nn.Linear(image_dim, projection_dim)
            if self.text_projection:
                self.text_token_projection = nn.Linear(text_dim, projection_dim)
        else: 
            if self.image_projection:
                self.image_token_projection = nn.Linear(image_dim, text_dim)

        # Create the LLM
        self.llm = LLM(
            num_layer,
            self.vocabulary_size,
            max_context_length + self.multimodal_prefix_len,
            projection_dim if projection_dim else text_dim,
            expansion_factor=expansion_factor,
            head_dim=head_dim,
            q_head=q_head,
            kv_head=kv_head,
            dropout_ratio=dropout_ratio,
            theta=theta,
            use_moe=use_moe,
            num_experts=num_experts,
            load_balancing_loss_weight=load_balancing_loss_weight,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            device=device,
        )
        self.device = device
        self.fine_tuning = fine_tuning

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_model.eval()
        return self

    def begin_fine_tunning(self) -> None:
        self.fine_tuning = True
        for name, param in self.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def exit_fine_tunning(self) -> None:
        self.fine_tuning = False
        for name, param in self.named_parameters():
            if "pos_emb" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.vision_model.requires_grad_(False)

    def _encode_images(
        self,
        image_paths: Sequence[str | PathLike[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return one CLIP CLS token per image with shape [images, 1, dim]."""
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        try:
            vision_inputs = self.vision_processor(
                images=images,
                return_tensors="pt",
            ).to(device)
        finally:
            for image in images:
                image.close()

        with torch.no_grad():
            outputs = self.vision_model(**vision_inputs)
            # CLIP prepends its CLS token at sequence position zero.
            image_tokens = outputs.last_hidden_state[:, :1, :]

        if self.image_projection:
            image_tokens = image_tokens.to(
                self.image_token_projection.weight.dtype
            )
            image_tokens = self.image_token_projection(image_tokens)

        return image_tokens.to(device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        image_paths: Sequence[str | PathLike[str] | None] | None = None,
        attention_mask: torch.Tensor | None = None,
        image_tokens: torch.Tensor | None = None,
        past_key_values: VLMCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, VLMCache]:
        """Predict next text tokens; attention_mask marks right-padded text inputs.

        image_tokens optionally supplies projected CLIP tokens [batch, 1, hidden]
        so generation can reuse the encoder output across decoding steps.

        With use_cache=True, supply image_paths or image_tokens only on the
        initial forward pass (one token or a full text prompt). Later passes
        supply only new text tokens and past_key_values; the cache already
        contains the image and separator keys/values.
        """
        batch_size, text_seq_len = token_ids.shape
        past_text_length = 0
        if past_key_values is not None:
            if not use_cache:
                raise ValueError("past_key_values requires use_cache=True.")
            if not isinstance(past_key_values, VLMCache):
                raise ValueError("past_key_values must be a VLMCache returned by VLM.")
            if image_paths is not None or image_tokens is not None:
                raise ValueError("Supply image inputs only when initializing the KV cache.")
            past_text_length = past_key_values.text_length
        if text_seq_len < 1 or not 0 < past_text_length + text_seq_len <= self.max_context_length:
            raise ValueError("Text length including cached tokens must be within max_context_length.")
        if token_ids.min() < 0 or token_ids.max() >= self.vocabulary_size:
            raise ValueError("Token ID outside the embedding vocabulary.")
        if image_tokens is not None and image_paths is not None:
            raise ValueError("Supply image_paths or cached image_tokens, not both.")
        if image_paths is None:
            image_paths = [None] * batch_size
        # Before: image_paths = ["photo.jpg", "", None]
        # After:  image_paths = ["photo.jpg", None, None]
        image_paths = [path or None for path in image_paths]
        if batch_size != len(image_paths):
            raise ValueError("Mismatch between text and image inputs")

        text_embeddings = self.word_embeddings_tensor[token_ids].float()
        if self.text_projection:
            text_embeddings = self.text_token_projection(text_embeddings)
        device = text_embeddings.device
        image_indices = list(range(batch_size)) if image_tokens is not None else [
            index for index, path in enumerate(image_paths) if path is not None
        ]
        if use_cache and image_indices and len(image_indices) != batch_size:
            raise ValueError("KV caching requires all samples to have images or all to be text-only.")

        input_embeddings = text_embeddings
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            if attention_mask.shape != token_ids.shape:
                raise ValueError("attention_mask must match token_ids.")
            if not attention_mask[:, 0].all() or (  # At least one real token, with padding only at the end.
                attention_mask[:, 1:] & ~attention_mask[:, :-1]
            ).any():
                raise ValueError("Only nonempty, right-padded text inputs are supported.")
        if use_cache and not attention_mask.all(): # Use KV cache only during inference, during autoregressive inference attention mask should be all true
            raise ValueError("KV caching requires unpadded text inputs.")
        valid_token_mask = attention_mask
        text_positions = torch.arange(text_seq_len, device=device)[None, :]
        batch_indices = torch.arange(batch_size, device=device)[:, None]
        if image_indices:
            has_image = torch.tensor(
                [index in image_indices for index in range(batch_size)], device=device
            )
            offsets = has_image.long() * self.multimodal_prefix_len
            text_positions = text_positions + offsets[:, None]
            seq_len = text_seq_len + self.multimodal_prefix_len
            input_embeddings = text_embeddings.new_zeros(
                batch_size, seq_len, text_embeddings.shape[-1]
            )
            input_embeddings[batch_indices, text_positions] = text_embeddings

            if image_tokens is None:
                image_tokens = self._encode_images(
                    [image_paths[index] for index in image_indices],
                    device=device,
                    dtype=text_embeddings.dtype,
                )
            if image_tokens.shape != (len(image_indices), 1, text_embeddings.shape[-1]):  # Example: image_tokens.shape == (3, 1, 768)
                raise ValueError("Cached image tokens have an incompatible shape.")
            input_embeddings[image_indices, :self.visual_seq_len] = image_tokens.to(
                device=device, dtype=text_embeddings.dtype,
            )
            input_embeddings[image_indices, self.visual_seq_len] = (
                self.seperation_token.to(text_embeddings.dtype)
            )
            # Mark real tokens as True and padding or unused positions as False.
            # With an image: [image, separator, word1, word2, padding]
            # Mask:         [True,  True,      True,  True,  False]
            # Text only:    [word1, word2, padding, unused, unused]
            # Mask:         [True,  True,  False,   False,  False]
            # Start with every position marked as padding.
            valid_token_mask = torch.zeros(batch_size, seq_len, device=device,
                                          dtype=torch.bool)
            # Copy the text mask into the positions where text tokens were placed.
            valid_token_mask[batch_indices, text_positions] = attention_mask
            # Mark the image token and separator as valid for samples with images.
            valid_token_mask[image_indices, :self.multimodal_prefix_len] = True

        # Padding is strictly after the text, so the shared causal mask
        # already prevents every real token from attending to padding.
        seq_len = input_embeddings.shape[1]
        if use_cache:
            # LLM derives the rectangular causal mask and RoPE offset from the
            # cache length, which already includes the image and separator.
            logits, load_balancing_loss, present_key_values = self.llm(
                input_embeddings,
                fine_tuning=self.fine_tuning,
                valid_token_mask=valid_token_mask,
                past_key_values=(None if past_key_values is None
                                 else past_key_values.past_key_values),
                use_cache=True,
            )
        else:
            causal_mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=device,
                dtype=input_embeddings.dtype,
            ).triu(diagonal=1)
            logits, load_balancing_loss = self.llm(
                input_embeddings,
                causal_mask,
                self.fine_tuning,
                valid_token_mask=valid_token_mask,
            )
        if image_indices:
            # Select text predictions at offset 2 for image samples and 0
            # for text-only samples, preserving batch order and gradients.
            logits = logits[batch_indices, text_positions]
        if use_cache:
            return logits.contiguous(), load_balancing_loss, VLMCache(
                present_key_values, past_text_length + text_seq_len,
            )
        return logits.contiguous(), load_balancing_loss
