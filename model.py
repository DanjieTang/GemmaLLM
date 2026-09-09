import math
from os import PathLike
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel


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

    def forward(self, tensor: torch.Tensor, attention_mask: torch.Tensor = None, fine_tuning: bool = False) -> torch.Tensor:
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

        if self.multi_query_attention:
            # If we are using multi query attention, duplicate key value heads
            key = torch.repeat_interleave(key, self.q_kv_scale, dim=-2)
            value = torch.repeat_interleave(value, self.q_kv_scale, dim=-2)

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
    ):
        skip_connection = tensor
        tensor = self.norm1(tensor)
        tensor = self.mqa(tensor, attention_mask=attention_mask, fine_tuning=fine_tuning)
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
        causal_mask: torch.Tensor,
        fine_tuning: bool,
        valid_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Track load-balancing across layers (only if MoE is used)
        load_balancing_sum = torch.tensor(0.0, device=self.device)

        for layer in self.transformer:
            tensor, load_balancing_loss = layer(
                tensor,
                attention_mask=causal_mask,
                fine_tuning=fine_tuning,
                valid_token_mask=valid_token_mask,
            )
            load_balancing_sum += load_balancing_loss

        load_balancing_loss = (load_balancing_sum / self.num_layer) * self.load_balancing_loss_weight

        # Classification
        tensor = self.output_norm(tensor)
        tensor = self.classifier(tensor)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next text tokens; attention_mask marks right-padded text inputs.

        image_tokens optionally supplies projected CLIP tokens [batch, 1, hidden]
        so generation can reuse the encoder output across decoding steps.
        """
        batch_size, text_seq_len = token_ids.shape
        if not 0 < text_seq_len <= self.max_context_length:
            raise ValueError("Text input length must be within max_context_length.")
        if token_ids.min() < 0 or token_ids.max() >= self.vocabulary_size:
            raise ValueError("Token ID outside the embedding vocabulary.")
        if image_tokens is not None and image_paths is not None:
            raise ValueError("Supply image_paths or cached image_tokens, not both.")
        if image_paths is None:
            image_paths = [None] * batch_size
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

        input_embeddings = text_embeddings
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            if attention_mask.shape != token_ids.shape:
                raise ValueError("attention_mask must match token_ids.")
            if not attention_mask[:, 0].all() or (
                attention_mask[:, 1:] & ~attention_mask[:, :-1]
            ).any():
                raise ValueError("Only nonempty, right-padded text inputs are supported.")
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
            if image_tokens.shape != (len(image_indices), 1, text_embeddings.shape[-1]):
                raise ValueError("Cached image tokens have an incompatible shape.")
            input_embeddings[image_indices, :self.visual_seq_len] = image_tokens
            input_embeddings[image_indices, self.visual_seq_len] = (
                self.seperation_token.to(text_embeddings.dtype)
            )
            valid_token_mask = torch.zeros(batch_size, seq_len, device=device,
                                          dtype=torch.bool)
            valid_token_mask[batch_indices, text_positions] = attention_mask
            valid_token_mask[image_indices, :self.multimodal_prefix_len] = True

        # Padding is strictly after the text, so the shared causal mask
        # already prevents every real token from attending to padding.
        seq_len = input_embeddings.shape[1]
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
        return logits.contiguous(), load_balancing_loss

    @torch.inference_mode()
    def generate(self, image_path: str | PathLike[str], bos_token_id: int,
                 eos_token_id: int, max_new_tokens: int = 256,
                 temperature: float = 0.0, pad_token_id: int | None = None) -> list[int]:
        """Generate one annotation, encoding the image once; no KV cache yet."""
        if max_new_tokens < 1 or temperature < 0:
            raise ValueError("max_new_tokens must be positive and temperature nonnegative.")
        for token_id in (bos_token_id, eos_token_id, pad_token_id):
            if token_id is not None and not 0 <= token_id < self.vocabulary_size:
                raise ValueError("Special token ID outside the vocabulary.")
        was_training = self.training
        self.eval()
        try:
            device = self.word_embeddings_tensor.device
            image_tokens = self._encode_images([image_path], device=device,
                                               dtype=self.seperation_token.dtype)
            tokens = torch.tensor([[bos_token_id]], device=device)
            generated = []
            for _ in range(min(max_new_tokens, self.max_context_length)):
                logits, _ = self(tokens, image_tokens=image_tokens)
                scores = logits[0, -1].clone()
                for token_id in (bos_token_id, pad_token_id):
                    if token_id is not None and token_id != eos_token_id:
                        scores[token_id] = float("-inf")
                if temperature == 0:
                    next_token = scores.argmax().item()
                else:
                    next_token = torch.multinomial(
                        (scores / temperature).softmax(-1), 1
                    ).item()
                if next_token == eos_token_id:
                    break
                generated.append(next_token)
                tokens = torch.cat((tokens, tokens.new_tensor([[next_token]])), dim=1)
            return generated
        finally:
            self.train(was_training)
