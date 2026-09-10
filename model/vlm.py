"""Vision-language model with CLIP encoding and text/image fusion."""

from os import PathLike
from typing import Sequence

import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

from .cache import VLMCache
from .llm import LLM


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
