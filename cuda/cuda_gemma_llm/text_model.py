"""PyTorch reference for the CUDA decoder: VLM's text-only generation path.

TextGenerator holds exactly the VLM pieces used for text-only generation
(token embeddings, optional text projection, and the LLM), with the same
state_dict names, so trained VLM checkpoints load into it directly.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import LLM  # noqa: E402
from run_sweep import parse_simple_yaml  # noqa: E402

# gemma-4-31B-it token embeddings (data/gemma-4-31B-it-embeddings.pt).
GEMMA_VOCAB_SIZE = 262144
GEMMA_TEXT_DIM = 5376
BOS_TOKEN_ID = 2

# VLM keyword arguments that shape the text decoder.
ARCHITECTURE_KEYS = (
    "num_layer", "max_context_length", "projection_dim", "expansion_factor",
    "head_dim", "q_head", "kv_head", "theta", "use_moe", "fine_tuning",
    "lora_rank", "lora_alpha",
)


class TextGenerator(nn.Module):
    def __init__(self,
                 num_layer: int,
                 max_context_length: int,
                 vocab_size: int = GEMMA_VOCAB_SIZE,
                 text_dim: int = GEMMA_TEXT_DIM,
                 projection_dim: int | None = None,
                 expansion_factor: int = 4,
                 head_dim: int = 64,
                 q_head: int | None = None,
                 kv_head: int | None = None,
                 theta: int = 10000,
                 use_moe: bool = False,
                 fine_tuning: bool = False,
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 word_embeddings: torch.Tensor | None = None,
                 embedding_dtype: torch.dtype = torch.bfloat16,
                 device: str = "cuda"):
        super().__init__()
        if use_moe:
            raise ValueError("The CUDA decoder does not implement MoE layers.")
        if word_embeddings is None:
            # Stand-in for the Gemma table, kept in its bf16 storage dtype.
            word_embeddings = torch.randn(vocab_size, text_dim, device=device,
                                          dtype=embedding_dtype)
        if word_embeddings.ndim != 2:
            raise ValueError("Expected a 2D token embedding tensor.")
        self.register_buffer("word_embeddings_tensor", word_embeddings.to(device), persistent=False)
        self.vocabulary_size, text_dim = word_embeddings.shape
        self.max_context_length = max_context_length
        self.fine_tuning = fine_tuning

        hidden_dim = projection_dim or text_dim
        self.text_projection = hidden_dim != text_dim
        if self.text_projection:
            self.text_token_projection = nn.Linear(text_dim, hidden_dim, device=device)
        self.llm = LLM(
            num_layer, self.vocabulary_size, max_context_length, hidden_dim,
            expansion_factor=expansion_factor, head_dim=head_dim, q_head=q_head,
            kv_head=kv_head, dropout_ratio=0.0, theta=theta, lora_rank=lora_rank,
            lora_alpha=lora_alpha, device=device,
        ).to(device)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Same as VLM.embed_tokens: [batch, tokens] -> [batch, tokens, hidden_dim]."""
        embeddings = self.word_embeddings_tensor[token_ids].float()
        if self.text_projection:
            embeddings = self.text_token_projection(embeddings)
        return embeddings


def architecture_from_yaml(path: Path) -> dict:
    """Read the decoder settings of a sweep YAML (the first value of any list)."""
    config = parse_simple_yaml(Path(path))
    architecture = {}
    for key in ARCHITECTURE_KEYS:
        if key in config:
            value = config[key]
            architecture[key] = value[0] if isinstance(value, list) else value
    return architecture


def load_checkpoint(path: Path, embeddings_path: Path | None = None,
                    max_context_length: int | None = None,
                    device: str = "cpu") -> tuple[TextGenerator, dict]:
    """Load a train.py checkpoint's text decoder; returns (model, architecture)."""
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu",
                            weights_only=True, mmap=True)
    config = checkpoint["model_config"]
    architecture = {key: config[key] for key in ARCHITECTURE_KEYS if key in config}
    if max_context_length is not None:
        architecture["max_context_length"] = max(
            architecture["max_context_length"], max_context_length)
    embeddings_path = Path(embeddings_path or config["word_embeddings_tensor"]).expanduser()
    word_embeddings = torch.load(embeddings_path, map_location="cpu",
                                 weights_only=True, mmap=True)
    model = TextGenerator(**architecture, word_embeddings=word_embeddings, device=device)
    # Keep only the text path; RoPE tables are rebuilt from the configuration.
    state_dict = {
        name: value for name, value in checkpoint["model_state_dict"].items()
        if name.startswith(("llm.", "text_token_projection.")) and "pos_emb" not in name
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [name for name in missing if "pos_emb" not in name]
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval(), architecture


def make_prompt(length: int, vocab_size: int) -> list[int]:
    """Same deterministic prompt as the CUDA benchmark: BOS, then spread-out IDs."""
    return [(BOS_TOKEN_ID if i == 0 else 7919 * i) % vocab_size for i in range(length)]


@torch.inference_mode()
def generate_greedy(model: TextGenerator, prompt: list[int], total_tokens: int,
                    return_first_logits: bool = False
                    ) -> tuple[list[int], torch.Tensor | None]:
    """generate.py's cached greedy loop, without EOS stopping or token masking.

    Runs until the sequence (prompt included) holds total_tokens tokens, so
    both implementations always do the same amount of work.
    """
    if not 0 < len(prompt) < total_tokens <= model.max_context_length:
        raise ValueError("Require 0 < prompt length < total_tokens <= max_context_length.")
    device = model.word_embeddings_tensor.device
    tokens = torch.tensor([prompt], device=device)
    cache = None
    first_logits = None
    while tokens.shape[1] < total_tokens:
        logits, _, cache = model.llm(
            model.embed_tokens(tokens if cache is None else tokens[:, -1:]),
            fine_tuning=model.fine_tuning, past_key_values=cache, use_cache=True,
        )
        scores = logits[0, -1]
        if return_first_logits and first_logits is None:
            first_logits = scores.float().cpu()
        next_token = scores.argmax().item()
        tokens = torch.cat((tokens, tokens.new_tensor([[next_token]])), dim=1)
    return tokens[0].tolist(), first_logits
