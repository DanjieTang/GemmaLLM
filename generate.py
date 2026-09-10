"""Generate an annotation from an image and a trained custom VLM checkpoint."""

import argparse
from os import PathLike
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import VLM


@torch.inference_mode()
def generate(model: VLM, image_path: str | PathLike[str], bos_token_id: int,
             eos_token_id: int, max_new_tokens: int = 256,
             temperature: float = 0.0, pad_token_id: int | None = None,
             use_cache: bool = True) -> list[int]:
    """Generate one annotation, reusing image and decoder KV caches by default."""
    if max_new_tokens < 1 or temperature < 0:
        raise ValueError("max_new_tokens must be positive and temperature nonnegative.")
    for token_id in (bos_token_id, eos_token_id, pad_token_id):
        if token_id is not None and not 0 <= token_id < model.vocabulary_size:
            raise ValueError("Special token ID outside the vocabulary.")
    was_training = model.training
    model.eval()
    try:
        device = model.word_embeddings_tensor.device
        image_tokens = model._encode_images([image_path], device=device,
                                           dtype=model.seperation_token.dtype)
        tokens = torch.tensor([[bos_token_id]], device=device)
        generated = []
        cache = None
        for _ in range(min(max_new_tokens, model.max_context_length)):
            if use_cache:
                logits, _, cache = model(
                    tokens if cache is None else tokens[:, -1:],
                    image_tokens=image_tokens if cache is None else None,
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                logits, _ = model(tokens, image_tokens=image_tokens)
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
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--embeddings_path", type=Path,
                        help="Override the embedding path saved in the checkpoint.")
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Zero uses greedy decoding; positive values sample.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True,
                            mmap=True)
    config = dict(checkpoint["model_config"])
    if args.embeddings_path:
        config["word_embeddings_tensor"] = str(args.embeddings_path.expanduser())
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or checkpoint["tokenizer_path"], local_files_only=True,
    )
    special_ids = checkpoint["special_token_ids"]
    for name, value in special_ids.items():
        if getattr(tokenizer, name) != value:
            raise ValueError(f"Tokenizer {name} differs from the training checkpoint.")
    model = VLM(**config, device=args.device).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokens = generate(model, str(args.image), **special_ids,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature)
    print(tokenizer.decode(tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
