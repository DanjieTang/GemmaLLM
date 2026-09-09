"""Generate an annotation from an image and a trained custom VLM checkpoint."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import VLM


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
    tokens = model.generate(str(args.image), **special_ids,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature)
    print(tokenizer.decode(tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
