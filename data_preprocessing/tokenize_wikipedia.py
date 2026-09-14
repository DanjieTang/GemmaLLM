"""Stream full multilingual Wikipedia editions into memory-mappable token shards."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Iterable, Iterator

from datasets import load_dataset
from huggingface_hub import HfApi
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DATASET = "wikimedia/wikipedia"
DEFAULT_LANGUAGES = ["en", "zh", "es", "fr", "de", "ja", "ru"]
DEFAULT_TOKENIZER = Path.home() / "models" / "gemma-4-E2B-it"


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def article_split(language: str, article_id: str, seed: int,
                  val_fraction: float) -> str:
    """Keep every occurrence/chunk of an article in one deterministic split."""
    key = json.dumps([seed, language, article_id], ensure_ascii=False).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return "val" if bucket < int(val_fraction * (1 << 64)) else "train"


def iter_windows(ids: np.ndarray, context_length: int) -> Iterator[np.ndarray]:
    """Yield up to context_length + 1 IDs; adjacent windows share one token.

    Each consecutive token pair is a prediction target exactly once. BOS/EOS
    are supplied at article boundaries by the caller, never at chunk boundaries.
    """
    if context_length < 1 or ids.ndim != 1 or len(ids) < 2:
        raise ValueError("Expected a positive context length and at least two IDs.")
    for start in range(0, len(ids) - 1, context_length):
        yield ids[start:start + context_length + 1]


class ShardWriter:
    """Write (rows, context_length + 1) int32 arrays and valid row lengths."""

    def __init__(self, directory: Path, context_length: int, shard_rows: int,
                 pad_token_id: int):
        self.directory = directory
        directory.mkdir(parents=True)
        self.context_length = context_length
        self.pad_token_id = pad_token_id
        self.tokens = np.empty((shard_rows, context_length + 1), dtype=np.int32)
        self.lengths = np.empty(shard_rows, dtype=np.int32)
        self.used = 0
        self.rows = 0
        self.articles = 0
        self.target_tokens = 0
        self.shards = []
        self.metadata = (directory / "articles.jsonl").open("w", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.metadata.close()

    def add_article(self, article: dict, ids: np.ndarray) -> None:
        first_row = self.rows
        for window in iter_windows(ids, self.context_length):
            row = self.tokens[self.used]
            row[:len(window)] = window
            row[len(window):] = self.pad_token_id
            self.lengths[self.used] = len(window)
            self.used += 1
            self.rows += 1
            self.target_tokens += len(window) - 1
            if self.used == len(self.tokens):
                self.flush()
        # Global row offsets refer to the ordered shards within this split.
        self.metadata.write(json.dumps({
            "id": article["id"], "url": article["url"], "title": article["title"],
            "first_row": first_row, "rows": self.rows - first_row,
            "article_tokens": len(ids),
        }, ensure_ascii=False) + "\n")
        self.articles += 1

    def flush(self) -> None:
        if not self.used:
            return
        name = f"{len(self.shards):05d}"
        token_file = f"tokens-{name}.npy"
        length_file = f"lengths-{name}.npy"
        np.save(self.directory / token_file, self.tokens[:self.used], allow_pickle=False)
        np.save(self.directory / length_file, self.lengths[:self.used], allow_pickle=False)
        self.shards.append({"tokens": token_file, "lengths": length_file,
                            "rows": self.used})
        self.used = 0

    def finish(self) -> dict:
        self.flush()
        self.metadata.flush()
        return {"articles": self.articles, "rows": self.rows,
                "target_tokens": self.target_tokens, "shards": self.shards,
                "article_metadata": "articles.jsonl"}


def tokenize_articles(articles: Iterable[dict], tokenizer: PreTrainedTokenizerBase,
                      directory: Path, config: dict, batch_size: int) -> dict:
    """Consume every article without filtering, deduplication, or truncation."""
    pending = []
    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(ShardWriter(
                directory / split, config["context_length"], config["shard_rows"],
                config["special_token_ids"]["pad_token_id"],
            )) for split in ("train", "val")
        }

        def flush_batch() -> None:
            texts = [article["title"] + "\n\n" + article["text"] for article in pending]
            encoded = tokenizer(
                texts, add_special_tokens=False, padding=False, truncation=False,
                return_attention_mask=False, return_token_type_ids=False,
            )["input_ids"]
            if len(encoded) != len(pending):
                raise ValueError("Tokenizer returned an unexpected batch size.")
            for article, body in zip(pending, encoded):
                ids = np.asarray([tokenizer.bos_token_id, *body, tokenizer.eos_token_id],
                                 dtype=np.int64)
                if ids.min() < 0 or ids.max() > np.iinfo(np.int32).max:
                    raise ValueError(f"Article {article['id']}: IDs cannot fit in int32.")
                split = article_split(config["language"], str(article["id"]),
                                      config["seed"], config["val_fraction"])
                writers[split].add_article(article, ids.astype(np.int32))
            pending.clear()

        with tqdm(desc=config["language"], unit="article", dynamic_ncols=True) as progress:
            for article in articles:
                # Fail loudly on corrupt records rather than silently dropping data.
                if any(not isinstance(article.get(key), str)
                       for key in ("id", "url", "title", "text")):
                    raise ValueError("Each article must have string id/url/title/text fields.")
                pending.append(article)
                if len(pending) == batch_size:
                    count = len(pending)
                    flush_batch()
                    progress.update(count)
            if pending:
                count = len(pending)
                flush_batch()
                progress.update(count)
        if not sum(writer.articles for writer in writers.values()):
            raise ValueError("The source contained no articles.")
        return {split: writer.finish() for split, writer in writers.items()}


def prepare_language(output_dir: Path, config: dict, restart_incomplete: bool) -> Path | None:
    """Skip matching completed editions; only restart explicitly marked work."""
    destination = output_dir / config["language"]
    if destination.exists():
        manifest_path = destination / "manifest.json"
        if (not manifest_path.is_file()
                or json.loads(manifest_path.read_text())["config"] != config):
            raise ValueError(f"{destination}: existing output has different settings; "
                             "choose another --output_dir.")
        print(f"{config['language']}: already complete; skipping.", flush=True)
        return None
    staging = output_dir / f".{config['language']}.incomplete"
    if staging.exists():
        config_path = staging / "config.json"
        if (not config_path.is_file()
                or json.loads(config_path.read_text()) != config):
            raise ValueError(f"{staging}: unrecognized or different settings; "
                             "choose another --output_dir.")
        if not restart_incomplete:
            raise ValueError(f"{staging}: interrupted run found. Use --restart_incomplete "
                             "to redo this language; completed languages are kept.")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    save_json(staging / "config.json", config)
    return staging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/wikipedia")
    parser.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER,
                        help="Local tokenizer directory; no model weights are loaded.")
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--snapshot", default="20231101",
                        help="Published Wikipedia snapshot date (default: 20231101).")
    parser.add_argument("--revision", default=None,
                        help="Dataset Git revision; default: reuse output revision or pin main.")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Articles per tokenizer batch, not a dataset limit.")
    parser.add_argument("--shard_rows", type=int, default=65536,
                        help="Windows per shard (about 64 MiB of tokens at context 256).")
    parser.add_argument("--val_fraction", type=float, default=0.01,
                        help="Fraction of whole articles held out, not a language mixing ratio.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restart_incomplete", action="store_true")
    args = parser.parse_args(argv)
    for name in ("context_length", "batch_size", "shard_rows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive.")
    if not math.isfinite(args.val_fraction) or not 0 <= args.val_fraction < 1:
        parser.error("--val_fraction must be in [0, 1).")
    if not re.fullmatch(r"\d{8}", args.snapshot):
        parser.error("--snapshot must have the form YYYYMMDD.")
    if (len(set(args.languages)) != len(args.languages)
            or any(not re.fullmatch(r"[a-z]+(?:[-_][a-z]+)*", lang)
                   for lang in args.languages)):
        parser.error("--languages must contain distinct Wikipedia language codes.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                              use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for reproducible tokenizer fingerprints.")
    special_ids = {name: getattr(tokenizer, name) for name in
                   ("bos_token_id", "eos_token_id", "pad_token_id")}
    if any(value is None or not 0 <= value <= np.iinfo(np.int32).max
           for value in special_ids.values()):
        raise ValueError("Tokenizer must define nonnegative int32 BOS, EOS, and PAD IDs.")
    fingerprint = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    source_path = output_dir / "source.json"
    if source_path.exists() and args.revision is None:
        source = json.loads(source_path.read_text())
        if source["dataset"] != DATASET or source["snapshot"] != args.snapshot:
            raise ValueError("Output uses a different source; choose another --output_dir.")
        revision = source["revision"]
    else:
        revision = HfApi().dataset_info(DATASET, revision=args.revision or "main").sha
        source = {"dataset": DATASET, "snapshot": args.snapshot, "revision": revision}
        if source_path.exists() and json.loads(source_path.read_text()) != source:
            raise ValueError("Output uses a different revision; choose another --output_dir.")
        save_json(source_path, source)
    print(f"Full editions: {', '.join(args.languages)}; snapshot={args.snapshot}; "
          f"revision={revision}\nOutput: {output_dir}", flush=True)
    for language in args.languages:
        config = {
            "format_version": 1, **source, "language": language,
            "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": fingerprint,
            "special_token_ids": special_ids, "context_length": args.context_length,
            "shard_rows": args.shard_rows, "val_fraction": args.val_fraction,
            "seed": args.seed, "text_format": "BOS + title + two newlines + text + EOS",
        }
        staging = prepare_language(output_dir, config, args.restart_incomplete)
        if staging is None:
            continue
        articles = load_dataset(DATASET, f"{args.snapshot}.{language}", split="train",
                                revision=revision, streaming=True)
        stats = tokenize_articles(articles, tokenizer, staging, config, args.batch_size)
        save_json(staging / "manifest.json", {"config": config, "splits": stats})
        staging.rename(output_dir / language)
        for split, values in stats.items():
            print(f"{language}/{split}: {values['articles']:,} articles, "
                  f"{values['rows']:,} windows, {values['target_tokens']:,} targets", flush=True)
    print("Finished. Shards require a loader that masks padding using lengths arrays.", flush=True)


if __name__ == "__main__":
    main()
