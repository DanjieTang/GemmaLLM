"""Extract Gemma token embeddings without loading or downloading the full model."""

import argparse
import json
from pathlib import Path
import struct
import time

import httpx
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers
from safetensors import safe_open
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def check_tokenizer(source: Path, target: Path, rows: int) -> dict:
    """Check the full vocabulary and encoding rules, not just vocabulary size."""
    source_json = json.loads((source / "tokenizer.json").read_text())
    target_json = json.loads((target / "tokenizer.json").read_text())
    if source_json != target_json:
        raise ValueError("Tokenizer definitions differ; retokenize before training.")
    local = AutoTokenizer.from_pretrained(source, local_files_only=True)
    remote = AutoTokenizer.from_pretrained(target, local_files_only=True)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(local, name) != getattr(remote, name):
            raise ValueError(f"Tokenizers have different {name} values.")
    if max(local.get_vocab().values()) >= rows:
        raise ValueError("Tokenizer IDs exceed the embedding matrix.")
    return {name: getattr(local, name) for name in
            ("bos_token_id", "eos_token_id", "pad_token_id")}


def read_range(client: httpx.Client, url: str, start: int, end: int):
    # A distinct URL per range also avoids incorrectly cached partial responses.
    return client.stream(
        "GET", f"{url}?embedding_range={start}-{end}",
        headers={**build_hf_headers(), "Range": f"bytes={start}-{end}"},
    )


def validate_range(response: httpx.Response, start: int, end: int) -> None:
    response.raise_for_status()
    expected = f"bytes {start}-{end}/"
    if response.status_code != 206 or not response.headers.get(
        "content-range", ""
    ).startswith(expected):
        raise RuntimeError(
            "Server did not honor the byte range. Use --download_shard to "
            "download only the containing shard instead."
        )


def extract_range(url: str, key: str, destination: Path) -> dict:
    """Build a single-tensor safetensors file using resumable HTTP ranges."""
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        with read_range(client, url, 0, 7) as response:
            validate_range(response, 0, 7)
            header_size = struct.unpack("<Q", response.read())[0]
        if not 0 < header_size < 100_000_000:
            raise ValueError("Invalid safetensors header length.")
        with read_range(client, url, 8, 7 + header_size) as response:
            validate_range(response, 8, 7 + header_size)
            tensor_info = json.loads(response.read())[key]
        start, end = tensor_info["data_offsets"]
        length = end - start
        header = json.dumps({key: {
            **tensor_info, "data_offsets": [0, length],
        }}).encode()
        header += b" " * (-len(header) % 8)
        prefix = struct.pack("<Q", len(header)) + header
        if destination.exists():
            with destination.open("rb") as existing:
                if existing.read(len(prefix)) != prefix:
                    raise ValueError(f"Incompatible partial download: {destination}")
        else:
            destination.write_bytes(prefix)
        completed = destination.stat().st_size - len(prefix)
        if not 0 <= completed <= length:
            raise ValueError(f"Invalid partial download size: {destination}")
        with tqdm(total=length, initial=completed, unit="B", unit_scale=True,
                  desc="Token embeddings") as progress:
            failures = 0
            while completed < length:
                range_start = 8 + header_size + start + completed
                range_end = min(range_start + 64 * 1024**2,
                                8 + header_size + end) - 1
                try:
                    with read_range(client, url, range_start, range_end) as response:
                        validate_range(response, range_start, range_end)
                        received = 0
                        with destination.open("ab") as output:
                            for chunk in response.iter_bytes(1024**2):
                                if received + len(chunk) > range_end - range_start + 1:
                                    raise ValueError("Received more bytes than requested.")
                                output.write(chunk)
                                received += len(chunk)
                                completed += len(chunk)
                                progress.update(len(chunk))
                        if received != range_end - range_start + 1:
                            raise httpx.ReadError("Incomplete byte range")
                    failures = 0
                except httpx.TransportError:
                    failures += 1
                    if failures >= 5:
                        raise
                    time.sleep(min(2**failures, 10))
    return tensor_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="google/gemma-4-31B-it")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--tokenizer_path", type=Path,
                        default=Path.home() / "models/gemma-4-E2B-it")
    parser.add_argument("--output", type=Path,
                        default=Path("data/gemma-4-31B-it-embeddings.pt"))
    parser.add_argument("--download_shard", action="store_true",
                        help="Fallback if HTTP ranges are unavailable; needs ~47 GiB.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    revision = HfApi().model_info(args.model_id, revision=args.revision).sha
    def download(name):
        return Path(hf_hub_download(args.model_id, name, revision=revision))
    index = json.loads(download("model.safetensors.index.json").read_text())
    keys = [key for key in index["weight_map"] if key.endswith("embed_tokens.weight")]
    if len(keys) != 1:
        raise ValueError(f"Expected one token embedding matrix, found {keys}")
    key = keys[0]
    config = json.loads(download("config.json").read_text())
    tokenizer_dir = download("tokenizer.json").parent
    download("tokenizer_config.json")
    text_config = config.get("text_config", config)
    special_ids = check_tokenizer(args.tokenizer_path.expanduser(), tokenizer_dir,
                                  text_config["vocab_size"])
    print(f"Tokenizer compatibility verified. Revision: {revision}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + f".{revision}.safetensors.part")
    shard = index["weight_map"][key]
    if args.download_shard:
        tensor_path = download(shard)
    else:
        url = hf_hub_url(args.model_id, shard, revision=revision)
        extract_range(url, key, partial)
        tensor_path = partial
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        embeddings = tensors.get_tensor(key)
        expected_shape = (text_config["vocab_size"], text_config["hidden_size"])
        if tuple(embeddings.shape) != expected_shape:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        temporary_output = args.output.with_suffix(".pt.tmp")
        torch.save(embeddings, temporary_output)
        temporary_output.replace(args.output)
        metadata = {
            "model_id": args.model_id, "revision": revision, "tensor_key": key,
            "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
            "tokenizer_path": str(args.tokenizer_path.expanduser().resolve()),
            "tokenizer_verified": True, **special_ids,
        }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(f"Saved {args.output}: {metadata['shape']}, {metadata['dtype']}")


if __name__ == "__main__":
    main()
