import struct
from unittest.mock import patch

import httpx
import pytest
from safetensors import safe_open
from safetensors.torch import save
import torch

from data_preprocessing.download_embeddings import extract_range


def test_extract_only_selected_tensor_and_resume(tmp_path):
    wanted = torch.arange(24).reshape(6, 4).to(torch.bfloat16)
    payload = save({"embed_tokens.weight": wanted, "other": torch.randn(3, 8)})
    ranges = []

    def serve(request):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        ranges.append((start, end))
        return httpx.Response(206, content=payload[start:end + 1],
                              headers={"content-range": f"bytes {start}-{end}/{len(payload)}"})

    client_class = httpx.Client
    def client(**kwargs):
        return client_class(transport=httpx.MockTransport(serve), **kwargs)

    destination = tmp_path / "partial.safetensors"
    with patch("data_preprocessing.download_embeddings.httpx.Client", side_effect=client):
        info = extract_range("https://example.com/weights", "embed_tokens.weight", destination)
        with safe_open(destination, framework="pt") as tensors:
            assert list(tensors.keys()) == ["embed_tokens.weight"]
            torch.testing.assert_close(tensors.get_tensor("embed_tokens.weight"), wanted)
        file_bytes = destination.read_bytes()
        prefix_size = 8 + struct.unpack("<Q", file_bytes[:8])[0]
        destination.write_bytes(file_bytes[:prefix_size + 3])
        ranges.clear()
        extract_range("https://example.com/weights", "embed_tokens.weight", destination)
        assert destination.read_bytes() == file_bytes
        source_header_size = struct.unpack("<Q", payload[:8])[0]
        assert ranges[-1][0] == 8 + source_header_size + info["data_offsets"][0] + 3


def test_server_ignoring_range_does_not_download_full_shard(tmp_path):
    client_class = httpx.Client
    def client(**kwargs):
        return client_class(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"ignored range")
        ), **kwargs)
    with patch("data_preprocessing.download_embeddings.httpx.Client", side_effect=client):
        with pytest.raises(RuntimeError, match="did not honor"):
            extract_range("https://example.com/weights", "embed_tokens.weight",
                          tmp_path / "partial")
    assert not (tmp_path / "partial").exists()
