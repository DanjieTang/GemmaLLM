import json
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from data_preprocessing import tokenize_wikipedia as wiki


@pytest.mark.parametrize("length", [2, 4, 5, 6, 8, 9, 10, 18])
def test_windows_preserve_every_prediction_pair_once(length):
    ids = np.arange(length, dtype=np.int32)
    windows = list(wiki.iter_windows(ids, 4))
    pairs = [(int(a), int(b)) for window in windows
             for a, b in zip(window[:-1], window[1:])]
    assert pairs == list(zip(range(length - 1), range(1, length)))
    assert all(2 <= len(window) <= 5 for window in windows)
    assert sum(len(window) - 1 for window in windows) == length - 1


def make_tokenizer(path):
    vocab = {token: index for index, token in enumerate(
        ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "Title", "hello", "world",
         "中文", "日本語", "русский", "español", "français", "Deutsch"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def read_windows(directory, split):
    manifest = json.loads((directory / "manifest.json").read_text())
    windows = []
    for shard in manifest["splits"][split]["shards"]:
        tokens = np.load(directory / split / shard["tokens"], mmap_mode="r")
        lengths = np.load(directory / split / shard["lengths"], mmap_mode="r")
        assert tokens.dtype == lengths.dtype == np.int32
        assert tokens.shape == (shard["rows"], 5)
        for row, length in zip(tokens, lengths):
            assert np.all(row[length:] == 0)
            windows.append(row[:length].tolist())
    return windows


def test_full_pipeline_retains_duplicates_unicode_tails_and_pins_source(tmp_path, monkeypatch):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    articles = [
        {"id": str(i), "url": f"https://example.test/{i}", "title": "Title",
         "text": "hello world 中文 日本語 русский español français Deutsch " * 3}
        for i in range(20)
    ]
    articles.extend([articles[0].copy(),
                     {"id": "empty", "url": "", "title": "", "text": ""}])
    dataset_calls = []
    revision_calls = []

    def load_dataset(*args, **kwargs):
        dataset_calls.append((args, kwargs))
        return iter(articles)

    def dataset_info(*args, **kwargs):
        revision_calls.append((args, kwargs))
        return SimpleNamespace(sha="fixed-revision")

    monkeypatch.setattr(wiki, "load_dataset", load_dataset)
    monkeypatch.setattr(wiki, "HfApi", lambda: SimpleNamespace(dataset_info=dataset_info))
    monkeypatch.setattr("sys.argv", [
        "tokenize_wikipedia.py", "--output_dir", str(tmp_path / "output"),
        "--tokenizer_path", str(tmp_path / "tokenizer"), "--languages", "en", "zh",
        "--context_length", "4", "--batch_size", "3", "--shard_rows", "2",
        "--val_fraction", "0.5",
    ])
    wiki.main()
    assert len(dataset_calls) == 2
    assert all(kwargs == {"split": "train", "revision": "fixed-revision", "streaming": True}
               for _, kwargs in dataset_calls)
    assert [args[1] for args, _ in dataset_calls] == ["20231101.en", "20231101.zh"]
    for language in ["en", "zh"]:
        directory = tmp_path / "output" / language
        manifest = json.loads((directory / "manifest.json").read_text())
        assert sum(stats["articles"] for stats in manifest["splits"].values()) == len(articles)
        split_ids = {}
        for split in ["train", "val"]:
            expected = [a for a in articles if wiki.article_split(language, a["id"], 0, .5) == split]
            windows = read_windows(directory, split)
            metadata = [json.loads(line) for line in
                        (directory / split / "articles.jsonl").read_text().splitlines()]
            assert len(metadata) == len(expected)
            split_ids[split] = {a["id"] for a in metadata}
            for article, record in zip(expected, metadata):
                body = tokenizer(article["title"] + "\n\n" + article["text"],
                                 add_special_tokens=False)["input_ids"]
                expected_ids = [tokenizer.bos_token_id, *body, tokenizer.eos_token_id]
                chunks = windows[record["first_row"]:record["first_row"] + record["rows"]]
                reconstructed = chunks[0] + [token for chunk in chunks[1:] for token in chunk[1:]]
                assert reconstructed == expected_ids
                assert record["article_tokens"] == len(expected_ids)
            assert manifest["splits"][split]["target_tokens"] == sum(len(w) - 1 for w in windows)
        assert split_ids["train"].isdisjoint(split_ids["val"])
    # A rerun must not download/tokenize completed editions or resolve main again.
    wiki.main()
    assert len(dataset_calls) == 2
    assert len(revision_calls) == 1


def test_restart_only_matching_incomplete_language(tmp_path):
    config = {"language": "en", "context_length": 256}
    staging = wiki.prepare_language(tmp_path, config, False)
    (staging / "partial.npy").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="restart_incomplete"):
        wiki.prepare_language(tmp_path, config, False)
    with pytest.raises(ValueError, match="different settings"):
        wiki.prepare_language(tmp_path, {**config, "context_length": 128}, True)
    assert (staging / "partial.npy").exists()
    assert wiki.prepare_language(tmp_path, config, True) == staging
    assert not (staging / "partial.npy").exists()
    wiki.save_json(staging / "manifest.json", {"config": config})
    staging.rename(tmp_path / "en")
    assert wiki.prepare_language(tmp_path, config, True) is None
    with pytest.raises(ValueError, match="different settings"):
        wiki.prepare_language(tmp_path, {**config, "context_length": 128}, False)


@pytest.mark.parametrize("flags", [
    ["--context_length", "0"], ["--batch_size", "0"], ["--shard_rows", "0"],
    ["--val_fraction", "nan"], ["--val_fraction", "1"],
    ["--languages", "../data"], ["--languages", "en", "en"],
])
def test_invalid_options_fail_early(flags):
    with pytest.raises(SystemExit):
        wiki.parse_args(flags)


def test_defaults_use_all_seven_editions_and_256_inputs():
    args = wiki.parse_args([])
    assert args.languages == ["en", "zh", "es", "fr", "de", "ja", "ru"]
    assert args.context_length == 256
    assert args.val_fraction == 0.01
