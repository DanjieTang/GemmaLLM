from collections import Counter
from contextlib import closing
from itertools import islice
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preprocessing import prepare_coding as coding
from data_preprocessing.coding_sources import EXPANDED_SOURCES, SOURCES, normalize_record
from test_tokenize_wikipedia import make_tokenizer


def split_group(split):
    return next(f"seed-{index}" for index in range(100)
                if coding.task_split(f"seed-{index}", 0, 0.3) == split)


def task(instruction, response, split="train", source="starcoder2", **extra):
    return {"instruction": instruction, "response": response,
            "sha1": split_group(split), "source": source, **extra}


def read_tasks(directory):
    return [json.loads(line) for split in ("train", "val")
            for line in (directory / split / "tasks.jsonl").read_text().splitlines()]


def test_multilingual_tasks_keep_code_and_only_check_python_syntax(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    rows = [
        task("Python", "```py\ndef answer():\n    return 42\n```"),
        task("JavaScript", "```js\nfunction answer() { return 42; }\n```", "val"),
        task("C++", "```c++\nint answer() { return 42; }\n```"),
        task("Shell", "~~~sh\nprintf 'hello\\n'\n~~~", "val"),
        task("Broken Python", "```python3\ndef broken(:\n```"),
        task("Uncompiled JavaScript", "```javascript\nfunction broken( {\n```"),
        task("Incomplete TypeScript", "```ts\nconst answer: number = 42;"),
        task("No labeled code", "The answer is forty-two."),
    ]
    output = tmp_path / "output"
    report = coding.curate(rows, tokenizer, output, val_fraction=0.3,
                           languages=coding.LANGUAGES, batch_size=16)
    assert report["counts"] == {
        "source_rows": 8, "accepted": 5, "invalid_python": 1,
        "incomplete_code_fence": 1, "no_code_block": 1,
    }
    accepted = {row["instruction"]: row for row in read_tasks(output)}
    assert set(report["languages"]) == {"python", "javascript", "cpp", "bash"}
    assert report["languages"]["javascript"]["tasks"] == 2
    assert accepted["Python"]["response"] == rows[0]["response"]
    assert accepted["Python"]["languages"] == ["python"]
    assert accepted["Shell"]["response"] == rows[3]["response"]


@pytest.mark.parametrize("opening, closing, expected", [
    ("```Python3", "```", "python"),
    ("~~~JS", "~~~~", "javascript"),
    ("````c#", "`````", "csharp"),
    ("```golang", "```", "go"),
])
def test_fence_aliases_and_indentation(opening, closing, expected):
    assert coding.code_blocks(f"{opening}\n    preserved indentation\n{closing}") == [
        (expected, "    preserved indentation\n"),
    ]


@pytest.mark.parametrize("response", [
    "```python\nx = 1", "````python\nx = 1\n```", "~~~python\nx = 1\n```",
])
def test_incomplete_or_mismatched_fences_are_rejected(response):
    assert coding.code_blocks(response) is None


def test_deduplication_is_global_within_a_tokenizer_batch_and_keeps_provenance(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    response = "```python\ndef increment(value):\n    return value + 1\n```"
    rows = [
        task("Increment a number", response, source="first", id="original", source_row=7,
             source_dataset="owner/dataset", source_config="config", source_revision="pin",
             source_license="mit"),
        task("Increment\n a   number", "```python\nx = 123\n```", source="second"),
        task("Same response", response, source="third"),
        task("Same code", "A different explanation.\n```py\n# comment\n"
             "def increment(value):\n  return value+1\n```", source="fourth"),
        task("Different task", "```python\ndef decrement(value):\n    return value - 1\n```",
             "val", source="second"),
    ]
    output = tmp_path / "output"
    report = coding.curate(rows, tokenizer, output, val_fraction=0.3,
                           batch_size=32, deduplicate_code=True)
    assert report["counts"] == {"source_rows": 5, "accepted": 2, "duplicate": 3}
    aggregate = Counter()
    for counts in report["source_counts"].values():
        aggregate.update(counts)
    assert dict(aggregate) == report["counts"]
    assert report["source_counts"]["second"] == {
        "source_rows": 2, "duplicate": 1, "accepted": 1,
    }
    first, second = read_tasks(output)
    assert first["source_id"] == "original"
    for key in ("source", "source_row", "source_dataset", "source_config",
                "source_revision", "source_license"):
        assert first[key] == rows[0][key]
    assert first["seed_sha1"] == rows[0]["sha1"]
    assert first["code_sha256"] != second["code_sha256"]


def test_complete_2048_token_tasks_survive_and_2049_token_tasks_are_dropped(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    rows = []
    for split in ("train", "val"):
        for total in (2047, 2048, 2049):
            instruction = f"{split} {total}"
            response = f"```python\nvalue = '{split}-{total}'\n```"
            baseline = len(tokenizer.encode(coding.format_task(instruction, response),
                                             add_special_tokens=False)) + 2
            instruction += " hello" * (total - baseline)
            assert len(tokenizer.encode(coding.format_task(instruction, response),
                                        add_special_tokens=False)) + 2 == total
            rows.append(task(instruction, response, split))
    output = tmp_path / "output"
    report = coding.curate(rows, tokenizer, output, context_length=2048, max_tokens=2048,
                           val_fraction=0.3, batch_size=4)
    assert report["counts"] == {"source_rows": 6, "accepted": 4, "too_long": 2}
    for split in ("train", "val"):
        tokens = np.load(output / split / "tokens.npy")
        offsets = np.load(output / split / "offsets.npy")
        assert np.diff(offsets).tolist() == [2047, 2048]
        metadata = [json.loads(line) for line in
                    (output / split / "tasks.jsonl").read_text().splitlines()]
        for index, row in enumerate(metadata):
            expected = [tokenizer.bos_token_id, *tokenizer.encode(
                coding.format_task(row["instruction"], row["response"]),
                add_special_tokens=False), tokenizer.eos_token_id]
            assert tokens[offsets[index]:offsets[index + 1]].tolist() == expected


def source_rows(source):
    rows = []
    for split in ("train", "val"):
        for index in range(100):
            row = {
                source.instruction_field: f"{source.name} {split} task {index}",
                source.response_field: f"```python\ndef {source.name}_{split}():\n    return {index}\n```",
                "id": f"{source.name}-{split}", "sha1": split_group(split),
            }
            if source.language_field:
                row[source.language_field] = "Python"
            normalized = normalize_record(source, row, len(rows))
            if coding.task_split(normalized["sha1"], 0, 0.3) == split:
                rows.append(row)
                break
        else:
            raise AssertionError(f"No synthetic {split} example found for {source.name}")
    return rows


def test_expanded_main_pins_sources_records_configs_and_limits_each_stream(tmp_path, monkeypatch):
    make_tokenizer(tmp_path / "tokenizer")
    revision_calls, reader_calls, closed_sources = [], [], []

    def dataset_info(dataset, revision):
        revision_calls.append((dataset, revision))
        return SimpleNamespace(sha=revision)

    def load_source_rows(source):
        reader_calls.append(source)
        try:
            yield from source_rows(source)
            raise AssertionError("The per-source row limit consumed an extra record")
        finally:
            closed_sources.append(source.name)

    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(dataset_info=dataset_info))
    monkeypatch.setattr(coding, "load_source_rows", load_source_rows)
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", [
        "prepare_coding", "--output_dir", str(output), "--tokenizer_path",
        str(tmp_path / "tokenizer"), "--sources", "expanded", "--languages", "all",
        "--context_length", "2048", "--max_tokens", "2048", "--max_source_rows", "2",
        "--val_fraction", "0.3", "--batch_size", "4",
    ])
    coding.main()
    manifest = json.loads((output / "manifest.json").read_text())
    sources = [SOURCES[name] for name in EXPANDED_SOURCES]
    assert revision_calls == [(source.dataset, source.revision) for source in sources]
    assert reader_calls == sources
    assert closed_sources == list(EXPANDED_SOURCES)
    assert manifest["config"]["sources"] == [source.to_manifest() for source in sources]
    assert manifest["config"]["language"] == "multilingual"
    assert manifest["config"]["max_tokens"] == 2048
    assert manifest["counts"] == {"source_rows": 10, "accepted": 10}
    assert all(stats["tasks"] == 5 for stats in manifest["splits"].values())
    assert set(manifest["source_counts"]) == set(EXPANDED_SOURCES)
    assert not (tmp_path / ".output.incomplete").exists()
    for row in read_tasks(output):
        source = SOURCES[row["source"]]
        assert row["source_revision"] == source.revision
        assert row["source_config"] == source.config


def test_source_failure_preserves_incomplete_output_without_publishing_manifest(tmp_path, monkeypatch):
    make_tokenizer(tmp_path / "tokenizer")
    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda dataset, revision: SimpleNamespace(sha=revision)))

    def failing_source(*args, **kwargs):
        yield from source_rows(SOURCES["starcoder2"])
        raise RuntimeError("Source stream interrupted")

    monkeypatch.setattr(coding, "load_source_rows", failing_source)
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", [
        "prepare_coding", "--output_dir", str(output), "--tokenizer_path",
        str(tmp_path / "tokenizer"), "--val_fraction", "0.3", "--batch_size", "1",
    ])
    with pytest.raises(RuntimeError, match="Source stream interrupted"):
        coding.main()
    staging = tmp_path / ".output.incomplete"
    assert not output.exists()
    assert (staging / "config.json").is_file()
    assert not (staging / "manifest.json").exists()
    assert all((staging / split / "tokens.bin").stat().st_size > 0
               for split in ("train", "val"))
    with pytest.raises(ValueError, match="already exists"):
        coding.main()


@pytest.mark.parametrize("limit", [None, 2, 3])
@pytest.mark.parametrize("layout", ["data/{config}-", "{config}/"])
def test_pinned_parquet_reader_orders_shards_and_closes_at_row_limit(
        tmp_path, monkeypatch, limit, layout):
    source = SOURCES["opencoder_educational"]
    prefix = layout.format(config=source.config)
    first = f"{prefix}train-00000-of-00002.parquet"
    second = f"{prefix}train-00001-of-00002.parquet"
    local_paths = {first: tmp_path / "first.parquet", second: tmp_path / "second.parquet"}
    pq.write_table(pa.table({"value": [0, 1]}), local_paths[first])
    pq.write_table(pa.table({"value": [2, 3]}), local_paths[second])
    listing_calls, downloads, opened, closed, batch_options = [], [], [], [], []

    def list_repo_files(dataset, **kwargs):
        listing_calls.append((dataset, kwargs))
        return [second, "README.md", "data/package_instruct-train.parquet", first]

    def download(dataset, path, **kwargs):
        downloads.append((dataset, path, kwargs))
        return local_paths[path]

    real_parquet_file = pq.ParquetFile

    class TrackedParquetFile:
        def __init__(self, path):
            self.path = path
            self.parquet = real_parquet_file(path)
            opened.append(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.parquet.close()
            closed.append(self.path)

        def iter_batches(self, **kwargs):
            batch_options.append(kwargs)
            yield from self.parquet.iter_batches(**kwargs)

    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(list_repo_files=list_repo_files))
    monkeypatch.setattr(coding, "hf_hub_download", download)
    monkeypatch.setattr(coding.pq, "ParquetFile", TrackedParquetFile)
    with closing(coding.load_source_rows(source)) as iterator:
        rows = list(iterator if limit is None else islice(iterator, limit))
    count = 4 if limit is None else limit
    assert rows == [{"value": value} for value in range(count)]
    assert listing_calls == [(source.dataset, {"repo_type": "dataset", "revision": source.revision})]
    reached = [first] if limit == 2 else [first, second]
    assert downloads == [(source.dataset, path, {"repo_type": "dataset", "revision": source.revision})
                         for path in reached]
    assert opened == closed == [local_paths[path] for path in reached]
    assert all(options["use_threads"] is False and 0 < options["batch_size"] <= 1024
               for options in batch_options)


def test_parquet_reader_fails_clearly_when_selected_config_has_no_shards(monkeypatch):
    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(
        list_repo_files=lambda *args, **kwargs: ["data/package_instruct-train.parquet"]))
    monkeypatch.setattr(coding, "hf_hub_download", lambda *args, **kwargs: pytest.fail(
        "A different source config must never be downloaded"))
    with pytest.raises(ValueError, match="No Parquet shards found for opencoder_educational"):
        list(coding.load_source_rows(SOURCES["opencoder_educational"]))


def test_codefeedback_reader_keeps_pinned_json_streaming_and_closes_early(monkeypatch):
    source = SOURCES["codefeedback"]
    calls, closed = [], []

    def load_dataset(dataset, **kwargs):
        calls.append((dataset, kwargs))
        try:
            yield {"query": "A question", "answer": "An answer", "lang": "python"}
            pytest.fail("The preview fetched an extra JSON record")
        finally:
            closed.append(True)

    monkeypatch.setattr(coding, "load_dataset", load_dataset)
    with closing(coding.load_source_rows(source)) as iterator:
        assert len(list(islice(iterator, 1))) == 1
    assert calls == [(source.dataset, {"split": "train", "revision": source.revision,
                                       "streaming": True})]
    assert closed == [True]


def test_cli_language_aliases_and_expansion():
    args = coding.parse_args(["--sources", "expanded", "--languages", "js", "py", "python3"])
    assert args.sources == EXPANDED_SOURCES
    assert args.languages == ("javascript", "python")


@pytest.mark.parametrize("flags", [
    ["--sources", "expanded", "starcoder2"], ["--sources", "starcoder2", "starcoder2"],
    ["--languages", "all", "python"], ["--batch_size", "0"], ["--max_source_rows", "0"],
])
def test_invalid_expansion_options_fail_early(flags):
    with pytest.raises(SystemExit):
        coding.parse_args(flags)
