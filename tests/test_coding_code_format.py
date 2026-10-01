import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preprocessing import prepare_coding as coding
from data_preprocessing.coding_sources import MULTILINGUAL_SOURCES, SOURCES, expand_sources
from lazy_dataloader import prepare_coding_dataset
from test_coding_expansion import read_tasks, task
from test_tokenize_wikipedia import make_tokenizer


def curate_code(tmp_path, rows, **options):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    output = tmp_path / "output"
    report = coding.curate(rows, tokenizer, output, context_length=2048, max_tokens=2048,
                           val_fraction=0.3, languages=coding.LANGUAGES, batch_size=8,
                           response_format="code", **options)
    tasks = {row["source_id"]: row for row in read_tasks(output)}
    return report, tasks, tokenizer, output


def test_code_format_keeps_exactly_one_canonical_solution(tmp_path):
    rows = [
        task("Prose around code", "Here is the code:\n\n```py\n\ndef one():\n    return 1\n\n```"
             "\n\nIt returns 1.", id="prose"),
        task("Setup, output and data blocks", "Install it:\n```bash\npip install numpy\n```\n"
             "```python\ndef two():\n    return 2\n```\nOutput:\n```\n2\n```\n"
             "```json\n{\"value\": 2}\n```", "val", id="auxiliary"),
        task("Configuration only", "```yml\nname: ci\non: push\n```", id="config"),
        task("Shell only", "```sh\npip install numpy\n```", "val", id="shell"),
        task("Nested fence", "````python\nTEMPLATE = '''\n```js\nx\n```\n'''\n````", id="nested"),
        task("Two languages", "```python\nx = 1\n```\n```javascript\nlet x = 1;\n```", id="mixed"),
        task("Two Python blocks", "```python\nx = 1\n```\nThen:\n```python\ny = 2\n```",
             id="steps"),
        task("Prose only", "Use a loop.", id="prose-only"),
        task("Output only", "```text\n42\n```", id="output-only"),
        task("JSON-wrapped prose", "```json\n{\"steps\": [\"Open Settings\"]}\n```",
             id="json-only"),
        task("Unknown language", "```brainfuck\n++++\n```", id="unknown"),
        task("Broken Python", "```python\ndef broken(:\n```", id="broken"),
        task("Incomplete fence", "```python\nx = 1", id="incomplete"),
    ]
    report, accepted, tokenizer, output = curate_code(tmp_path, rows)
    assert report["counts"] == {
        "source_rows": 13, "accepted": 5, "multiple_code_blocks": 2, "no_code_block": 3,
        "unsupported_language": 1, "invalid_python": 1, "incomplete_code_fence": 1,
    }
    assert {key: row["response"] for key, row in accepted.items()} == {
        "prose": "```python\ndef one():\n    return 1\n```",
        "auxiliary": "```python\ndef two():\n    return 2\n```",
        "config": "```yaml\nname: ci\non: push\n```",
        "shell": "```bash\npip install numpy\n```",
        "nested": "````python\nTEMPLATE = '''\n```js\nx\n```\n'''\n````",
    }
    assert accepted["auxiliary"]["edits"] == ["extra_blocks_removed"]
    assert accepted["prose"]["edits"] == []
    assert set(report["languages"]) == {"python", "yaml", "bash"}
    for split in ("train", "val"):
        tokens = np.load(output / split / "tokens.npy")
        offsets = np.load(output / split / "offsets.npy")
        rows = [json.loads(line) for line in
                (output / split / "tasks.jsonl").read_text().splitlines()]
        for index, row in enumerate(rows):
            expected = [tokenizer.bos_token_id, *tokenizer.encode(
                coding.format_task(row["instruction"], row["response"]),
                add_special_tokens=False), tokenizer.eos_token_id]
            assert tokens[offsets[index]:offsets[index + 1]].tolist() == expected


def test_code_format_reads_fences_nested_in_lists(tmp_path):
    rows = [
        task("Flask app", "1. Install:\n     ```bash\n     pip install flask\n     ```\n"
             "2. Write `app.py`:\n     ```python\n     from flask import Flask\n\n"
             "     app = Flask(__name__)\n     ```\n3. Run it:\n  ```bash\n  python app.py\n  ```",
             id="nested"),
        task("Steps", "1. First:\n   ```python\n   x = 1\n   ```\n2. Then:\n   ```python\n"
             "   y = 2\n   ```", id="steps"),
        task("Hidden program", "```\nprint('app')\n```\nRun:\n```bash\npython app.py\n```",
             id="unlabeled"),
        task("Fence inside code", "```python\ndef fence():\n    return '''\n    ```\n    '''\n```",
             "val", id="inner"),
    ]
    report, accepted, *_ = curate_code(tmp_path, rows)
    assert report["counts"] == {"source_rows": 4, "accepted": 2, "multiple_code_blocks": 2}
    assert accepted["nested"]["response"] == (
        "```python\nfrom flask import Flask\n\napp = Flask(__name__)\n```")
    assert accepted["nested"]["edits"] == ["extra_blocks_removed"]
    assert accepted["inner"]["response"] == rows[3]["response"]
    assert coding.indented_code_blocks("  ```js\n  let x;\n```") is None


def test_code_format_keeps_a_complete_program_that_contains_every_step(tmp_path):
    rows = [
        task("Steps then program", "Import:\n```python\nimport math\n```\nCompute:\n"
             "```python\nroot = math.sqrt(9)\n```\nComplete:\n```python\nimport math\n\n"
             "root = math.sqrt(9)\nprint(root)\n```", id="complete"),
        task("Explanation repeats a call", "```python\nimport numpy as np\n\n"
             "values = np.sqrt([4, 9])\n```\nHere `np.sqrt` is applied:\n"
             "```python\nnp.sqrt([4, 9])\n```", "val", id="snippet"),
        task("Diverging versions", "```python\nx = 1\n```\nBetter:\n```python\nx = 2\n```",
             id="versions"),
        task("Prefix only", "```python\nx = 1\n```\nOr:\n```python\nx = 10\n```", id="prefix"),
        task("Other language", "```python\nx = 1\n```\n```javascript\nlet x = 1;\n```",
             id="languages"),
    ]
    report, accepted, *_ = curate_code(tmp_path, rows)
    assert report["counts"] == {"source_rows": 5, "accepted": 2, "multiple_code_blocks": 3}
    assert accepted["complete"]["response"] == (
        "```python\nimport math\n\nroot = math.sqrt(9)\nprint(root)\n```")
    assert accepted["snippet"]["response"] == (
        "```python\nimport numpy as np\n\nvalues = np.sqrt([4, 9])\n```")


def test_code_format_cleans_leaked_answers_and_rejects_copied_solutions(tmp_path):
    solution = "def add(left, right):\n    return left + right"
    copied = solution.replace("add", "plus")
    rows = [
        task("**Question**: Add two numbers in Python.\n\n**Answer**:\n```python\n"
             f"{solution}\n```", f"Sure:\n```python\n{solution}\n```", id="leaked"),
        task("**Answer** the following: write a subtraction function.",
             "```python\ndef sub(a, b):\n    return a - b\n```", "val", id="header"),
        task(f"My code:\n{copied}\nPlease repeat it.", f"```python\n{copied}\n```", id="copied"),
        task("[PYTHON]\ndef f(x): return x\nassert f(1) == ??\n[/PYTHON]\n[THOUGHT]",
             "```python\nassert f(1) == 1\n```", id="execution"),
        task("**Answer**: nothing precedes the leak", "```python\nx = 1\n```", id="empty"),
    ]
    report, accepted, *_ = curate_code(tmp_path, rows)
    assert report["counts"] == {"source_rows": 5, "accepted": 2, "solution_in_prompt": 1,
                                "execution_prediction": 1, "empty_prompt": 1}
    assert accepted["leaked"]["instruction"] == "Add two numbers in Python."
    assert accepted["leaked"]["edits"] == ["answer_section_removed", "question_label_removed"]
    assert accepted["header"]["instruction"] == rows[1]["instruction"]
    assert report["edits"]["starcoder2"] == {"answer_section_removed": 1,
                                            "question_label_removed": 1}


def test_code_format_deduplicates_identical_solutions_with_different_prose(tmp_path):
    rows = [
        task("Square a number", "Easy:\n```python\ndef square(x):\n    return x * x\n```",
             id="first"),
        task("Compute a square", "```py\ndef square(x):\n    return x * x\n```\nDone.",
             id="second"),
        task("Cube a number", "```python\ndef cube(x):\n    return x ** 3\n```", "val",
             id="third"),
    ]
    report, accepted, *_ = curate_code(tmp_path, rows)
    assert report["counts"] == {"source_rows": 3, "accepted": 2, "duplicate": 1}
    assert set(accepted) == {"first", "third"}


@pytest.mark.parametrize("response_format, accepted_ids", [
    ("code", {"plain"}), ("original", {"plain", "marker"}),
])
def test_code_format_rejects_text_that_tokenizes_to_structural_tokens(
        tmp_path, response_format, accepted_ids):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    rows = [task("Print hello", "```python\nprint('hello')\n```", id="plain"),
            task("Stop at [EOS] early", "```python\nprint('world')\n```", "val", id="marker"),
            task("Another hello", "```python\nprint('hello world')\n```", "val", id="val")]
    report = coding.curate(rows, tokenizer, tmp_path / "output", val_fraction=0.3,
                           batch_size=4, response_format=response_format)
    kept = {row["source_id"] for row in read_tasks(tmp_path / "output")}
    assert kept - {"val"} == accepted_ids
    assert report["counts"].get("special_token_text", 0) == (response_format == "code")


def test_source_language_exclusions_only_affect_their_source(tmp_path):
    rows = [
        task("Ling Python", "```python\nx = 1\n```", source="ling_coder", id="ling-python"),
        task("Ling Rust", "```rust\nfn main() {}\n```", source="ling_coder", id="ling-rust"),
        task("Other Python", "```python\ny = 2\n```", "val", source="opencoder_mceval",
             id="other-python"),
    ]
    report, accepted, *_ = curate_code(tmp_path, rows,
                                       excluded_languages={"ling_coder": ("python",)})
    assert set(accepted) == {"ling-rust", "other-python"}
    assert report["source_counts"]["ling_coder"] == {
        "source_rows": 2, "excluded_language": 1, "accepted": 1,
    }


def test_original_format_is_unchanged_by_default(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    response = "Here it is:\n```python\nx = 1\n```\nOutput:\n```\n1\n```"
    rows = [task("Keep prose", response, id="train"),
            task("Keep this too", "```python\ny = 2\n```", "val", id="val")]
    report = coding.curate(rows, tokenizer, tmp_path / "output", val_fraction=0.3)
    assert "edits" not in report
    kept = {row["source_id"]: row for row in read_tasks(tmp_path / "output")}
    assert kept["train"]["response"] == response
    assert "edits" not in kept["train"]


@pytest.mark.parametrize("label, expected", [
    ('Python3 title="a.py"', "python"), ("js:app.js", "javascript"), ("F#", "fsharp"),
    ("VB.NET", "vbnet"), ("PL/SQL", "sql"), ("yml", "yaml"), ("Dockerfile", "dockerfile"),
    ("rkt", "racket"), ("", ""),
])
def test_fence_info_strings_map_to_canonical_tags(label, expected):
    assert coding.canonical_language(label) == expected


def test_language_tables_are_consistent():
    languages = set(coding.LANGUAGES)
    assert len(languages) == len(coding.LANGUAGES)
    assert set(coding.LANGUAGE_ALIASES.values()) <= languages | coding.AUXILIARY_LABELS
    assert not coding.AUXILIARY_LABELS & languages
    assert coding.SECONDARY_LANGUAGES <= languages
    assert coding.SETUP_SHELLS <= languages
    for source in SOURCES.values():
        assert set(source.exclude_languages) <= languages


@pytest.mark.parametrize("code, expected", [
    ("pip install numpy\n$ python app.py\n", True),
    ("# setup\nnpm install express\nnode server.js", True),
    ("cd build && make", True),
    ("cp a.txt b.txt\n", False),
    ("for f in *.txt; do echo $f; done", False),
    ("\n", False),
    ("pip install x\n" * 9, False),
])
def test_setup_blocks_are_short_install_or_run_commands(code, expected):
    assert coding.is_setup_block("bash", code) is expected
    assert coding.is_setup_block("python", code) is False


def test_multilingual_preset_orders_sources_and_rejects_mixing():
    assert expand_sources(["multilingual"]) == MULTILINGUAL_SOURCES
    assert set(MULTILINGUAL_SOURCES) <= set(SOURCES)
    assert MULTILINGUAL_SOURCES[0] == "starcoder2"
    assert MULTILINGUAL_SOURCES[-1] == "opencoder_diverse"
    for names in (["multilingual", "starcoder2"], ["expanded", "multilingual"]):
        with pytest.raises(ValueError):
            expand_sources(names)
    args = coding.parse_args(["--sources", "multilingual", "--response_format", "code"])
    assert args.sources == MULTILINGUAL_SOURCES and args.response_format == "code"
    with pytest.raises(SystemExit):
        coding.parse_args(["--response_format", "prose"])


def test_uncached_shards_use_a_temporary_directory_and_are_deleted(tmp_path, monkeypatch):
    source = SOURCES["ling_coder"]
    assert source.cache_shards is False
    shards = ["data/train_00001_of_00002.parquet", "data/train_00000_of_00002.parquet"]
    local_dirs, downloads = [], []

    def download(dataset, path, **kwargs):
        local_dirs.append(Path(kwargs["local_dir"]))
        downloads.append((dataset, path, kwargs["repo_type"], kwargs["revision"]))
        target = local_dirs[-1] / path
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"value": [path, path]}), target)
        return str(target)

    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(
        list_repo_files=lambda *args, **kwargs: [*shards, "README.md"]))
    monkeypatch.setattr(coding, "hf_hub_download", download)
    values = []
    for row in coding.load_source_rows(source):
        values.append(row["value"])
        on_disk = [path.relative_to(local_dirs[0]).as_posix()
                   for path in local_dirs[0].rglob("*.parquet")]
        assert on_disk == [row["value"]]
    assert values == [shard for shard in sorted(shards) for _ in range(2)]
    assert downloads == [(source.dataset, shard, "dataset", source.revision)
                         for shard in sorted(shards)]
    assert len(set(local_dirs)) == 1 and not local_dirs[0].exists()
    iterator = coding.load_source_rows(source)
    next(iterator)
    iterator.close()
    assert not local_dirs[-1].exists()


def test_code_contests_reads_train_shards_and_keeps_problems_in_one_split(tmp_path, monkeypatch):
    make_tokenizer(tmp_path / "tokenizer")
    source = SOURCES["code_contests"]
    names = {split: [name for name in (f"problem {index}" for index in range(200))
                     if coding.task_split(f"code_contests:{name}", 0, 0.3) == split][:3]
             for split in ("train", "val")}
    problems = [{"name": name, "description": f"Print hello for {name}.",
                 "input_file": "", "output_file": "", "generated_tests": {"input": ["x" * 50]},
                 "solutions": {"language": [2, 3, 4],
                               "solution": [f"int main() {{ return {index}; }}",
                                            f"print({index})", f"class Main {{ int x = {index}; }}"]}}
                for index, name in enumerate(names["train"] + names["val"])]
    shards = ["data/test-00000-of-00001.parquet", "data/train-00001-of-00002.parquet",
              "data/train-00000-of-00002.parquet", "data/valid-00000-of-00001.parquet"]
    downloads, decoded_columns = [], []

    def download(dataset, path, **kwargs):
        downloads.append(path)
        target = Path(kwargs["local_dir"]) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        half = problems[:3] if "00000-of-00002" in path else problems[3:]
        pq.write_table(pa.Table.from_pylist(half if "/train-" in path else problems), target)
        return str(target)

    real_parquet_file = pq.ParquetFile

    class TrackedParquetFile(real_parquet_file):
        def iter_batches(self, **kwargs):
            decoded_columns.append(kwargs["columns"])
            return super().iter_batches(**kwargs)

    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda dataset, revision: SimpleNamespace(sha=revision),
        list_repo_files=lambda *args, **kwargs: shards))
    monkeypatch.setattr(coding, "hf_hub_download", download)
    monkeypatch.setattr(coding.pq, "ParquetFile", TrackedParquetFile)
    output = tmp_path / "coding"
    monkeypatch.setattr("sys.argv", [
        "prepare_coding", "--output_dir", str(output), "--tokenizer_path",
        str(tmp_path / "tokenizer"), "--sources", "code_contests", "--response_format", "code",
        "--languages", "all", "--context_length", "512", "--max_tokens", "512",
        "--val_fraction", "0.3",
    ])
    coding.main()
    assert downloads == ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    assert decoded_columns == [list(source.columns)] * 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"source_rows": 18, "accepted": 18}
    assert manifest["config"]["sources"][0]["columns"] == list(source.columns)
    assert {language: stats["tasks"] for language, stats in manifest["languages"].items()} == {
        "cpp": 6, "python": 6, "java": 6}
    for split in ("train", "val"):
        rows = [json.loads(line) for line in
                (output / split / "tasks.jsonl").read_text().splitlines()]
        assert sorted({row["source_id"].rsplit(":", 1)[0] for row in rows}) == sorted(names[split])
        assert all(row["response"].startswith(f"```{row['languages'][0]}\n") for row in rows)


def test_code_format_main_records_settings_and_feeds_the_training_loader(tmp_path, monkeypatch):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    rows = [{"instruction": f"**Question**: hello {index}", "id": index, "sha1": f"seed-{index}",
             "response": f"Explanation.\n```python\nvalue_{index} = {index}\n```"}
            for index in range(30)]
    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="pinned")))
    monkeypatch.setattr(coding, "load_source_rows", lambda source: iter(rows))
    output = tmp_path / "coding"
    monkeypatch.setattr("sys.argv", [
        "prepare_coding", "--output_dir", str(output), "--tokenizer_path",
        str(tmp_path / "tokenizer"), "--response_format", "code", "--languages", "all",
        "--context_length", "64", "--max_tokens", "64", "--val_fraction", "0.3",
    ])
    coding.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["response_format"] == "code"
    assert {"response_normalization", "prompt_normalization"} <= set(manifest["config"])
    assert manifest["counts"] == {"source_rows": 30, "accepted": 30}
    assert manifest["edits"] == {"starcoder2": {"question_label_removed": 30}}
    for row in read_tasks(output):
        assert row["instruction"] == f"hello {row['source_id']}"
        assert row["response"] == (f"```python\nvalue_{row['source_id']} = "
                                   f"{row['source_id']}\n```")
    training, validation = prepare_coding_dataset(output, 4, 64, 16, tokenizer)
    for loader in (training, validation):
        batch = next(iter(loader))
        assert batch["input_ids"][:, 0].eq(tokenizer.bos_token_id).all()
