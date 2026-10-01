from dataclasses import FrozenInstanceError
import hashlib

import pytest

from data_preprocessing.coding_sources import (
    EXPANDED_SOURCES, SOURCES, expand_sources, normalize_record, source_tasks,
)


def test_registry_records_pinned_revisions_and_provenance():
    for name, source in SOURCES.items():
        manifest = source.to_manifest()
        assert manifest["name"] == name
        assert len(source.revision) == 40
        int(source.revision, 16)
        assert manifest["source_url"] == (
            f"https://huggingface.co/datasets/{source.dataset}/tree/{source.revision}"
        )
        assert manifest["license"] in {"mit", "odc-by", "apache-2.0", "cc-by-4.0"}
        assert manifest["split"] == "train"
        with pytest.raises(FrozenInstanceError):
            source.revision = "main"


def test_starcoder_preserves_original_group_and_unmodified_text():
    source = SOURCES["starcoder2"]
    instruction = "  Write a function.\r\n"
    response = "```python\r\ndef f():\r\n    return 1\r\n```\n"
    original = {"id": 0, "sha1": "original-seed", "instruction": instruction,
                "response": response}
    normalized = normalize_record(source, original, 7)
    assert normalized == {
        "instruction": instruction, "response": response,
        "sha1": "original-seed", "id": 0,
        "source": "starcoder2", "source_row": 7,
        "source_dataset": source.dataset, "source_config": None,
        "source_revision": source.revision, "source_license": "odc-by",
    }
    assert "source" not in original
    assert normalize_record(source, {"instruction": "x", "response": "y"}, 0)["sha1"] is None


def test_prompt_groups_are_stable_and_namespaced():
    source = SOURCES["opencoder_educational"]
    row = {"instruction": "Write\n a  function", "output": "def f():\n    pass",
           "seq_id": "task-1"}
    first = normalize_record(source, row, 1)
    other_row = {**row, "instruction": "Write a function", "seq_id": "task-2"}
    reordered = normalize_record(source, other_row, 999)
    other_source = normalize_record(SOURCES["opencoder_package"], row, 1)
    expected_digest = hashlib.sha256(b"Write a function").hexdigest()
    assert first["sha1"] == f"opencoder_educational:{expected_digest}"
    assert reordered["sha1"] == first["sha1"]
    assert other_source["sha1"] != first["sha1"]
    assert first["id"] == "task-1"
    assert first["response"] == row["output"]
    assert first["source_config"] == "educational_instruct"


@pytest.mark.parametrize("ids, expected", [
    ({"id": "primary", "seq_id": "secondary"}, "primary"),
    ({"id": None, "seq_id": "secondary"}, "secondary"),
    ({"id": 0}, 0),
    ({}, 12),
])
def test_codefeedback_column_mapping_language_and_source_ids(ids, expected):
    row = {"query": "A question", "answer": "An answer", "lang": "Python3", **ids}
    normalized = normalize_record(SOURCES["codefeedback"], row, 12)
    assert normalized["instruction"] == "A question"
    assert normalized["response"] == "An answer"
    assert normalized["language"] == "Python3"
    assert normalized["id"] == expected


def test_chat_sources_use_the_first_exchange_and_the_id_field():
    source = SOURCES["ling_coder"]
    row = {"mid": "m-1", "messages": [
        {"role": "SYSTEM", "content": "Be brief."},
        {"role": "HUMAN", "content": "Write Rust."},
        {"role": "ASSISTANT", "content": "```rust\nfn main() {}\n```"},
        {"role": "HUMAN", "content": "Now in Go."},
        {"role": "ASSISTANT", "content": "```go\nfunc main() {}\n```"},
    ]}
    normalized = normalize_record(source, row, 3)
    assert normalized["instruction"] == "Write Rust."
    assert normalized["response"] == "```rust\nfn main() {}\n```"
    assert normalized["id"] == "m-1"
    assert normalized["sha1"] == f"ling_coder:{hashlib.sha256(b'Write Rust.').hexdigest()}"
    for messages in (None, [{"role": "ASSISTANT", "content": "x"}],
                     [{"role": "HUMAN", "content": "x"}, {"role": "HUMAN", "content": "y"}]):
        unanswered = normalize_record(source, {"messages": messages}, 4)
        assert unanswered["instruction"] is None and unanswered["response"] is None
        assert unanswered["id"] == 4


def code_contests_problem(**overrides):
    return {"name": "1A. Theatre Square", "description": "Count flagstones.\n\nInput\n\nn m a\n",
            "input_file": "", "output_file": "",
            "solutions": {"language": [2, 2, 2, 4, 4, 3, 1, 0],
                          "solution": ["int main(){}", "int main(){return 0;}",
                                       "int main(){ return 1; }", "class A{}", "class Main{}",
                                       "print(1)\n", "print 1", "?"]},
            **overrides}


def test_code_contests_problems_become_one_shortest_solution_per_language():
    source = SOURCES["code_contests"]
    tasks = source_tasks(source, code_contests_problem(), 5)
    suffix = "program that solves this problem, reading from standard input and writing to " \
             "standard output."
    assert [(task["id"], task["response"]) for task in tasks] == [
        ("1A. Theatre Square:cpp", "```cpp\nint main(){}\n```"),
        ("1A. Theatre Square:python", "```python\nprint(1)\n```"),
        ("1A. Theatre Square:java", "```java\nclass A{}\n```"),
    ]
    assert [task["instruction"] for task in tasks] == [
        f"Count flagstones.\n\nInput\n\nn m a\n\nWrite a {language} {suffix}"
        for language in ("C++", "Python 3", "Java")
    ]
    for task in tasks:
        assert task["sha1"] == "code_contests:1A. Theatre Square"
        assert (task["source"], task["source_row"], task["source_license"]) == (
            "code_contests", 5, "cc-by-4.0")


def test_code_contests_edge_cases():
    source = SOURCES["code_contests"]
    files = source_tasks(source, code_contests_problem(
        input_file="input.txt", output_file="output.txt"), 0)
    assert files[0]["instruction"].endswith(
        "reading from the file input.txt and writing to the file output.txt.")
    assert all(task["instruction"] is None
               for task in source_tasks(source, code_contests_problem(description=" "), 0))
    assert source_tasks(source, code_contests_problem(
        solutions={"language": [1, 0], "solution": ["print 1", "?"]}), 0) == []
    fenced = source_tasks(source, code_contests_problem(
        solutions={"language": [3], "solution": ["s = '''\n```\n'''\nprint(s)"]}), 0)
    assert fenced[0]["response"] == "````python\ns = '''\n```\n'''\nprint(s)\n````"
    assert source_tasks(SOURCES["starcoder2"], {"instruction": "x", "response": "y"}, 2) == [
        normalize_record(SOURCES["starcoder2"], {"instruction": "x", "response": "y"}, 2)]


def test_source_expansion_preserves_order_and_excludes_optional_realuser():
    assert expand_sources(["expanded"]) == EXPANDED_SOURCES
    assert EXPANDED_SOURCES[0] == "starcoder2"
    assert "opencoder_realuser" not in EXPANDED_SOURCES
    assert expand_sources(["opencoder_realuser", "starcoder2"]) == (
        "opencoder_realuser", "starcoder2",
    )


@pytest.mark.parametrize("names", [
    [], ["expanded", "starcoder2"], ["starcoder2", "starcoder2"], ["unknown"],
])
def test_source_expansion_rejects_ambiguous_or_unknown_selections(names):
    with pytest.raises(ValueError):
        expand_sources(names)
