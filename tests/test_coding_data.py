import json
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data_preprocessing import prepare_coding as coding
from lazy_dataloader import collate_annotations, prepare_coding_dataset
from test_tokenize_wikipedia import make_tokenizer
from test_vlm import FakeVisionModel, FakeVisionProcessor
import train
import train_mtp


def records():
    return [{"id": i, "sha1": f"seed-{i // 2}",
             "instruction": f"Write function {i} " + "hello " * (i % 3),
             "response": f"```python\ndef f{i}():\n    return {i}\n```"}
            for i in range(40)]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    monkeypatch.setattr(coding, "HfApi", lambda: SimpleNamespace(
        dataset_info=lambda *a, **kw: SimpleNamespace(sha="pinned")))
    monkeypatch.setattr(coding, "load_source_rows", lambda source: iter(records()))
    root = tmp_path / "coding"
    with patch("sys.argv", ["prepare_coding", "--output_dir", str(root),
                           "--tokenizer_path", str(tmp_path / "tokenizer"),
                           "--context_length", "64", "--val_fraction", "0.3"]):
        coding.main()
        with pytest.raises(ValueError, match="already exists"):
            coding.main()
    return root, tokenizer


def test_curate_filters_deduplicates_and_keeps_whole_solutions(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    rows = records()
    rows += [dict(rows[0]), {**rows[1], "instruction": "Different instruction"},
             {**rows[2], "instruction": " \n" + rows[2]["instruction"]},
             {**rows[3], "response": "Just prose"},
             {**rows[4], "response": "```python\ndef broken(:\n```"},
             {**rows[5], "instruction": "hello " * 1000,
              "response": "```python\nx = 1000\n```"},
             {"instruction": None}]
    report = coding.curate(rows, tokenizer, tmp_path / "out", 64, 0.3)
    assert report["counts"] == {"source_rows": 47, "accepted": 40, "duplicate": 3,
                                "no_python_block": 1, "invalid_python": 1,
                                "too_long": 1, "invalid_record": 1}
    split_groups = []
    for split in ("train", "val"):
        folder = tmp_path / "out" / split
        tasks = [json.loads(line) for line in (folder / "tasks.jsonl").read_text().splitlines()]
        split_groups.append({task["seed_sha1"] for task in tasks})
        tokens = np.load(folder / "tokens.npy")
        offsets = np.load(folder / "offsets.npy")
        for index, task in enumerate(tasks):
            expected = [tokenizer.bos_token_id, *tokenizer.encode(
                coding.format_task(task["instruction"], task["response"]),
                add_special_tokens=False), tokenizer.eos_token_id]
            assert tokens[offsets[index]:offsets[index + 1]].tolist() == expected
        assert report["splits"][split]["target_tokens"] == len(tokens) - len(tasks)
    assert not split_groups[0] & split_groups[1]


def test_total_token_boundary_keeps_614_and_discards_615(tmp_path):
    # The tokenizer emits 612 body tokens for an exactly-614-token example.
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2,
                                encode=lambda text, **kw: [3] * int(text.splitlines()[1]))
    rows = []
    for split in ("train", "val"):
        group = next(f"seed-{i}" for i in range(100)
                     if coding.task_split(f"seed-{i}", 0, 0.3) == split)
        for total in (613, 614, 615):
            rows.append({"id": f"{split}-{total}", "sha1": group,
                         "instruction": f"{total - 2}\n{split}",
                         "response": f"```python\nx = '{split}-{total}'\n```"})
    report = coding.curate(rows, tokenizer, tmp_path / "out", val_fraction=0.3)
    assert coding.parse_args([]).max_tokens == 614
    assert coding.parse_args([]).context_length == 614
    assert report["counts"] == {"source_rows": 6, "accepted": 4, "too_long": 2}
    for split in ("train", "val"):
        lengths = np.diff(np.load(tmp_path / "out" / split / "offsets.npy"))
        assert lengths.tolist() == [613, 614]


@pytest.mark.parametrize("workers", [0, 2])
def test_loader_masks_padding_shifts_targets_and_reopens_memmaps(corpus, workers):
    root, tokenizer = corpus
    loaders = prepare_coding_dataset(root, 8, 64, 16, tokenizer, num_workers=workers)
    assert sorted(loaders[0].sampler) == list(range(len(loaders[0].dataset)))
    for loader in loaders:
        total = 0
        for batch in loader:
            assert set(batch["image_paths"]) == {None}
            mask = batch["attention_mask"]
            assert torch.all(batch["labels"][~mask] == -100)
            assert torch.all(batch["input_ids"][:, 0] == tokenizer.bos_token_id)
            for row, length in enumerate(mask.sum(1)):
                assert batch["labels"][row, length - 1] == tokenizer.eos_token_id
                torch.testing.assert_close(batch["input_ids"][row, 1:length],
                                           batch["labels"][row, :length - 1])
            total += int(mask.sum())
        split = loader.dataset.directory.name
        assert total == json.loads((root / "manifest.json").read_text())["splits"][split]["target_tokens"]
        restored = pickle.loads(pickle.dumps(loader.dataset))
        assert restored.tokens is None
        torch.testing.assert_close(restored[0][0], loader.dataset[0][0])
    with pytest.raises(ValueError, match="context"):
        prepare_coding_dataset(root, 2, 32, 16, tokenizer)
    tokenizer.add_tokens(["new-token"])
    with pytest.raises(ValueError, match="tokenizer"):
        prepare_coding_dataset(root, 2, 64, 17, tokenizer)


def test_loader_pads_to_max_context_length(corpus):
    root, tokenizer = corpus
    loaders = prepare_coding_dataset(root, 2, 64, 16, tokenizer, pad_to_max_length=True)
    for loader in loaders:
        for batch in loader:
            assert batch["input_ids"].shape == batch["labels"].shape == (2, 64)
            assert torch.all(batch["input_ids"][~batch["attention_mask"]] == tokenizer.pad_token_id)
            assert torch.all(batch["labels"][~batch["attention_mask"]] == -100)
    validation = loaders[1]
    batch = next(iter(validation))
    for row in range(2):
        tokens = validation.dataset[row][0]
        length = len(tokens) - 1
        assert batch["attention_mask"][row].sum() == length
        torch.testing.assert_close(batch["input_ids"][row, :length], tokens[:-1])
        torch.testing.assert_close(batch["labels"][row, :length], tokens[1:])


def test_collate_rejects_samples_longer_than_pad_length():
    with pytest.raises(ValueError, match="exceeds pad length"):
        collate_annotations([(torch.arange(5), None)], 0, pad_to_length=3)


@pytest.mark.parametrize("trainer", [train, train_mtp])
def test_pad_to_max_length_flag(trainer):
    assert trainer.parse_args(["--coding_dir", "coding"]).pad_to_max_length is False
    assert trainer.parse_args(["--coding_dir", "coding", "--pad_to_max_length"]).pad_to_max_length
    with pytest.raises(SystemExit):
        trainer.parse_args(["--train_path", "x", "--val_path", "y", "--pad_to_max_length"])


@pytest.mark.parametrize("trainer", [train, train_mtp])
def test_coding_mode_excludes_other_sources(trainer):
    assert trainer.parse_args(["--coding_dir", "coding"]).data_root is None
    for flags in (["--data_root", "data"], ["--wikipedia_dir", "wiki"],
                  ["--train_path", "x", "--val_path", "y"],
                  ["--train_image_paths", "images"], ["--wikipedia_languages", "en"]):
        with pytest.raises(SystemExit):
            trainer.parse_args(["--coding_dir", "coding", *flags])


def test_corrupt_offsets_and_tokens_are_rejected(corpus):
    root, tokenizer = corpus
    token_path = root / "train/tokens.npy"
    tokens = np.load(token_path)
    tokens[0] = 100
    np.save(token_path, tokens)
    training, _ = prepare_coding_dataset(root, 2, 64, 16, tokenizer)
    with pytest.raises(ValueError, match="token IDs"):
        training.dataset[0]
    offset_path = root / "val/offsets.npy"
    offsets = np.load(offset_path)
    offsets[1] = offsets[0]
    np.save(offset_path, offsets)
    with pytest.raises(ValueError, match="offsets"):
        prepare_coding_dataset(root, 2, 64, 16, tokenizer)


@pytest.mark.parametrize("trainer", [train, train_mtp])
def test_both_trainers_update_weights_and_save_coding_checkpoint(corpus, tmp_path, trainer,
                                                                 capsys):
    root, _ = corpus
    embeddings = tmp_path / "embeddings.pt"
    torch.save(torch.randn(16, 6), embeddings)
    output = tmp_path / "checkpoints"
    flags = ["--coding_dir", str(root), "--tokenizer_path", str(tmp_path / "tokenizer"),
             "--embeddings_path", str(embeddings), "--output_dir", str(output),
             "--device", "cpu", "--max_context_length", "64", "--num_layer", "1",
             "--projection_dim", "4", "--head_dim", "2", "--q_head", "2",
             "--kv_head", "1", "--expansion_factor", "2", "--batch_size", "2",
             "--max_steps", "1", "--image_inference_every", "1",
             "--text_inference_every", "1", "--inference_max_new_tokens", "2"]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch("sys.argv", [trainer.__file__, *flags]), \
             patch("model.CLIPVisionModel.from_pretrained", return_value=FakeVisionModel()), \
             patch("model.CLIPImageProcessor.from_pretrained", return_value=FakeVisionProcessor()), \
             patch.object(trainer, "prepare_annotation_dataset") as annotations, \
             patch.object(trainer, "prepare_wikipedia_dataset") as wikipedia, \
             patch.object(train, "print_image_inference") as preview, \
             patch.object(train, "print_text_inference",
                          wraps=train.print_text_inference) as text_preview:
            trainer.main()
        annotations.assert_not_called()
        wikipedia.assert_not_called()
        preview.assert_not_called()
        assert [call.args[3] for call in text_preview.call_args_list] == [1]
        assert "[Text inference | iteration 1]" in capsys.readouterr().out
        saved = torch.load(output / "latest.pt", weights_only=True)
        assert torch.isfinite(torch.tensor(saved["loss"]))
        assert saved["optimizer_state_dict"]["state"]
    finally:
        torch.set_num_threads(previous)
