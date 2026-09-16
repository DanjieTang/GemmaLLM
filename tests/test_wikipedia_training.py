import hashlib
import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data_preprocessing.tokenize_wikipedia import ShardWriter, save_json
from lazy_dataloader import prepare_mixed_dataset, prepare_wikipedia_dataset
from test_annotation_training import make_pair, tiny_model
from test_tokenize_wikipedia import make_tokenizer
from test_vlm import FakeVisionModel, FakeVisionProcessor
import train
import train_mtp


@pytest.fixture
def wikipedia(tmp_path):
    tokenizer = make_tokenizer(tmp_path / "tokenizer")
    root = tmp_path / "wikipedia"
    for language in ("en", "zh"):
        config = {
            "format_version": 1, "language": language, "context_length": 4,
            "tokenizer_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "special_token_ids": {name: getattr(tokenizer, name) for name in
                                  ("bos_token_id", "eos_token_id", "pad_token_id")},
        }
        splits = {}
        for split in ("train", "val"):
            with ShardWriter(root / language / split, 4, 2, 0) as writer:
                writer.add_article({"id": "long", "title": "Title", "url": ""},
                                   np.array([1, 4, 5, 6, 7, 8, 2], dtype=np.int32))
                writer.add_article({"id": "empty", "title": "", "url": ""},
                                   np.array([1, 2], dtype=np.int32))
                splits[split] = writer.finish()
        save_json(root / language / "manifest.json", {"config": config, "splits": splits})
    return root, tokenizer


def loaders(wikipedia, **kwargs):
    root, tokenizer = wikipedia
    options = dict(languages=None, batch_size=3, max_context_length=4,
                   vocabulary_size=16, tokenizer=tokenizer)
    options.update(kwargs)
    return prepare_wikipedia_dataset(root, **options)


@pytest.mark.parametrize("languages", [None, ["all"], ["en", "zh"]])
def test_shards_keep_continuations_and_mask_only_padding(wikipedia, languages):
    training, validation = loaders(wikipedia, languages=languages)
    assert len(training.dataset) == len(validation.dataset) == 6
    batch = next(iter(validation))
    assert batch["input_ids"].tolist() == [[1, 4, 5, 6], [7, 8, 0, 0], [1, 0, 0, 0]]
    assert batch["labels"].tolist() == [[4, 5, 6, 7], [8, 2, -100, -100],
                                       [2, -100, -100, -100]]
    assert batch["attention_mask"].tolist() == [[True]*4, [True, True, False, False],
                                               [True, False, False, False]]
    assert batch["image_paths"] == [None]*3
    assert sum(int((b["labels"] != -100).sum()) for b in validation) == 14
    # Every window is seen once, even when shard order changes each epoch.
    for _ in range(2):
        assert sorted(training.sampler) == list(range(6))
    assert all(isinstance(array, np.memmap)
               for pair in validation.dataset._mappings.values() for array in pair)
    restored = pickle.loads(pickle.dumps(validation.dataset))
    assert not restored._mappings
    assert restored[1][0].tolist() == [7, 8, 2]


@pytest.mark.parametrize("workers", [0, 2])
def test_language_selection_limits_and_worker_loading(wikipedia, workers):
    training, validation = loaders(wikipedia, languages=["zh"], max_samples=2,
                                   num_workers=workers)
    assert len(training.dataset) == 2
    assert sorted(training.sampler) == [0, 1]
    assert sum(len(batch["input_ids"]) for batch in training) == 2
    assert next(iter(validation))["labels"].tolist() == [[4, 5, 6, 7], [8, 2, -100, -100]]


@pytest.mark.parametrize("change,match", [
    ({"context_length": 8}, "context"),
    ({"tokenizer_sha256": "other"}, "tokenizer"),
    ({"special_token_ids": {}}, "tokenizer"),
    ({"format_version": 2}, "format"),
])
def test_incompatible_preprocessing_is_rejected(wikipedia, change, match):
    path = wikipedia[0] / "en/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"].update(change)
    save_json(path, manifest)
    with pytest.raises(ValueError, match=match):
        loaders(wikipedia)


def test_incomplete_languages_and_corrupt_arrays_are_rejected(wikipedia):
    root, _ = wikipedia
    (root / ".de.incomplete").mkdir()
    # Discovery ignores unfinished editions; requesting one must fail.
    assert len(loaders(wikipedia)[0].dataset) == 6
    with pytest.raises(ValueError, match="completed"):
        loaders(wikipedia, languages=["de"])
    path = root / "en/train/lengths-00000.npy"
    np.save(path, np.array([1, 3], dtype=np.int32))
    training, _ = loaders(wikipedia)
    with pytest.raises(ValueError, match="invalid length"):
        training.dataset[0]
    np.save(path, np.array([5], dtype=np.int32))
    with pytest.raises(ValueError, match="shapes"):
        loaders(wikipedia)


@pytest.mark.parametrize("trainer", [train, train_mtp])
@pytest.mark.parametrize("flags", [
    ["--wikipedia_languages", "en"],
    ["--wikipedia_dir", "wiki", "--train_path", "t.npy", "--val_path", "v.npy"],
    ["--wikipedia_dir", "wiki", "--train_image_paths", "images.npy"],
    ["--wikipedia_dir", "wiki", "--wikipedia_languages", "en", "en"],
    ["--wikipedia_dir", "wiki", "--wikipedia_languages", "all", "en"],
])
def test_conflicting_data_options_fail_early(trainer, flags):
    with pytest.raises(SystemExit):
        trainer.parse_args(flags)


@pytest.mark.parametrize("trainer", [train, train_mtp])
def test_all_languages_is_the_default_for_both_trainers(trainer):
    assert trainer.parse_args([]).wikipedia_languages == ["all"]
    assert trainer.parse_args(["--wikipedia_dir", "wiki"]).wikipedia_languages == ["all"]
    assert trainer.parse_args([
        "--wikipedia_dir", "wiki", "--wikipedia_languages", "all",
    ]).wikipedia_languages == ["all"]


@pytest.mark.parametrize("trainer", [train, train_mtp])
@pytest.mark.parametrize("mixed", [False, True])
def test_both_entrypoints_train_validate_and_save_wikipedia(wikipedia, tmp_path, trainer, mixed):
    root, tokenizer = wikipedia
    embeddings = tmp_path / "embeddings.pt"
    torch.save(torch.randn(16, 6), embeddings)
    output = tmp_path / "checkpoint"
    flags = [
        "--wikipedia_dir", str(root), "--wikipedia_languages", "en",
        "--tokenizer_path", str(tmp_path / "tokenizer"),
        "--embeddings_path", str(embeddings), "--output_dir", str(output),
        "--device", "cpu", "--max_context_length", "4", "--num_layer", "1",
        "--projection_dim", "4", "--head_dim", "2", "--q_head", "2",
        "--kv_head", "1", "--expansion_factor", "2", "--batch_size", "3",
        "--max_steps", "1", "--inference_every", "1",
    ]
    if trainer is train_mtp:
        flags += ["--mtp_depth", "2"]
    if mixed:
        for split in ("train", "val"):
            make_pair(tmp_path / "images" / split, "one", [1, 9, 10, 2])
        flags += ["--data_root", str(tmp_path / "images"), "--train_folders", "train",
                  "--val_folders", "val", "--batch_size", "4"]
    args = trainer.parse_args(flags)
    assert (args.data_root is not None) == mixed
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch("sys.argv", [trainer.__file__, *flags]), \
             patch("model.CLIPVisionModel.from_pretrained", return_value=FakeVisionModel()), \
             patch("model.CLIPImageProcessor.from_pretrained", return_value=FakeVisionProcessor()), \
             patch.object(trainer, "print_image_inference") as preview, \
             patch.object(trainer, "prepare_annotation_dataset") as annotations, \
             patch.object(trainer, "prepare_dataset") as legacy:
            trainer.main()
        if mixed:
            preview.assert_called_once()
            assert preview.call_args.args[2] == str(tmp_path / "images/val/images/one.png")
        else:
            preview.assert_not_called()
        annotations.assert_not_called()
        legacy.assert_not_called()
        saved = torch.load(output / "latest.pt", weights_only=True)
        assert torch.isfinite(torch.tensor(saved["loss"]))
        assert saved["optimizer_state_dict"]["state"]
        if trainer is train_mtp:
            assert saved["metrics"]["mtp_1_tokens"] == (6 if mixed else 4)
            assert saved["metrics"]["mtp_2_tokens"] == (3 if mixed else 2)
    finally:
        torch.set_num_threads(previous)


def mixed_loaders(wikipedia, tmp_path, **kwargs):
    root, tokenizer = wikipedia
    for split in ("train", "val"):
        for name in ("one", "two"):
            make_pair(tmp_path / "images" / split, name, [1, 9, 10, 2])
    return prepare_mixed_dataset(
        str(tmp_path / "images"), ["train"], ["val"], root, ["all"],
        8, 4, 16, tokenizer, **kwargs,
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_mixed_shuffle_visits_every_example_and_preserves_splits(wikipedia, tmp_path, workers):
    training, validation = mixed_loaders(wikipedia, tmp_path, num_workers=workers)
    torch.manual_seed(7)
    first, second = list(training.sampler), list(training.sampler)
    assert sorted(first) == sorted(second) == list(range(8))
    assert first != second
    batch = next(iter(training))
    assert len(batch["image_paths"]) == 8
    assert sum(path is not None for path in batch["image_paths"]) == 2
    assert all("/train/" in path for path in batch["image_paths"] if path)
    for row, path in enumerate(batch["image_paths"]):
        if path:
            assert batch["labels"][row].tolist() == [9, 10, 2, -100]
    assert int((batch["labels"] != -100).sum()) == 20
    assert list(validation.sampler) == list(range(8))
    # Previews index images directly, without scanning Wikipedia token windows.
    with patch.object(type(validation.dataset.datasets[1]), "__getitem__",
                      side_effect=AssertionError("Should not read Wikipedia")):
        paths = list(train.iter_validation_images(validation.dataset))
    assert len(paths) == 2 and all("/val/" in path for path in paths)


def test_mixed_sample_limit_applies_to_each_source_and_split(wikipedia, tmp_path):
    training, validation = mixed_loaders(wikipedia, tmp_path, max_samples=1)
    for loader in (training, validation):
        assert [len(child) for child in loader.dataset.datasets] == [1, 1]
        assert len(loader.dataset) == 2


@pytest.mark.parametrize("use_moe", [False, True])
def test_mixed_forward_matches_individual_rows_and_trains_both_paths(wikipedia, tmp_path, use_moe):
    _, validation = mixed_loaders(wikipedia, tmp_path)
    batch = next(iter(validation))
    model, _ = tiny_model(tmp_path / "model", use_moe)
    model.eval()
    logits, auxiliary_loss = model(batch["input_ids"], image_paths=batch["image_paths"],
                                   attention_mask=batch["attention_mask"])
    for row, path in enumerate(batch["image_paths"]):
        length = int(batch["attention_mask"][row].sum())
        single, _ = model(batch["input_ids"][row:row + 1, :length], image_paths=[path])
        torch.testing.assert_close(logits[row, :length], single[0], rtol=1e-5, atol=1e-6)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
    (loss + auxiliary_loss).backward()
    assert torch.isfinite(loss)
    for parameter in (model.seperation_token, model.text_token_projection.weight,
                      model.llm.classifier.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
