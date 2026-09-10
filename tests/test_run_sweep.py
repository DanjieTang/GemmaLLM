from collections import OrderedDict
import itertools
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from run_sweep import (
    build_command, checkpoint_directory, get_varying_keys, normalize_grid,
    parse_simple_yaml, read_train_args,
)
from train import parse_args


ROOT = Path(__file__).resolve().parents[1]


def test_folder_lists_stay_together_and_scalar_lists_sweep():
    config = parse_simple_yaml(ROOT / "sweep_config.yaml")
    multi_value = read_train_args(ROOT / "train.py", multi_value_only=True)
    keys, values = normalize_grid(config, multi_value)
    combinations = [dict(zip(keys, row)) for row in itertools.product(*values)]
    assert len(combinations) == 2
    assert get_varying_keys(config, multi_value) == ["lr"]
    directories = set()
    for params in combinations:
        directory = checkpoint_directory(params, "test")
        assert directory == checkpoint_directory(params, "test")
        directories.add(directory)
        cmd = build_command(sys.executable, ROOT / "train.py", params, "test")
        args = parse_args(cmd[2:])
        assert args.train_folders == ["train", "OpenImageV7_train"]
        assert args.val_folders == ["val", "OpenImageV7_val"]
        assert args.data_root == "data"
        assert args.project is None
    assert len(directories) == 2


def test_sweep_dry_run_preserves_state_and_emits_usable_commands(tmp_path):
    state = tmp_path / "state.json"
    state.write_text('{"completed": {}, "failed": {}, "in_progress": null}')
    before = state.read_bytes()
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_sweep.py"),
         "--config", str(ROOT / "sweep_config.yaml"),
         "--state-file", str(state), "--dry-run"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    commands = [shlex.split(line.removeprefix("Command: "))
                for line in result.stdout.splitlines() if line.startswith("Command: ")]
    assert len(commands) == 2
    parsed = [parse_args(command[2:]) for command in commands]
    assert parsed[0].output_dir != parsed[1].output_dir
    assert all(args.output_dir.parent == Path("checkpoints/annotation_sweep")
               for args in parsed)
    assert state.read_bytes() == before
    assert not (tmp_path / "checkpoints").exists()


def test_command_preserves_paths_with_spaces():
    params = {"data_root": "data with spaces", "train_folders": ["a b", "c"]}
    cmd = build_command(sys.executable, ROOT / "train.py", params, "test")
    args = parse_args(shlex.split(shlex.join(cmd))[2:])
    assert args.data_root == "data with spaces"
    assert args.train_folders == ["a b", "c"]


def test_empty_folder_list_rejected():
    with pytest.raises(ValueError, match="empty list"):
        normalize_grid(OrderedDict(train_folders=[]), {"train_folders"})


def test_boolean_model_options_sweep_independently(tmp_path):
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "use_moe:\n  - true\n  - false\nfine_tuning:\n  - true\n  - false\n"
        "num_experts: 2\nlora_rank: 4\nlora_alpha: 8\ndropout_ratio: 0.2\n"
        "theta: 20000\nload_balancing_loss_weight: 0.03\n"
    )
    config = parse_simple_yaml(config_path)
    assert set(config) <= read_train_args(ROOT / "train.py")
    multi_value = read_train_args(ROOT / "train.py", multi_value_only=True)
    keys, values = normalize_grid(config, multi_value)
    assert get_varying_keys(config, multi_value) == ["use_moe", "fine_tuning"]
    modes = set()
    for row in itertools.product(*values):
        cmd = build_command(sys.executable, ROOT / "train.py", dict(zip(keys, row)), "test")
        args = parse_args(cmd[2:])
        modes.add((args.use_moe, args.fine_tuning))
        assert args.num_experts == 2
        assert args.lora_rank == 4
        assert args.lora_alpha == 8
        assert args.dropout_ratio == 0.2
        assert args.theta == 20000
        assert args.load_balancing_loss_weight == 0.03
    assert modes == {(False, False), (False, True), (True, False), (True, True)}
