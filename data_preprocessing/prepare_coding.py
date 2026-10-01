"""Stream and curate complete coding tasks from pinned public datasets."""

import argparse
import ast
from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from itertools import islice
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
import warnings

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if __package__:
    from .coding_sources import CodingSource, PRESETS, SOURCES, expand_sources, source_tasks
else:
    from coding_sources import CodingSource, PRESETS, SOURCES, expand_sources, source_tasks


DATASET = "bigcode/self-oss-instruct-sc2-exec-filter-50k"
DEFAULT_REVISION = "356bb069eee815daa6e23e9a282eeefe1490ad44"
DEFAULT_MAX_TOKENS = 614
PYTHON_BLOCK = re.compile(r"^```(?:python|py)\s*\n(.*?)^```", re.MULTILINE | re.DOTALL)
LANGUAGE_ALIASES = {
    "py": "python", "python3": "python", "js": "javascript", "jsx": "javascript",
    "ts": "typescript", "tsx": "typescript", "c++": "cpp", "cxx": "cpp",
    "cc": "cpp", "c#": "csharp", "cs": "csharp", "golang": "go",
    "rs": "rust", "rb": "ruby", "sh": "bash", "shell": "bash",
    "shellscript": "bash", "zsh": "bash", "kt": "kotlin", "kts": "kotlin",
    "objective-c": "objectivec", "objc": "objectivec", "rscript": "r",
    "python2": "python", "py3": "python", "node": "javascript", "nodejs": "javascript",
    "node.js": "javascript", "mjs": "javascript", "cjs": "javascript", "es6": "javascript",
    "h": "c", "hpp": "cpp", "arduino": "cpp", "ino": "cpp", "cu": "cuda",
    "obj-c": "objectivec", "objective-c++": "objectivec", "ksh": "bash",
    "ps1": "powershell", "pwsh": "powershell", "bat": "batch", "cmd": "batch",
    "dos": "batch", "batchfile": "batch", "mysql": "sql", "postgresql": "sql",
    "postgres": "sql", "psql": "sql", "plsql": "sql", "pl/sql": "sql", "plpgsql": "sql",
    "pl/pgsql": "sql", "tsql": "sql", "t-sql": "sql", "sqlite": "sql", "sqlite3": "sql",
    "mssql": "sql", "hiveql": "sql", "sparksql": "sql", "f#": "fsharp", "fs": "fsharp",
    "vb": "vbnet", "vb.net": "vbnet", "visualbasic": "vbnet", "vbs": "vbscript",
    "perl6": "raku", "delphi": "pascal", "objectpascal": "pascal", "freepascal": "pascal",
    "octave": "matlab", "mathematica": "wolfram", "wl": "wolfram", "ahk": "autohotkey",
    "elisp": "lisp", "emacs-lisp": "lisp", "common-lisp": "lisp", "commonlisp": "lisp",
    "clj": "clojure", "cljs": "clojure", "clojurescript": "clojure", "ex": "elixir",
    "exs": "elixir", "erl": "erlang", "hs": "haskell", "ml": "ocaml", "f90": "fortran",
    "f95": "fortran", "f77": "fortran", "asm": "assembly", "nasm": "assembly",
    "masm": "assembly", "x86asm": "assembly", "armasm": "assembly", "scm": "scheme",
    "rkt": "racket", "jl": "julia", "docker": "dockerfile", "make": "makefile",
    "mk": "makefile", "terraform": "hcl", "tf": "hcl", "gql": "graphql", "proto": "protobuf",
    "proto3": "protobuf", "sass": "scss", "systemverilog": "verilog", "sol": "solidity",
    "cr": "crystal", "gd": "gdscript", "as3": "actionscript", "cbl": "cobol",
    "hx": "haxe", "pine": "pinescript", "coffee": "coffeescript", "gawk": "awk",
    "dlang": "d", "luau": "lua", "gradle": "groovy", "sbt": "scala", "jsonc": "json",
    "json5": "json", "yml": "yaml", "xaml": "xml", "svg": "xml", "xsd": "xml",
    "xslt": "xml", "plist": "xml", "htm": "html", "xhtml": "html",
}
LANGUAGES = (
    "python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go",
    "rust", "ruby", "php", "swift", "kotlin", "bash", "sql", "html", "css",
    "scala", "r", "julia", "lua", "perl", "dart", "haskell", "elixir",
    "clojure", "objectivec", "matlab", "powershell", "ocaml", "erlang",
    "fortran", "pascal", "groovy", "solidity", "assembly", "scheme", "lisp",
    "fsharp", "racket", "d", "tcl", "awk", "coffeescript", "raku", "sml", "prolog",
    "ada", "cobol", "nim", "zig", "crystal", "elm", "vbnet", "vba", "vbscript",
    "batch", "autohotkey", "applescript", "wolfram", "actionscript", "apex", "abap",
    "smalltalk", "forth", "haxe", "vala", "mojo", "gdscript", "glsl", "hlsl", "cuda",
    "verilog", "vhdl", "sas", "stata", "pinescript", "cypher", "graphql", "protobuf",
    "dockerfile", "makefile", "cmake", "hcl", "nix", "scss", "less", "vue", "svelte",
    "yaml", "xml", "toml", "ini",
)
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)
INDENTED_FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})([^\n]*)$", re.MULTILINE)
# Fence labels for sample output, prose, or notation rather than a solution.
# JSON is included because JSON-only answers mostly wrap prose steps or escaped
# code strings rather than a program.
AUXILIARY_LABELS = frozenset({
    "", "text", "txt", "plaintext", "plain", "output", "out", "console", "terminal",
    "shell-session", "sh-session", "log", "stdout", "stderr", "result", "csv", "tsv",
    "mermaid", "pseudo", "pseudocode", "markdown", "md", "diff", "patch", "regex",
    "math", "latex", "tex", "http", "pycon", "none", "nohighlight", "input", "json",
})
# Configuration files are solutions only when no program accompanies them.
SECONDARY_LANGUAGES = frozenset({"yaml", "xml", "toml", "ini"})
SETUP_SHELLS = frozenset({"bash", "powershell", "batch"})
SETUP_COMMAND = re.compile(
    r"(?:\$\s*)?(?:sudo\s+)?(?:(?:pip3?|python3?|py|conda|mamba|poetry|pipenv|uv|npm|npx|"
    r"pnpm|yarn|bun|deno|node|gem|bundle|ruby|cargo|rustup|rustc|go|composer|php|dotnet|"
    r"nuget|mvn|gradle|java|javac|gcc|g\+\+|clang|clang\+\+|make|cmake|swift|kotlinc|"
    r"rscript|perl|brew|apt|apt-get|yum|dnf|apk|choco|winget|cd|mkdir|touch|ls|export|set|"
    r"source|chmod|git|docker|docker-compose|kubectl|curl|wget|ssh|scp|systemctl|service|"
    r"tsc|ts-node|pytest|jupyter|streamlit|flask|gunicorn|uvicorn|rails|virtualenv|"
    r"install-module|install-package)\s|\./\S)",
    re.IGNORECASE)
QUESTION_LABEL = re.compile(
    r"\A\s*\*\*\s*(?:Question|Problem|问题|题目)\s*[:：]?\s*\*\*\s*[:：]?\s*", re.IGNORECASE)
ANSWER_SECTION = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?\*\*[ \t]*(?:(?:Example|Sample|Expected|Reference|Suggested)"
    r"[ \t]+)?(?:Answer|Solution|答案|解答|参考答案)[ \t]*"
    r"(?:[:：][ \t]*\*\*|\*\*[ \t]*(?:[:：]|$))",
    re.IGNORECASE | re.MULTILINE)
EXECUTION_PREDICTION = re.compile(r"\[/?(?:ANSWER|THOUGHT)\]")
RESPONSE_FORMATS = ("original", "code")


def canonical_language(value: str) -> str:
    """Map a fence info string, such as ``Python3 title="a.py"``, to a language tag."""
    words = value.strip().lower().split()
    value = re.split(r"[{:,]", words[0], maxsplit=1)[0] if words else ""
    return LANGUAGE_ALIASES.get(value, value)


def code_blocks(response: str) -> list[tuple[str, str]] | None:
    """Read complete Markdown fences, retaining code indentation verbatim."""
    blocks = []
    opened = None
    for match in FENCE.finditer(response):
        fence, label = match.groups()
        if opened is None:
            opened = (fence, canonical_language(label), match.end() + 1)
        elif (fence[0] == opened[0][0] and len(fence) >= len(opened[0])
              and not label.strip()):
            blocks.append((opened[1], response[opened[2]:match.start()]))
            opened = None
    return blocks if opened is None else None


def indented_code_blocks(response: str) -> list[tuple[str, str]] | None:
    """Read complete fences at any indentation, such as inside Markdown lists.

    A block closes only at its opening indentation, which is removed from its
    lines; fence-like lines indented differently remain part of the code.
    """
    blocks = []
    opened = None
    for match in INDENTED_FENCE.finditer(response):
        indent, fence, label = match.groups()
        if opened is None:
            opened = (indent, fence, canonical_language(label), match.end() + 1)
        elif (indent == opened[0] and fence[0] == opened[1][0]
              and len(fence) >= len(opened[1]) and not label.strip()):
            lines = response[opened[3]:match.start()].split("\n")
            blocks.append((opened[2], "\n".join(
                line[len(indent):] if line.startswith(indent) else line.lstrip(" \t")
                for line in lines)))
            opened = None
    return blocks if opened is None else None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_split(group: str, seed: int, val_fraction: float) -> str:
    bucket = int(digest(f"{seed}:{group}")[:16], 16)
    return "val" if bucket < int(val_fraction * (1 << 64)) else "train"


def format_task(instruction: str, response: str) -> str:
    """Use the same prefix at inference, ending immediately after Response."""
    return f"### Instruction\n{instruction}\n\n### Response\n{response}"


def trim_blank_lines(code: str) -> str:
    """Remove blank leading and trailing lines, preserving indentation."""
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def format_code_block(language: str, code: str) -> str:
    """Fence one solution, lengthening the fence when the code contains one."""
    nested = [len(run) + 1 for run in re.findall(r"^ {0,3}(`{3,})", code, re.MULTILINE)]
    fence = "`" * max([3, *nested])
    return f"{fence}{language}\n{code}\n{fence}"


def is_setup_block(language: str, code: str) -> bool:
    """Recognize short shell blocks that only install, build, run, or change directory."""
    if language not in SETUP_SHELLS:
        return False
    lines = [line.strip() for line in code.splitlines()
             if line.strip() and not line.strip().startswith(("#", "::", "REM "))]
    return 0 < len(lines) <= 8 and all(SETUP_COMMAND.match(line) for line in lines)


def select_solution(blocks: list[tuple[str, str]]) -> tuple[tuple[str, str] | None, str | None]:
    """Choose the one solution block, ignoring outputs and accompanying setup files.

    Output, prose, and unlabeled blocks are never solutions. Data formats and
    shell setup commands count only when no other code block is present, and
    never beside an unlabeled block that may hold the program. When several
    same-language blocks remain, one containing all the others is the complete
    program; otherwise the task is rejected rather than concatenated.
    """
    primary, secondary, unlabeled = [], [], False
    for language, code in blocks:
        if not code.strip():
            continue
        if language in AUXILIARY_LABELS:
            unlabeled = unlabeled or not language
        elif language in SECONDARY_LANGUAGES or is_setup_block(language, code):
            secondary.append((language, code))
        else:
            primary.append((language, code))
    candidates = primary or secondary
    if not candidates:
        return None, "no_code_block"
    if not primary and unlabeled:
        return None, "multiple_code_blocks"
    if len(candidates) > 1:
        complete = [(language, code) for language, code in candidates
                    if all(contains_code(code, other) for _, other in candidates)]
        if not complete or len({language for language, _ in candidates}) > 1:
            return None, "multiple_code_blocks"
        return complete[0], None
    return candidates[0], None


def contains_code(outer: str, inner: str) -> bool:
    """Whether inner occurs in outer, ignoring whitespace, at identifier boundaries."""
    outer, inner = " ".join(outer.split()), " ".join(inner.split())
    pattern = re.escape(inner)
    if re.match(r"\w", inner):
        pattern = r"(?<!\w)" + pattern
    if re.search(r"\w\Z", inner):
        pattern += r"(?!\w)"
    return re.search(pattern, outer) is not None


def clean_instruction(instruction: str) -> tuple[str, list[str]]:
    """Remove a leaked answer section and a leading Question label from a prompt."""
    edits = []
    leaked = ANSWER_SECTION.search(instruction)
    if leaked:
        instruction = instruction[:leaked.start()]
        edits.append("answer_section_removed")
    labeled = QUESTION_LABEL.match(instruction)
    if labeled:
        instruction = instruction[labeled.end():]
        edits.append("question_label_removed")
    return instruction.strip(), edits


def contains_solution(instruction: str, code: str, minimum_characters: int = 40) -> bool:
    """Detect prompts that already contain the whole solution, ignoring spacing."""
    code = " ".join(code.split())
    return len(code) >= minimum_characters and code in " ".join(instruction.split())


def parses_as_python(code: str) -> bool:
    try:
        return bool(ast.parse(code).body)
    except (SyntaxError, ValueError, RecursionError):
        return False


def original_task(instruction: str, response: str, declared: str | None,
                  permitted: Collection[str]) -> tuple[str, str, list, list] | str:
    """Keep the source response, requiring complete labeled code in permitted languages."""
    if declared and declared in LANGUAGES and declared not in permitted:
        return "excluded_language"
    blocks = code_blocks(response)
    if blocks is None:
        return "incomplete_code_fence"
    # Accept raw Python only with a publisher language label and when the
    # entire response parses, so prose never gets wrapped as source code.
    if not blocks and declared == "python" and "python" in permitted:
        try:
            if ast.parse(response).body:
                blocks = [("python", response)]
                response = "```python\n" + response + "\n```"
        except (SyntaxError, ValueError, RecursionError):
            pass
    relevant = [(language, code) for language, code in blocks if language in permitted]
    if not relevant or not any(code.strip() for _, code in relevant):
        return "no_python_block" if set(permitted) == {"python"} else "no_code_block"
    return instruction, response, relevant, []


def code_task(instruction: str, response: str, declared: str | None,
              permitted: Collection[str]) -> tuple[str, str, list, list] | str:
    """Reduce a task to a cleaned prompt and exactly one fenced solution block."""
    if EXECUTION_PREDICTION.search(instruction):
        return "execution_prediction"
    instruction, edits = clean_instruction(instruction)
    if not instruction:
        return "empty_prompt"
    blocks = indented_code_blocks(response)
    if blocks is None:
        return "incomplete_code_fence"
    if (not blocks and declared == "python" and "python" in permitted
            and parses_as_python(response)):
        blocks = [("python", response)]
    solution, reason = select_solution(blocks)
    if reason is not None:
        return reason
    language, code = solution
    if language not in LANGUAGES:
        return "unsupported_language"
    if language not in permitted:
        return "excluded_language"
    code = trim_blank_lines(textwrap.dedent(code))
    if contains_solution(instruction, code):
        return "solution_in_prompt"
    if len(blocks) > 1:
        edits.append("extra_blocks_removed")
    return instruction, format_code_block(language, code), [(language, code)], edits


def load_source_rows(source: CodingSource) -> Iterable[dict]:
    """Read pinned sources in bounded batches, caching only reached Parquet shards.

    Local, synchronous Parquet reads avoid PyArrow's remote fragment-scanner
    shutdown crash when a preview stops before exhausting its input. Sources
    with ``cache_shards=False`` download each shard to a temporary directory
    and delete it after reading, so only one shard occupies disk at a time.
    """
    if source.name == "codefeedback":
        yield from load_dataset(source.dataset, split=source.split,
                                revision=source.revision, streaming=True)
        return
    paths = sorted(
        path for path in HfApi().list_repo_files(
            source.dataset, repo_type="dataset", revision=source.revision)
        if path.endswith(".parquet") and (
            source.config is None or path.startswith(f"{source.config}/")
            or path.startswith(f"data/{source.config}-"))
        and (source.shard_prefix is None or path.startswith(source.shard_prefix))
    )
    if not paths:
        raise ValueError(f"No Parquet shards found for {source.name} at {source.revision}.")
    with ExitStack() as stack:
        download_options = {}
        if not source.cache_shards:
            download_options["local_dir"] = stack.enter_context(
                tempfile.TemporaryDirectory(prefix=f"{source.name}-"))
        for path in paths:
            local_path = hf_hub_download(source.dataset, path, repo_type="dataset",
                                         revision=source.revision, **download_options)
            try:
                with pq.ParquetFile(local_path) as parquet:
                    columns = None if source.columns is None else list(source.columns)
                    for batch in parquet.iter_batches(batch_size=1024, use_threads=False,
                                                      columns=columns):
                        yield from batch.to_pylist()
            finally:
                if not source.cache_shards:
                    Path(local_path).unlink(missing_ok=True)


def curate(records: Iterable[dict], tokenizer: PreTrainedTokenizerBase,
           destination: Path, context_length: int = DEFAULT_MAX_TOKENS,
           val_fraction: float = 0.02, seed: int = 0, *,
           max_tokens: int = DEFAULT_MAX_TOKENS, languages: tuple[str, ...] = ("python",),
           batch_size: int = 1, deduplicate_code: bool = False,
           response_format: str = "original",
           excluded_languages: Mapping[str, Collection[str]] | None = None) -> dict:
    """Write flat int32 tokens and offsets delimiting complete BOS/task/EOS tasks.

    Only hashes, offsets and a bounded tokenizer batch stay in memory. Never
    execute source code. Preserve original StarCoder2 seed split assignments.
    ``response_format="code"`` reduces each response to one fenced solution;
    ``excluded_languages`` maps source names to languages dropped for that source.
    """
    if (context_length < 1 or max_tokens < 2 or batch_size < 1
            or not 0 < val_fraction < 1 or response_format not in RESPONSE_FORMATS):
        raise ValueError("Invalid curation limits, validation fraction, or response format.")
    allowed = frozenset(languages)
    excluded_languages = excluded_languages or {}
    permitted_by_source = {}
    prepare_task = code_task if response_format == "code" else original_task
    # Source text such as a literal "<eos>" can tokenize to a structural token.
    structural_ids = ({tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
                      - {None} if response_format == "code" else set())
    counts, source_counts, language_counts, edit_counts = Counter(), {}, {}, {}
    seen_prompts, seen_responses, seen_code = set(), set(), set()
    offsets = {split: [0] for split in ("train", "val")}
    pending = []

    def count(source: str, reason: str) -> None:
        counts[reason] += 1
        source_counts.setdefault(source, Counter())[reason] += 1

    with ExitStack() as stack:
        # Parsing third-party code reports its invalid escape sequences; they
        # do not affect validity and would otherwise flood the build log.
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore", SyntaxWarning)
        streams, metadata = {}, {}
        for split in offsets:
            folder = destination / split
            folder.mkdir(parents=True)
            streams[split] = stack.enter_context((folder / "tokens.bin").open("wb"))
            metadata[split] = stack.enter_context(
                (folder / "tasks.jsonl").open("w", encoding="utf-8"))

        def flush() -> None:
            if not pending:
                return
            texts = [format_task(item["instruction"], item["response"]) for item in pending]
            if batch_size == 1:
                encoded = [tokenizer.encode(texts[0], add_special_tokens=False)]
            else:
                encoded = tokenizer(texts, add_special_tokens=False, padding=False,
                                    truncation=False, return_attention_mask=False)["input_ids"]
            if len(encoded) != len(pending):
                raise ValueError("Tokenizer returned an unexpected batch size.")
            for item, body in zip(pending, encoded):
                prompt_hash, response_hash = item["instruction_sha256"], item["response_sha256"]
                code_hash = item["code_sha256"]
                source = item["source"]
                if (prompt_hash in seen_prompts or response_hash in seen_responses
                        or (deduplicate_code and code_hash in seen_code)):
                    count(source, "duplicate")
                    continue
                ids = [tokenizer.bos_token_id, *body, tokenizer.eos_token_id]
                if len(ids) > min(max_tokens, context_length + 1):
                    count(source, "too_long")
                    continue
                if structural_ids.intersection(body):
                    count(source, "special_token_text")
                    continue
                if min(ids) < 0 or max(ids) > np.iinfo(np.int32).max:
                    raise ValueError("Token IDs cannot fit in int32.")
                seen_prompts.add(prompt_hash)
                seen_responses.add(response_hash)
                if deduplicate_code:
                    seen_code.add(code_hash)
                split = task_split(item["seed_sha1"], seed, val_fraction)
                np.asarray(ids, dtype=np.int32).tofile(streams[split])
                offsets[split].append(offsets[split][-1] + len(ids))
                metadata[split].write(json.dumps({**item, "tokens": len(ids)},
                                                ensure_ascii=False) + "\n")
                count(source, "accepted")
                edit_counts.setdefault(source, Counter()).update(item.get("edits", ()))
                for language in item["languages"]:
                    language_counts.setdefault(language, Counter()).update(
                        tasks=1, tokens=len(ids))
            pending.clear()

        for record in tqdm(records, desc="Curating coding tasks", unit="task", mininterval=5):
            source = record.get("source", "starcoder2")
            count(source, "source_rows")
            instruction, response = record.get("instruction"), record.get("response")
            if (not isinstance(instruction, str) or not instruction.strip()
                    or not isinstance(response, str) or not response.strip()
                    or not isinstance(record.get("sha1"), str) or not record["sha1"]):
                count(source, "invalid_record")
                continue
            instruction = instruction.strip().replace("\r\n", "\n")
            response = response.strip().replace("\r\n", "\n")
            declared = record.get("language")
            declared = canonical_language(declared) if isinstance(declared, str) else None
            permitted = permitted_by_source.get(source)
            if permitted is None:
                permitted = permitted_by_source[source] = allowed.difference(
                    excluded_languages.get(source, ()))
            task = prepare_task(instruction, response, declared, permitted)
            if isinstance(task, str):
                count(source, task)
                continue
            instruction, response, relevant, edits = task
            canonical_code = []
            try:
                for language, code in relevant:
                    if not code.strip():
                        raise ValueError("Empty code block")
                    if language == "python":
                        tree = ast.parse(code)
                        if not tree.body:
                            raise ValueError("Empty Python code block")
                        canonical_code.append((language, ast.dump(tree, include_attributes=False)))
                    else:
                        canonical_code.append((language, "\n".join(
                            line.rstrip() for line in code.strip().splitlines())))
            except (SyntaxError, ValueError, RecursionError):
                count(source, "invalid_python")
                continue
            prompt_hash = digest(" ".join(instruction.split()))
            response_hash = digest(response)
            code_hash = digest(json.dumps(canonical_code, ensure_ascii=False))
            if (prompt_hash in seen_prompts or response_hash in seen_responses
                    or (deduplicate_code and code_hash in seen_code)):
                count(source, "duplicate")
                continue
            provenance = {key: record[key] for key in (
                "source_row", "source_dataset", "source_config", "source_revision", "source_license"
            ) if key in record}
            pending.append({
                **provenance, "source": source, "source_id": record.get("id"),
                "seed_sha1": record["sha1"], "instruction_sha256": prompt_hash,
                "response_sha256": response_hash, "code_sha256": code_hash,
                "languages": sorted({language for language, _ in relevant}),
                **({"edits": edits} if response_format == "code" else {}),
                "instruction": instruction, "response": response,
            })
            if len(pending) >= batch_size:
                flush()
        flush()
        if any(len(values) == 1 for values in offsets.values()):
            raise ValueError("No tasks in a split; use more records or adjust the split fraction.")
    splits = {}
    for split, values in offsets.items():
        folder = destination / split
        required_bytes = values[-1] * np.dtype(np.int32).itemsize + 4096
        if shutil.disk_usage(folder).free < required_bytes:
            raise OSError(f"Need {required_bytes:,} free bytes to finalize {split}; "
                          "incomplete files have been preserved.")
        source = np.memmap(folder / "tokens.bin", mode="r", dtype=np.int32)
        output = np.lib.format.open_memmap(folder / "tokens.npy", mode="w+",
                                          dtype=np.int32, shape=(values[-1],))
        for start in range(0, len(source), 1_000_000):
            output[start:start + 1_000_000] = source[start:start + 1_000_000]
        output.flush()
        del source, output
        (folder / "tokens.bin").unlink()
        np.save(folder / "offsets.npy", np.asarray(values, dtype=np.int64), allow_pickle=False)
        splits[split] = {"tasks": len(values) - 1, "tokens": values[-1],
                         "target_tokens": values[-1] - len(values) + 1}
    report = {"counts": dict(counts), "splits": splits,
              "source_counts": {name: dict(stats) for name, stats in source_counts.items()},
              "languages": {name: dict(stats) for name, stats in sorted(language_counts.items())}}
    if response_format == "code":
        report["edits"] = {name: dict(stats) for name, stats in edit_counts.items()}
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/coding_python"))
    parser.add_argument("--tokenizer_path", type=Path,
                        default=Path.home() / "models/gemma-4-E2B-it")
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Override the StarCoder2 revision; other sources use registry pins.")
    parser.add_argument("--sources", nargs="+", default=["starcoder2"],
                        choices=[*PRESETS, *SOURCES],
                        help="Ordered source names, or one preset: expanded (five sources) "
                             "or multilingual (nine sources).")
    parser.add_argument("--languages", nargs="+", default=["python"],
                        help="Code-fence languages to retain, or all for supported languages.")
    parser.add_argument("--response_format", choices=RESPONSE_FORMATS, default="original",
                        help="original keeps source responses; code reduces each response "
                             "to exactly one fenced solution block without prose.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Maximum number of complete tasks per tokenizer batch.")
    parser.add_argument("--max_source_rows", type=int,
                        help="Optional per-source scan limit for a reproducible preview.")
    parser.add_argument("--deduplicate_code", action="store_true",
                        help="Also remove repeated parsed Python/code-block content.")
    parser.add_argument("--context_length", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Maximum decoder inputs per example (default: 614).")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="Discard examples above this total token count, including "
                             "prompt, response, BOS/EOS (default: 614).")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.context_length < 1 or args.max_tokens < 2 or not 0 < args.val_fraction < 1:
        parser.error("Require positive context_length, max_tokens >= 2, and val_fraction in (0, 1).")
    if args.batch_size < 1 or (args.max_source_rows is not None and args.max_source_rows < 1):
        parser.error("batch_size and max_source_rows must be positive.")
    try:
        args.sources = expand_sources(args.sources)
    except ValueError as exc:
        parser.error(str(exc))
    if args.languages == ["all"]:
        args.languages = LANGUAGES
    else:
        args.languages = tuple(dict.fromkeys(canonical_language(x) for x in args.languages))
        if any(language not in LANGUAGES for language in args.languages):
            parser.error(f"Supported languages: {', '.join(LANGUAGES)}, or all.")
    return args


def main() -> None:
    args = parse_args()
    destination = args.output_dir.expanduser().resolve()
    staging = destination.with_name(f".{destination.name}.incomplete")
    if destination.exists() or staging.exists():
        raise ValueError(f"Output already exists: {destination} or {staging}; choose a new --output_dir.")
    tokenizer_path = args.tokenizer_path.expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    special_ids = {key: getattr(tokenizer, key) for key in
                   ("bos_token_id", "eos_token_id", "pad_token_id")}
    if not tokenizer.is_fast or any(value is None for value in special_ids.values()):
        raise ValueError("A fast tokenizer with BOS, EOS, and PAD is required.")
    sources = []
    api = HfApi()
    for name in args.sources:
        source = SOURCES[name]
        revision = args.revision if name == "starcoder2" else source.revision
        resolved = api.dataset_info(source.dataset, revision=revision).sha
        sources.append(replace(source, revision=resolved))
    config = {
        "format_version": 1, "kind": "coding",
        "language": args.languages[0] if len(args.languages) == 1 else "multilingual",
        "languages": list(args.languages), "sources": [source.to_manifest() for source in sources],
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": digest(tokenizer.backend_tokenizer.to_str()),
        "special_token_ids": special_ids, "context_length": args.context_length,
        "max_tokens": args.max_tokens,
        "max_source_rows": args.max_source_rows, "batch_size": args.batch_size,
        "val_fraction": args.val_fraction, "seed": args.seed,
        "text_format": "BOS + ### Instruction\\n{instruction}\\n\\n### Response\\n{response} + EOS",
        "loss": "all non-padding next tokens, including instruction tokens",
        "split_policy": "SHA256 of seed and original StarCoder2 seed_sha1; other sources "
                        "use source name and whitespace-normalized instruction SHA256",
        "deduplication": "global whitespace-normalized instruction or exact response",
        "deduplicate_code": args.deduplicate_code,
        "code_deduplication": "Python AST without positions; other code strips trailing line "
                              "whitespace; includes language and code-block order",
        "validation": "complete labeled code fences; Python AST syntax check; other languages "
                      "are not compiled; source code is never executed",
        "response_format": args.response_format,
    }
    if args.response_format == "code":
        config.update(
            response_normalization=(
                "each response is exactly one fenced solution block with a canonical "
                "language tag; prose and output, unlabeled, or notation blocks are removed; "
                "data-format blocks and shell setup commands are removed beside a program; "
                "tasks needing several code blocks are rejected, never concatenated"),
            prompt_normalization=(
                "leaked **Answer**/**Solution** sections and a leading **Question** label "
                "are removed; prompts containing the whole solution or "
                "execution-prediction markers are rejected"),
        )
    if len(sources) == 1:
        source = sources[0]
        config.update(dataset=source.dataset, revision=source.revision,
                      source_split=source.split, source_license=source.license,
                      source_url=source.to_manifest()["source_url"])

    def records() -> Iterable[dict]:
        for source in sources:
            print(f"Reading {source.name}: {source.dataset} ({source.config or 'default'}) "
                  f"at {source.revision}", flush=True)
            rows = load_source_rows(source)
            iterator = iter(rows)
            try:
                limited = (islice(iterator, args.max_source_rows)
                           if args.max_source_rows is not None else iterator)
                for index, record in enumerate(limited):
                    yield from source_tasks(source, record, index)
            finally:
                # Close a partially consumed source when a preview reaches its
                # row limit, without opening or downloading subsequent shards.
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
                del limited, iterator, rows, close

    staging.mkdir(parents=True)
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report = curate(records(), tokenizer, staging, args.context_length, args.val_fraction,
                    args.seed, max_tokens=args.max_tokens, languages=args.languages,
                    batch_size=args.batch_size, deduplicate_code=args.deduplicate_code,
                    response_format=args.response_format,
                    excluded_languages={source.name: source.exclude_languages
                                        for source in sources if source.exclude_languages})
    (staging / "manifest.json").write_text(
        json.dumps({"config": config, **report}, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    print(json.dumps({"output_dir": str(destination), **report}, indent=2))


if __name__ == "__main__":
    main()
