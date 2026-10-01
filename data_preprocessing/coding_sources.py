"""Pinned instruction datasets and adapters for the coding corpus builder."""

from dataclasses import asdict, dataclass
import hashlib
import re


@dataclass(frozen=True)
class CodingSource:
    """Describe a reproducible dataset and its instruction/response columns.

    With ``messages_field``, the instruction and response fields name the chat
    roles of the first exchange. ``exclude_languages`` drops tasks whose code is
    in those languages, and ``cache_shards=False`` deletes each downloaded
    Parquet shard after reading it instead of keeping it in the Hugging Face cache.
    ``shard_prefix`` restricts the Parquet shards read, ``columns`` limits the
    columns decoded, and ``adapter`` names a function in ``ADAPTERS`` that turns
    one source row into any number of tasks.
    """

    name: str
    dataset: str
    revision: str
    license: str
    instruction_field: str
    response_field: str
    config: str | None = None
    split: str = "train"
    language_field: str | None = None
    group_field: str | None = None
    messages_field: str | None = None
    id_field: str | None = None
    exclude_languages: tuple[str, ...] = ()
    cache_shards: bool = True
    shard_prefix: str | None = None
    columns: tuple[str, ...] | None = None
    adapter: str | None = None

    def to_manifest(self) -> dict:
        return {
            **asdict(self),
            "exclude_languages": list(self.exclude_languages),
            "columns": None if self.columns is None else list(self.columns),
            "source_url": (
                f"https://huggingface.co/datasets/{self.dataset}/tree/{self.revision}"
            ),
        }


_OPENCODER_STAGE1_REVISION = "1bcab575f5e2d1c1fd6652720418524c27b3d58b"
_OPENCODER_STAGE2_REVISION = "7d28f40d579edd7c24402d17d0c7639f991e6f8d"


SOURCES = {
    source.name: source
    for source in (
        CodingSource(
            name="starcoder2",
            dataset="bigcode/self-oss-instruct-sc2-exec-filter-50k",
            revision="356bb069eee815daa6e23e9a282eeefe1490ad44",
            license="odc-by",
            instruction_field="instruction",
            response_field="response",
            group_field="sha1",
        ),
        CodingSource(
            name="opencoder_educational",
            dataset="OpenCoder-LLM/opc-sft-stage2",
            revision=_OPENCODER_STAGE2_REVISION,
            license="mit",
            instruction_field="instruction",
            response_field="output",
            config="educational_instruct",
        ),
        CodingSource(
            name="opencoder_package",
            dataset="OpenCoder-LLM/opc-sft-stage2",
            revision=_OPENCODER_STAGE2_REVISION,
            license="mit",
            instruction_field="instruction",
            response_field="output",
            config="package_instruct",
        ),
        CodingSource(
            name="codefeedback",
            dataset="m-a-p/CodeFeedback-Filtered-Instruction",
            revision="a08c213a9748c66c15d0225814be80a2e77adf4a",
            license="apache-2.0",
            instruction_field="query",
            response_field="answer",
            language_field="lang",
        ),
        CodingSource(
            name="opencoder_diverse",
            dataset="OpenCoder-LLM/opc-sft-stage1",
            revision=_OPENCODER_STAGE1_REVISION,
            license="mit",
            instruction_field="instruction",
            response_field="output",
            config="largescale_diverse_instruct",
        ),
        CodingSource(
            name="opencoder_realuser",
            dataset="OpenCoder-LLM/opc-sft-stage1",
            revision=_OPENCODER_STAGE1_REVISION,
            license="mit",
            instruction_field="instruction",
            response_field="output",
            config="realuser_instruct",
        ),
        # About 1,000 tasks for each of 30+ languages, derived from
        # McEval-Instruct (CC-BY-SA-4.0 upstream).
        CodingSource(
            name="opencoder_mceval",
            dataset="OpenCoder-LLM/opc-sft-stage2",
            revision=_OPENCODER_STAGE2_REVISION,
            license="mit",
            instruction_field="instruction",
            response_field="output",
            config="mceval_instruct",
        ),
        CodingSource(
            name="opencoder_evol",
            dataset="OpenCoder-LLM/opc-sft-stage2",
            revision=_OPENCODER_STAGE2_REVISION,
            license="mit",
            instruction_field="instruction",
            response_field="output",
            config="evol_instruct",
        ),
        # 5.1M chat pairs, 73% Python. Python is excluded to rebalance the mix,
        # and the 6.3 GB of shards are deleted as they are read.
        CodingSource(
            name="ling_coder",
            dataset="inclusionAI/Ling-Coder-SFT",
            revision="7c631c8eb4f9b72f724a8cc88eb5c19f1cdecfa8",
            license="apache-2.0",
            instruction_field="HUMAN",
            response_field="ASSISTANT",
            messages_field="messages",
            id_field="mid",
            exclude_languages=("python",),
            cache_shards=False,
        ),
        # Competitive-programming statements with human-written correct
        # solutions. Only the train split is read; valid and test stay held out.
        # The 7.6 GB of shards are mostly test cases, which are never decoded.
        CodingSource(
            name="code_contests",
            dataset="deepmind/code_contests",
            revision="802411c3010cb00d1b05bad57ca77365a3c699d6",
            license="cc-by-4.0",
            instruction_field="description",
            response_field="solutions",
            cache_shards=False,
            shard_prefix="data/train-",
            columns=("name", "description", "solutions", "input_file", "output_file"),
            adapter="code_contests",
        ),
    )
}

EXPANDED_SOURCES = (
    "starcoder2",
    "opencoder_educational",
    "opencoder_package",
    "codefeedback",
    "opencoder_diverse",
)

# Ordered by curation quality because deduplication keeps the first copy.
MULTILINGUAL_SOURCES = (
    "starcoder2",
    "opencoder_educational",
    "opencoder_package",
    "opencoder_mceval",
    "code_contests",
    "codefeedback",
    "opencoder_evol",
    "ling_coder",
    "opencoder_diverse",
)

PRESETS = {"expanded": EXPANDED_SOURCES, "multilingual": MULTILINGUAL_SOURCES}


def first_exchange(messages, user_role: str, assistant_role: str) -> tuple[str | None, str | None]:
    """Return the first user turn and the assistant turn that answers it."""
    if not isinstance(messages, list):
        return None, None
    for index, message in enumerate(messages[:-1]):
        reply = messages[index + 1]
        if (isinstance(message, dict) and isinstance(reply, dict)
                and str(message.get("role", "")).lower() == user_role.lower()
                and str(reply.get("role", "")).lower() == assistant_role.lower()):
            return message.get("content"), reply.get("content")
    return None, None


def normalize_record(source: CodingSource, record: dict, row_index: int) -> dict:
    """Adapt column names without changing source text or executing any code.

    Original StarCoder2 seed hashes retain their historical split assignments.
    Other datasets group whitespace-equivalent prompts within their namespace;
    row order and reused source IDs therefore do not alter split assignments.
    """
    if source.messages_field is not None:
        instruction, response = first_exchange(record.get(source.messages_field),
                                               source.instruction_field,
                                               source.response_field)
    else:
        instruction = record.get(source.instruction_field)
        response = record.get(source.response_field)
    if source.group_field is not None:
        group = record.get(source.group_field)
        if source.name != "starcoder2" and isinstance(group, str) and group:
            group = f"{source.name}:{group}"
    else:
        prompt = " ".join(instruction.split()) if isinstance(instruction, str) else ""
        group = f"{source.name}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"
    source_id = record.get(source.id_field or "id")
    if source_id is None:
        source_id = record.get("seq_id")
    if source_id is None:
        source_id = row_index
    normalized = {
        "instruction": instruction,
        "response": response,
        "sha1": group,
        "id": source_id,
        "source": source.name,
        "source_row": row_index,
        "source_dataset": source.dataset,
        "source_config": source.config,
        "source_revision": source.revision,
        "source_license": source.license,
    }
    if source.language_field is not None:
        normalized["language"] = record.get(source.language_field)
    return normalized


# CodeContests solution language codes; Python 2 (1) and unknown (0) are skipped.
CODE_CONTESTS_LANGUAGES = {2: ("cpp", "C++"), 3: ("python", "Python 3"), 4: ("java", "Java")}


def fence_code(language: str, code: str) -> str:
    """Fence raw source, lengthening the fence when the code contains one."""
    runs = [len(run) + 1 for run in re.findall(r"^[ \t]*(`{3,})", code, re.MULTILINE)]
    fence = "`" * max([3, *runs])
    return f"{fence}{language}\n{code.strip(chr(10))}\n{fence}"


def code_contests_tasks(source: CodingSource, record: dict, row_index: int) -> list[dict]:
    """Expand one CodeContests problem into one task per solution language.

    Each prompt is the statement plus the target language and I/O convention;
    the response is that language's shortest correct solution, which carries the
    least template and debugging code, so prompts stay unique. All tasks of a
    problem share one split group.
    """
    solutions = record.get("solutions") or {}
    by_language = {}
    for language, code in zip(solutions.get("language") or [], solutions.get("solution") or []):
        if language in CODE_CONTESTS_LANGUAGES and isinstance(code, str) and code.strip():
            by_language.setdefault(language, []).append(code)
    description, name = record.get("description"), record.get("name")
    reads = (f"the file {record['input_file']}" if record.get("input_file")
             else "standard input")
    writes = (f"the file {record['output_file']}" if record.get("output_file")
              else "standard output")
    tasks = []
    for language, codes in sorted(by_language.items()):
        tag, display = CODE_CONTESTS_LANGUAGES[language]
        instruction = None
        if isinstance(description, str) and description.strip():
            instruction = (f"{description.strip()}\n\nWrite a {display} program that solves "
                           f"this problem, reading from {reads} and writing to {writes}.")
        tasks.append({
            "instruction": instruction,
            "response": fence_code(tag, min(codes, key=len)),
            "sha1": f"{source.name}:{name}",
            "id": f"{name}:{tag}",
            "source": source.name,
            "source_row": row_index,
            "source_dataset": source.dataset,
            "source_config": source.config,
            "source_revision": source.revision,
            "source_license": source.license,
        })
    return tasks


ADAPTERS = {"code_contests": code_contests_tasks}


def source_tasks(source: CodingSource, record: dict, row_index: int) -> list[dict]:
    """Convert one source row into its tasks, through the source's adapter if any."""
    if source.adapter is not None:
        return ADAPTERS[source.adapter](source, record, row_index)
    return [normalize_record(source, record, row_index)]


def expand_sources(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Expand a preset or validate an explicit, ordered source selection."""
    names = tuple(names)
    if len(names) == 1 and names[0] in PRESETS:
        return PRESETS[names[0]]
    if not names:
        raise ValueError("Select at least one coding source.")
    if any(name in PRESETS for name in names):
        raise ValueError("Use a preset alone, or list individual coding sources.")
    if len(set(names)) != len(names):
        raise ValueError("Select distinct coding sources; repeated names are not allowed.")
    unknown = [name for name in names if name not in SOURCES]
    if unknown:
        raise ValueError(f"Unknown coding sources: {', '.join(unknown)}.")
    return names
