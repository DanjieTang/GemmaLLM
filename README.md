# My implementation of the Gemma LLM.

## Model structure

The PyTorch architecture lives in the `model/` package:

| File | Component |
| --- | --- |
| `rope.py` | `ROPEEmbedding`: rotary positional embeddings |
| `attention.py` | `Attention`: gated attention, grouped KV heads, and LoRA |
| `feed_forward.py` | `FeedForward`: gated feed-forward network and LoRA |
| `moe.py` | `MOE`: expert routing and load-balancing loss |
| `llm_layer.py` | `LLMLayer`: attention and feed-forward decoder block |
| `llm.py` | `LLM`: decoder stack and vocabulary classifier |
| `mtp.py` | `MTPModule`: sequential multi-token prediction block with a shared output head |
| `vlm.py` | `VLM`: CLIP image encoding and text/image fusion |
| `cache.py` | `KVCache`, `PastKeyValues`, and `VLMCache`: shared cache types |

## Training data.

    a) Text data: All English Wikipedia 6.5 million pages(~2 billion tokens.).

    b) Multimodal data: COCO 2017, Open Images V7

### Coding tasks across languages

#### Prompt-to-code corpus (recommended)

Build a multilingual corpus in which every example is a prompt followed only by
code, with complete tasks of up to **2,048 total tokens**:

```bash
uv run python -m data_preprocessing.prepare_coding \
  --sources multilingual --languages all --response_format code \
  --context_length 2048 --max_tokens 2048 --deduplicate_code \
  --output_dir data/coding_multilingual_code_2048
```

Every example has the same shape, with no prose before or after the code:

````text
<bos>### Instruction
Write a Rust function that returns the n-th Fibonacci number.

### Response
```rust
fn fibonacci(n: u32) -> u64 { ... }
```<eos>
````

The fence label is always one of the 93 canonical tags in `LANGUAGES`
(`data_preprocessing/prepare_coding.py`), such as `python`, `cpp`, `csharp`,
`rust`, or `bash`; aliases like `py`, `c++`, `sh`, and `yml` never appear. At
inference, encode BOS and the prefix through `### Response\n`, and the model
chooses the language. To request one, also append its opening fence (for
example, three backticks, `rust`, and a newline), then stop at the closing
fence or EOS.

The completed local build with the existing tokenizer produced:

| Metric | Original Python corpus (614 tokens) | Expanded, original responses | Multilingual, code only |
| --- | --- | --- | --- |
| Complete tasks | 48,089 | 1,279,303 | 2,037,969 |
| Tokens | 15,293,274 | 828,767,371 | 1,008,678,059 |
| Tasks in Python | 100% | 84.5% | 51.4% |
| Language tags present | 1 | 38 | 92 |

The code-only corpus has 1,997,435 training and 40,534 validation tasks and
occupies 8.6 GB. Each task has exactly one language. After Python (1,048,205
tasks), the largest are C++ (89,767), Java (89,181), JavaScript (84,815), C#
(78,118), Rust (54,109), Swift (53,114), Go (52,062), Bash (49,917), and
TypeScript (48,172); R, PHP, Kotlin, Racket, Scala, Lua, Ruby, D, Clojure,
HTML, Haskell, and Julia each have 15,000 to 31,000. The median task is 404
tokens, the 95th percentile 1,123, and 93% of tasks fit in 1,024 tokens. About
10% of prompts are in Chinese, mostly from `opencoder_diverse`; their answers
are still code. The build's `validation_report.json` records the integrity
checks.

For a 1,024-token training context, build the same corpus with
`--context_length 1024 --max_tokens 1025 --output_dir data/coding_multilingual_code_1024`.
Every training input is then at most 1,024 tokens; the stored task holds one
more, its final `<eos>` target. That build keeps 1,890,497 tasks (1,852,926
training and 37,571 validation, 92.8% of the 2,048-token corpus) and 821,732,297
tokens in 7.2 GB. Its median task is 381 tokens and its 95th percentile 886.
Python rises to 53.6% of tasks because the dropped long tasks are more often in
other languages; CodeContests, McEval, and OpenCoder package lose the most.
Rebuilding, rather than filtering the 2,048-token corpus, lets a shorter
duplicate of a dropped long task take its place.

`--response_format code` normalizes each source task as follows. The default,
`original`, keeps source responses unchanged.

- **One solution block.** Explanations, sample output, and unlabeled, `text`,
  or JSON blocks are removed; JSON-only answers in these sources mostly wrap
  prose steps or escaped code strings, so JSON is never the answer. A
  configuration block (YAML, XML, TOML, INI) or a short shell block that only
  installs, builds, runs, or changes directory is removed when it accompanies a
  program, and becomes the answer only when nothing else is present. When
  several same-language blocks remain and one contains all the others, it is
  kept as the complete program. Otherwise the task is rejected, never
  concatenated.
- **Readable fences.** Fences nested in Markdown lists are parsed; their list
  indentation and any common indentation are removed. A code block that itself
  contains a fence gets a longer outer fence.
- **Clean prompts.** Leaked `**Answer**`/`**Solution**` sections, which some
  OpenCoder prompts include along with the code, are cut from the prompt, and a
  leading `**Question**:` label is removed. Prompts that already contain the
  whole solution, and execution-prediction tasks (`[THOUGHT]`/`[ANSWER]`), are
  rejected.
- **Intact boundaries.** Tasks whose text tokenizes to BOS, EOS, or PAD (for
  example, a literal `<eos>` in a prompt) are rejected, so EOS only ends tasks.

Each task in `tasks.jsonl` lists its `edits`, and the manifest counts them per
source alongside every rejection reason. Python must parse; other languages
are not compiled, and no code is executed.

The `multilingual` preset reads nine pinned sources in this order;
deduplication keeps the first copy:

| Source name | Dataset / subset | Accepted tasks | Publisher's dataset license |
| --- | --- | --- | --- |
| `starcoder2` | [StarCoder2 self-alignment](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k), 50,661 execution-filtered Python tasks | 48,835 | ODC-BY |
| `opencoder_educational` | [OpenCoder stage 2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), `educational_instruct` | 88,971 | MIT |
| `opencoder_package` | [OpenCoder stage 2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), `package_instruct` | 167,665 | MIT |
| `opencoder_mceval` | [OpenCoder stage 2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), `mceval_instruct`: about 1,000 tasks per language, derived from McEval-Instruct (CC-BY-SA-4.0) | 33,019 | MIT |
| `code_contests` | [CodeContests](https://huggingface.co/datasets/deepmind/code_contests), train split: competitive-programming problems with human-written correct solutions | 25,203 | CC-BY-4.0 |
| `codefeedback` | [CodeFeedback filtered instructions](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 93,037 | Apache-2.0 |
| `opencoder_evol` | [OpenCoder stage 2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), `evol_instruct` | 25,415 | MIT |
| `ling_coder` | [Ling-Coder-SFT](https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT), 5.1M chat pairs, Python excluded | 814,619 | Apache-2.0 |
| `opencoder_diverse` | [OpenCoder stage 1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1), `largescale_diverse_instruct` | 741,205 | MIT |

Ling-Coder-SFT is 73% Python, so its registry entry sets
`exclude_languages=("python",)` to rebalance the mix; remove that field to also
take its Python tasks. Its 26 shards (6.3 GB) are downloaded one at a time to a
temporary directory and deleted after reading (`cache_shards=False`), so they
never accumulate in the Hugging Face cache. A third of its rows are
execution-prediction tasks, which are rejected.

CodeContests is the only source of human-written, contest-verified solutions.
Each problem becomes one task per language (C++, Python 3, Java), pairing the
statement plus a closing line such as *Write a C++ program that solves this
problem, reading from standard input and writing to standard output.* with that
language's shortest correct solution, which carries the least template and
debugging code. Prompts therefore stay unique, and all of a problem's tasks
share one split. Only the train split is read, so CodeContests' `valid` and
`test` problems remain a held-out benchmark. Its 7.6 GB of shards are mostly
test cases; they are streamed through a temporary directory, and only the
statement and solution columns are decoded.

Check upstream terms before redistributing a built corpus.

#### Original-response corpus

`--response_format original` (the default) keeps each source response,
including explanations around the code. The `expanded` preset uses five of the
sources above, all except `opencoder_mceval`, `code_contests`, `opencoder_evol`,
and `ling_coder`:

```bash
uv run python -m data_preprocessing.prepare_coding \
  --sources expanded --languages all \
  --context_length 2048 --max_tokens 2048 --deduplicate_code \
  --output_dir data/coding_multilingual_2048
```

That build (table above) contains 1,253,746 training tasks and 25,557
validation tasks and occupies about 7.0 GiB. It was made when `LANGUAGES` had
38 tags, so rerunning it now accepts some additional languages. Python appears
in 84.5% of its tasks; language counts overlap there because a task can contain
code in several languages.

`--languages all` enables every supported tag. Select a subset with, for
example, `--languages python javascript java`. Use explicit source names instead
of a preset to select or order the inputs. The optional `opencoder_realuser`
source is also available from OpenCoder stage 1. The source registry lives in
`data_preprocessing/coding_sources.py`.

#### Shared behavior

Source rows are read incrementally in bounded batches. Parquet shards are
downloaded to the Hugging Face cache as they are reached, then read locally,
except for sources with `cache_shards=False`; CodeFeedback's JSON source is
streamed. Allow about 4 GB for the source cache in addition to the prepared
corpus and temporary output files, which need roughly 1.5 times the final size
while the token array is finalized. The full source text is not held in memory,
although deduplication hashes and task offsets grow with the accepted corpus.

The command uses the existing local tokenizer and writes
`{train,val}/tasks.jsonl`, flat memory-mapped `tokens.npy` arrays with
`offsets.npy`, and a `manifest.json` under the output directory. Each task
retains its source ID, dataset, subset, revision, license, and detected
languages. The manifest records accepted/rejected counts per source, task/token
counts per language, and train/validation totals.

Tasks require nonempty prompts/responses and complete, labeled code blocks in
the selected languages. Python blocks must parse successfully; other languages
are not compiled, and no source code is executed. The builder removes duplicate
whitespace-normalized prompts or exact responses across all sources.
`--deduplicate_code` additionally removes tasks with identical ordered blocks in
the selected languages, comparing Python ASTs without source positions and other
code after removing trailing line whitespace. This catches repeated solutions
with different explanations, but does not establish semantic uniqueness or
benchmark decontamination.

Complete examples above `--max_tokens` or `--context_length + 1` are excluded and
counted; solutions are never truncated or chunked. `--context_length` limits
decoder inputs, which have one fewer token than a complete example. Both limits
can be raised for longer tasks, with a matching training context. The split is
deterministic and approximately 98/2: original StarCoder2 seed groups stay
together, while other sources group normalized prompts within their source.
Use `--val_fraction` and `--seed` to change the split.

For a small reproducible preview, add `--max_source_rows 1000` and choose a fresh
output directory; the limit applies to each source before filtering.
Preprocessing `--batch_size` controls only tokenization batches. Existing output
directories are never overwritten. Failed builds remain in a sibling directory
named `.<output_name>.incomplete`; use a fresh output path to retry.

The legacy command still builds only the original Python source with the
614-token limit and writes to `data/coding_python`:

```bash
uv run python -m data_preprocessing.prepare_coding
```

Train exclusively on the code-only corpus:

```bash
uv run python train.py --coding_dir data/coding_multilingual_code_2048 \
  --embeddings_path data/gemma-4-31B-it-embeddings.pt \
  --max_context_length 2048 --batch_size 1 \
  --output_dir checkpoints/coding_multilingual_code_2048
```

To watch generations during training, add `--text_inference_every N`: every N
iterations it prompts with a validation task through `### Response\n` and prints
the greedy continuation beside the reference. Wikipedia windows are prompted with
their first half. `--image_inference_every N` does the same for image
annotations. Both are off by default, and `--inference_max_new_tokens` caps the
generated length.

For the 1,024-token build, point `--coding_dir` at
`data/coding_multilingual_code_1024` and pass `--max_context_length 1024`. The
trainer rejects a corpus built for a longer context than `--max_context_length`.

`train_mtp.py` accepts the same coding options, and any corpus built by
`prepare_coding` can be passed to `--coding_dir`. `--coding_dir` rejects
combinations with Wikipedia, image annotations, or legacy token arrays. The
collator masks padding, and the existing objective trains on both prompt and
response tokens. This is full-sequence language-model training, not
response-only supervised fine-tuning. To initialize from existing weights,
supply `--init_checkpoint` and matching architecture options. Training still
uses the existing VLM wrapper, which loads CLIP even for text-only batches. The
larger context and vocabulary can require substantial memory; batch size 1 is a
starting point, not a guarantee.

The text format is `BOS + ### Instruction\n{prompt}\n\n### Response\n{solution} + EOS`.
For generation, encode BOS and that prefix through `### Response\n` without EOS.
The tokenizer must match preprocessing and the embeddings.

Local Python checks validate syntax, not solution correctness. Upstream execution
tests are not rerun. Keep external coding benchmarks separate. Corpus size alone
does not establish coding quality; evaluate held-out tasks after training.

## Throughput optimizations

    a) KV cache: 5.476× speedup with a 1,024-token output.

    b) MTP Speculative decoding: 1.738× speedup on an English Wikipedia prediction task.

    c) CUDA cublas implementation: 1.33× speedup for a tensor size of (2, 64, 512) — (batch, sequence, hidden).

    d) Flash attention

Original speed = 28.09 tokens/s

Latest speed = 355.56 tokens/s

## Key insights from this implementation.

    a)RMS Normalization

    b)ROPE Embedding

    c)MultiQueryAttention

    d)GeGLU Activations

    e)Pre-Norm Transformers

    f)Mixtral of Experts

    g)LoRA: Low-Rank Adaptation of Large Language Models

    h)Gated Attention for Large Language Models

    i)Late fusion for LLM image capability

    j)FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## Training detail.

    a) 199.7 Million parameters Mixture of Experts Architecture

    b) Contextual length of 256 tokens.

## Image annotation.

    a) Used gemma 4 31b served with vllm on dgx spark to annotate 2 million(2,074,056) images.
