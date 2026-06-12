# basic-memory-benchmarks

Standalone, reproducible benchmark suite for comparing Basic Memory against competitor memory systems.

## Goals

- Deterministic retrieval benchmarks (Recall@5/10, MRR, Precision@5, content-hit, latency)
- Optional LLM-as-judge scoring (Pydantic Evals)
- Public artifacts with provenance and reproducibility metadata
- Clean dependency isolation from the core `basic-memory` repository

## Current v1 Scope

- Providers:
  - `bm-local` (warm `bm mcp` stdio session)
  - `bm-cloud` (optional, credential-gated)
  - `mem0-local`
  - `zep-reference` (reference-only in v1)
- Datasets:
  - LoCoMo (primary)
  - LongMemEval scaffold (placeholder)
  - Built-in synthetic smoke corpus

## Installation

```bash
uv sync --group dev
```

Optional judge dependencies:

```bash
uv sync --group dev --extra judge
```

## Quickstart

### 1) Fetch LoCoMo dataset

```bash
uv run bm-bench datasets fetch --dataset locomo
```

### 2) Convert LoCoMo into benchmark corpus

```bash
uv run bm-bench convert locomo
```

### 3) Run retrieval benchmark

```bash
uv run bm-bench run retrieval \
  --providers bm-local,mem0-local \
  --corpus-dir benchmarks/generated/locomo/docs \
  --queries-path benchmarks/generated/locomo/queries.json
```

### 4) End-to-end QA scoring

Generates an answer per query from each provider's retrieved context, then
grades it against the expected answer with an LLM judge. This is the stage that
produces benchmark-comparable accuracy numbers; retrieval metrics alone measure
only the search layer.

```bash
uv run bm-bench run qa --run-dir benchmarks/runs/<run-id> \
  --answerer claude:claude-haiku-4-5 \
  --judge claude:claude-sonnet-4-6
```

Runner specs select the transport:

- `claude:<model>` — Claude Code CLI in print mode. Bills the operator's Claude
  subscription plan; no API key needed. Requires `claude` on PATH.
- `openai-compat:<model>@<base_url>` — any OpenAI-compatible endpoint (Ollama,
  LM Studio, vLLM, OpenAI). Set `OPENAI_API_KEY` if the endpoint requires auth.

The same answerer and judge are used for every provider in the run, so
cross-provider comparisons hold the model constant. Answer and judge prompts
are fixed in `scoring/qa.py`; the answerer is instructed to abstain ("I don't
know") when the retrieved memories don't contain the answer, and abstention is
graded correct only when the gold answer marks the question unanswerable
(LoCoMo adversarial cases).

### 5) Optional retrieval-context judge (legacy)

Scores whether the *retrieved context* contains the expected answer, without
answer generation:

```bash
uv run bm-bench run judge --run-dir benchmarks/runs/<run-id>
```

### 6) Publish run artifacts

```bash
uv run bm-bench publish --run-dir benchmarks/runs/<run-id>
```

## Basic Memory source policy

By default this project tracks Basic Memory from `main`.

Each run manifest stores:
- BM source (`github main` or local path override)
- resolved BM commit SHA

Local override:

```bash
uv run bm-bench run retrieval \
  --bm-local-path /path/to/basic-memory
```

## Mem0 local requirements

`mem0-local` requires model credentials available in environment.

At minimum, set:

```bash
export OPENAI_API_KEY=...
```

If unavailable, provider status will be recorded as `SKIPPED(reason)`.

## BM indexing readiness

`bm-local` verifies index readiness before querying.

- If the installed `bm` supports `bm status --json`, readiness is polled from that output.
- If `--json` is not available in the installed `bm`, the benchmark proceeds after reindex.

## Run Artifacts

Per run (`benchmarks/runs/<run-id>/`):

- `manifest.json`
- `provider-status.json`
- `per-query-retrieval.jsonl`
- `retrieval-summary.json`
- `per-query-qa.jsonl` (optional)
- `qa-summary.json` (optional)
- `per-query-judge.jsonl` (optional)
- `judge-summary.json` (optional)
- `summary.md`

## Just commands

```bash
just bench-smoke
just bench-fetch-locomo
just bench-convert-locomo
just bench-run-bm-local
just bench-run-mem0-local
just bench-run-full
just bench-judge
just bench-publish RUN_DIR=benchmarks/runs/<run-id>
```

## Notes on dataset publication

Dataset publication follows licensing constraints:
- If redistribution is permitted: snapshot + checksum may be published.
- If not: canonical source links + downloader + checksum verification are published.
