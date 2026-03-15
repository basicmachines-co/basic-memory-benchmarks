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

### 4) Optional judge benchmark

```bash
uv run bm-bench run judge --run-dir benchmarks/runs/<run-id>
```

### 5) Publish run artifacts

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
