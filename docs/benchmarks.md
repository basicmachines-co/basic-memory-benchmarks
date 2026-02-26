# Benchmark Architecture

## Providers

- `bm-local`: Basic Memory local execution via warm `bm mcp` stdio session
- `bm-cloud`: Optional cloud mode (credential gated)
- `mem0-local`: Mem0 package execution in local environment
- `zep-reference`: reference-only placeholder in v1

## Artifact contract

Each run writes:

- `manifest.json`
- `provider-status.json`
- `per-query-retrieval.jsonl`
- `retrieval-summary.json`
- `summary.md`

Optional judge outputs:

- `per-query-judge.jsonl`
- `judge-summary.json`

## Fairness

The runner validates query-set alignment across providers and records warnings in
`retrieval-summary.json` and `summary.md`.
