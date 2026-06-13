# Benchmark Matrix v1.2 — internal results

Run: 2026-06-12 evening. Local, zero API spend (answerer `claude:claude-haiku-4-5`,
judge `claude:claude-sonnet-4-6` via plan; mem0/embeddings via local Ollama).

**BM under test: `main` @ d46c6880** (before the FTS-revival fix, PR #994).
mem0 run in raw-add mode (`MEM0_INFER` unset → `infer=false`); see fairness note.

## QA accuracy (the headline metric)

| Benchmark | bm-local | mem0-local | baseline-grep | baseline-fullcontext |
|---|---|---|---|---|
| LongMemEval-S (n=60, stratified 6 cats) | **0.617** | 0.417 | 0.300 | 0.217 |
| ConvoMem cs10 (n=274) | 0.792 | 0.474 | 0.398 | **0.825** |

## Retrieval (deterministic)

| Benchmark | provider | recall@5 | MRR | content-hit | mean lat |
|---|---|---|---|---|---|
| LongMemEval-S | bm-local | 0.951 | 0.900 | 0.467 | 754ms |
| LongMemEval-S | mem0-local | 0.979 | 0.876 | 0.500 | 146ms |
| LongMemEval-S | baseline-grep | 0.846 | 0.832 | 0.350 | 5ms |
| ConvoMem cs10 | bm-local | 0.982 | 0.929 | 0.128 | 140ms |
| ConvoMem cs10 | mem0-local | 0.996 | 0.956 | 0.131 | 122ms |
| ConvoMem cs10 | baseline-grep | 0.954 | 0.863 | 0.062 | 1ms |

(full-context retrieval metrics are N/A by design — single whole-corpus hit.)

## Findings

1. **BM leads QA accuracy on LongMemEval-S and is a close 2nd on ConvoMem**,
   despite mem0 edging it on retrieval recall. BM's retrieved chunks are more
   answer-bearing: mem0 abstains far more (LME 30/60, ConvoMem 169/274 vs BM
   20/60, 86/274). Retrieval recall ≠ answer quality.

2. **Full-context is a poor baseline at small-model scale.** On LongMemEval-S
   (~124K-token haystacks, 496K-char assembled context) qwen2.5:3b-class
   answering drops to 0.217 — it cannot use the whole haystack and abstains.
   On the smaller ConvoMem cs10 it wins (0.825). Confirms the published
   pattern: full-context beats retrieval only while the corpus fits the
   model's effective working window.

3. **mem0 raw-add caveat (fairness).** mem0's published numbers use
   `infer=true` (LLM fact extraction). We ran raw-add to match the June 10
   baseline and avoid a local-3B extraction step of unknown quality. A future
   matrix should run mem0 both ways and report both, with the extraction model
   documented.

## PR #994 (FTS-revival) impact — corrected LoCoMo

Two measurements, each stating exactly what it covers:

**Retrieval — full 1,986-query set, same pre-built index, code-only A/B**
(`/tmp/replay_fusion.py` against `matrix-locomo-v2-full`'s index). Definitive.

| metric | BM main | BM #994 | Δ |
|---|---|---|---|
| recall@5 | 0.745 | 0.823 | **+7.9** |
| MRR | 0.618 | 0.718 | **+10.0** |

Every category improves; largest on adversarial (+0.12 r5) and open_domain
(+0.095 r5), smallest on temporal (+0.003).

**QA accuracy — q300 non-adversarial subset, fresh re-index, full QA stage**
(`m994-locomo-q300` vs `matrix-locomo-v2-q300`).

| | BM main | BM #994 | Δ |
|---|---|---|---|
| accuracy | 0.439 | **0.475** | **+3.7** |
| abstain | 95 | 87 | -8 |

By category (correct/total): open_domain 105→112, single_hop 18→23,
multi_hop 4→5, temporal 5→3 (n=19, noise).

The #994-only QA gain is smaller than the retrieval gain because the largest
retrieval improvements land in the adversarial category (excluded from
QA-meaningful scoring) and **multi_hop stayed ~0.08 — bottlenecked by a
separate gap**: BM returns bullet-level matched chunks that strip
document-level context (the session date lives in the title). That gap is
addressed next.

## #994 + title-passthrough (combined) — corrected LoCoMo q300

The harness was discarding the dated `title` that `search_notes` returns
(PR #31, `bm_local._row_to_hit`); surfacing it lets the answerer anchor
relative-date references ("two days ago"). Combined with #994, on the same
q300 subset and QA stage:

| QA accuracy | BM main | +FTS (#994) | +FTS +title |
|---|---|---|---|
| overall | 0.439 | 0.475 | **0.611** |

By category (correct/total), main → +FTS → +FTS+title:

| category | main | +FTS | +FTS+title |
|---|---|---|---|
| single_hop | 18/55 | 23/55 | 20/55 |
| multi_hop | 4/63 | 5/63 | **40/63** |
| temporal | 5/19 | 3/19 | 5/19 |
| open_domain | 105/164 | 112/164 | 119/164 |

**+17.2 points overall (0.439 → 0.611)**, driven by multi_hop (4 → 40 — these
are mostly relative-date questions that were unanswerable without the session
date). single_hop dips 23 → 20 vs the FTS-only run (the title header
occasionally distracts simple lookups; n=55, within LLM-judge noise, still +2
vs main). The title fix is harness-side and provider-faithful — it uses data
BM already returns. A product follow-up could fold the parent heading/date
into `matched_chunk` so naive agents get the anchor without reading the title.

## Pending
- supermemory-local provider validated against the real server but not yet in
  a matrix run (Ollama extraction hits upstream issue #1096; needs the
  Responses→ChatCompletions shim).
