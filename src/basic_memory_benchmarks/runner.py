"""Benchmark runner orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from basic_memory_benchmarks.exceptions import ProviderSkippedError
from basic_memory_benchmarks.fairness import validate_fairness
from basic_memory_benchmarks.models import (
    DatasetProvenance,
    PerQueryRetrievalResult,
    ProviderStatus,
    QueryCase,
    RetrievalSummary,
    RunConfig,
    RunManifest,
    RuntimeInfo,
)
from basic_memory_benchmarks.providers import create_provider
from basic_memory_benchmarks.providers.base import BenchmarkProvider
from basic_memory_benchmarks.reporting.artifacts import write_artifacts
from basic_memory_benchmarks.scoring.judge import run_optional_judge
from basic_memory_benchmarks.scoring.retrieval import evaluate_query, summarize_provider
from basic_memory_benchmarks.utils import git_sha, resolve_remote_main_sha, runtime_info, utc_now_iso


def load_queries(path: Path) -> list[QueryCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Query file must contain a list: {path}")
    return [QueryCase.model_validate(item) for item in payload]


def _resolve_bm_sha(run_config: RunConfig) -> str | None:
    if run_config.bm_local_path:
        local_sha = git_sha(Path(run_config.bm_local_path))
        if local_sha:
            return local_sha
    return resolve_remote_main_sha("https://github.com/basicmachines-co/basic-memory")


def run_retrieval(
    *,
    run_config: RunConfig,
    dataset: DatasetProvenance,
    provider_factory: Callable[[str], BenchmarkProvider] = create_provider,
) -> Path:
    queries = load_queries(Path(run_config.queries_path))
    corpus_path = Path(run_config.corpus_dir)
    output_root = Path(run_config.output_root)
    run_dir = output_root / run_config.run_id

    retrieval_rows: list[PerQueryRetrievalResult] = []
    provider_status: list[ProviderStatus] = []
    summaries: list[RetrievalSummary] = []

    rows_by_provider: dict[str, list[PerQueryRetrievalResult]] = {}

    for provider_name in run_config.providers:
        provider = provider_factory(provider_name)
        provider_rows: list[PerQueryRetrievalResult] = []

        try:
            provider.ingest(corpus_path, run_config)
            for query in queries:
                started = time.perf_counter()
                hits = provider.search(query.query, run_config.top_k, run_config)
                latency_ms = (time.perf_counter() - started) * 1000.0
                provider_rows.append(
                    evaluate_query(
                        provider=provider_name,
                        query=query,
                        hits=hits,
                        latency_ms=latency_ms,
                    )
                )

            summary = summarize_provider(provider_name, provider_rows)
            summaries.append(summary)
            retrieval_rows.extend(provider_rows)
            rows_by_provider[provider_name] = provider_rows
            provider_status.append(
                ProviderStatus(
                    provider=provider_name,
                    state="ok",
                    metadata=provider.version_info(),
                )
            )
        except ProviderSkippedError as exc:
            provider_status.append(
                ProviderStatus(provider=provider_name, state="skipped", reason=str(exc))
            )
            if not run_config.allow_provider_skip:
                raise
        except Exception as exc:
            provider_status.append(
                ProviderStatus(provider=provider_name, state="error", reason=str(exc))
            )
            if not run_config.allow_provider_skip:
                raise
        finally:
            try:
                provider.cleanup(run_config)
            except Exception:
                # Cleanup errors should not mask run state.
                pass

    fairness_warnings = validate_fairness(rows_by_provider)

    os_name, py_version = runtime_info()
    manifest = RunManifest(
        run_id=run_config.run_id,
        created_at_utc=utc_now_iso(),
        benchmark_git_sha=git_sha(Path.cwd()) or "unknown",
        bm_source=run_config.bm_source,
        bm_resolved_sha=_resolve_bm_sha(run_config),
        bm_local_path=run_config.bm_local_path,
        mem0_version=next(
            (
                status.metadata.get("mem0ai")
                for status in provider_status
                if status.provider == "mem0-local" and status.metadata
            ),
            None,
        ),
        provider_versions={
            status.provider: status.metadata
            for status in provider_status
            if status.metadata
        },
        dataset=dataset,
        runtime=RuntimeInfo(os=os_name, python_version=py_version, started_at_utc=utc_now_iso()),
        config=run_config,
    )

    write_artifacts(
        run_dir=run_dir,
        manifest=manifest,
        provider_status=provider_status,
        retrieval_rows=retrieval_rows,
        retrieval_summaries=summaries,
        fairness_warnings=fairness_warnings,
    )
    return run_dir


def run_judge(
    *,
    run_dir: Path,
    model: str,
) -> Path:
    retrieval_path = run_dir / "per-query-retrieval.jsonl"
    if not retrieval_path.exists():
        raise FileNotFoundError(f"Missing retrieval artifact: {retrieval_path}")

    rows: list[PerQueryRetrievalResult] = []
    with retrieval_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(PerQueryRetrievalResult.model_validate(json.loads(line)))

    grouped: dict[str, list[PerQueryRetrievalResult]] = {}
    for row in rows:
        grouped.setdefault(row.provider, []).append(row)

    judge_rows = []
    judge_summaries = []
    for provider, provider_rows in grouped.items():
        provider_case_results, provider_summary = run_optional_judge(
            provider_rows,
            provider=provider,
            model=model,
        )
        judge_rows.extend(provider_case_results)
        judge_summaries.append(provider_summary)

    judge_jsonl = run_dir / "per-query-judge.jsonl"
    with judge_jsonl.open("w", encoding="utf-8") as file:
        for row in judge_rows:
            file.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")

    judge_summary_path = run_dir / "judge-summary.json"
    judge_summary_path.write_text(
        json.dumps({"providers": [item.model_dump(mode="json") for item in judge_summaries]}, indent=2),
        encoding="utf-8",
    )
    return run_dir
