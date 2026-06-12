"""Tests for grouped (per-question corpus) retrieval execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from basic_memory_benchmarks.exceptions import ProviderSkippedError
from basic_memory_benchmarks.models import DatasetProvenance, RunConfig, SearchHit
from basic_memory_benchmarks.providers.base import BenchmarkProvider
from basic_memory_benchmarks.runner import run_retrieval


class RecordingProvider(BenchmarkProvider):
    """Records lifecycle calls; class-level log shared across instances."""

    name = "recording"
    calls: list[tuple[str, str, str]] = []  # (event, corpus_or_query, run_id)
    fail_groups: set[str] = set()
    skip_all = False
    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1

    def ingest(self, corpus_path: Path, run_config: RunConfig) -> None:
        if type(self).skip_all:
            raise ProviderSkippedError("creds missing")
        for group_id in type(self).fail_groups:
            if f"-{group_id}" in run_config.run_id:
                raise RuntimeError(f"boom in {group_id}")
        type(self).calls.append(("ingest", str(corpus_path), run_config.run_id))

    def search(self, query: str, limit: int, run_config: RunConfig) -> list[SearchHit]:
        type(self).calls.append(("search", query, run_config.run_id))
        return [SearchHit(source_doc_id="doc-1", text="ctx", score=1.0)]

    def cleanup(self, run_config: RunConfig) -> None:
        type(self).calls.append(("cleanup", "", run_config.run_id))

    def version_info(self) -> dict[str, str]:
        return {"recording": "1.0"}


@pytest.fixture(autouse=True)
def _reset_recording_provider():
    RecordingProvider.calls = []
    RecordingProvider.fail_groups = set()
    RecordingProvider.skip_all = False
    RecordingProvider.instances = 0


def _setup_grouped_corpus(tmp_path: Path, groups: list[str]) -> tuple[Path, Path]:
    corpus_root = tmp_path / "groups"
    queries = []
    for group_id in groups:
        docs = corpus_root / group_id / "docs"
        docs.mkdir(parents=True)
        (docs / f"{group_id}-s000.md").write_text(f"# {group_id}\n", encoding="utf-8")
        queries.append(
            {
                "id": group_id,
                "query": f"question for {group_id}",
                "category": "single-session-user",
                "group": group_id,
                "ground_truth": ["doc-1"],
                "expected_answer": "answer",
                "metadata": {"question_date": "2023/05/30"},
            }
        )
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(queries), encoding="utf-8")
    return corpus_root, queries_path


def _run_config(tmp_path: Path, corpus_root: Path, queries_path: Path) -> RunConfig:
    return RunConfig(
        run_id="testrun",
        dataset_id="longmemeval_s",
        dataset_path=str(queries_path),
        corpus_dir=str(corpus_root),
        queries_path=str(queries_path),
        output_root=str(tmp_path / "runs"),
        providers=["recording"],
    )


def _provenance() -> DatasetProvenance:
    return DatasetProvenance(
        dataset_id="longmemeval_s",
        source_url="test",
        checksum_sha256="0" * 64,
        license_note="test",
        fetched_at_utc="now",
    )


class TestGroupedExecution:
    def test_each_group_isolated(self, tmp_path):
        corpus_root, queries_path = _setup_grouped_corpus(tmp_path, ["qa", "qb"])
        config = _run_config(tmp_path, corpus_root, queries_path)

        run_dir = run_retrieval(
            run_config=config,
            dataset=_provenance(),
            provider_factory=lambda name: RecordingProvider(),
        )

        ingests = [c for c in RecordingProvider.calls if c[0] == "ingest"]
        assert len(ingests) == 2
        # Group-suffixed run ids isolate provider namespaces.
        assert {run_id for _, _, run_id in ingests} == {"testrun-qa", "testrun-qb"}
        # Each ingest points at its own group corpus.
        assert {Path(corpus).parent.name for _, corpus, _ in ingests} == {"qa", "qb"}
        # Fresh provider instance per group (plus one flat instance is never made).
        assert RecordingProvider.instances == 2

        rows = [
            json.loads(line)
            for line in (run_dir / "per-query-retrieval.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
        assert {row["query_id"] for row in rows} == {"qa", "qb"}
        # Metadata flows into retrieval rows for the QA stage.
        assert all(row["metadata"]["question_date"] == "2023/05/30" for row in rows)

        status = json.loads((run_dir / "provider-status.json").read_text())
        provider_meta = status[0]["metadata"]
        assert provider_meta["grouped_mode"] == "true"
        assert provider_meta["group_count"] == "2"

    def test_failed_group_recorded_and_run_continues(self, tmp_path):
        corpus_root, queries_path = _setup_grouped_corpus(tmp_path, ["qa", "qb", "qc"])
        RecordingProvider.fail_groups = {"qb"}
        config = _run_config(tmp_path, corpus_root, queries_path)

        run_dir = run_retrieval(
            run_config=config,
            dataset=_provenance(),
            provider_factory=lambda name: RecordingProvider(),
        )

        rows = [
            json.loads(line)
            for line in (run_dir / "per-query-retrieval.jsonl").read_text().splitlines()
        ]
        assert {row["query_id"] for row in rows} == {"qa", "qc"}

        status = json.loads((run_dir / "provider-status.json").read_text())
        provider_meta = status[0]["metadata"]
        assert provider_meta["failed_group_count"] == "1"
        assert provider_meta["failed_groups"] == "qb"

    def test_skip_on_first_group_skips_provider(self, tmp_path):
        corpus_root, queries_path = _setup_grouped_corpus(tmp_path, ["qa", "qb"])
        RecordingProvider.skip_all = True
        config = _run_config(tmp_path, corpus_root, queries_path)

        run_dir = run_retrieval(
            run_config=config,
            dataset=_provenance(),
            provider_factory=lambda name: RecordingProvider(),
        )

        status = json.loads((run_dir / "provider-status.json").read_text())
        assert status[0]["state"] == "skipped"
        # Only the first group was attempted.
        assert RecordingProvider.instances == 1

    def test_missing_group_corpus_raises(self, tmp_path):
        corpus_root, queries_path = _setup_grouped_corpus(tmp_path, ["qa"])
        queries = json.loads(queries_path.read_text())
        queries.append({**queries[0], "id": "missing", "group": "missing"})
        queries_path.write_text(json.dumps(queries), encoding="utf-8")
        config = _run_config(tmp_path, corpus_root, queries_path)
        config = config.model_copy(update={"allow_provider_skip": False})

        with pytest.raises(FileNotFoundError, match="Missing group corpus"):
            run_retrieval(
                run_config=config,
                dataset=_provenance(),
                provider_factory=lambda name: RecordingProvider(),
            )

    def test_mixed_grouped_and_ungrouped_rejected(self, tmp_path):
        corpus_root, queries_path = _setup_grouped_corpus(tmp_path, ["qa"])
        queries = json.loads(queries_path.read_text())
        queries.append(
            {
                "id": "flat",
                "query": "ungrouped question",
                "category": "single_hop",
                "ground_truth": [],
            }
        )
        queries_path.write_text(json.dumps(queries), encoding="utf-8")
        config = _run_config(tmp_path, corpus_root, queries_path)
        config = config.model_copy(update={"allow_provider_skip": False})

        with pytest.raises(ValueError, match="mixed grouped/ungrouped"):
            run_retrieval(
                run_config=config,
                dataset=_provenance(),
                provider_factory=lambda name: RecordingProvider(),
            )
