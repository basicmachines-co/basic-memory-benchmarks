"""Basic Memory local provider via external bm CLI contract."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

from basic_memory_benchmarks.models import RunConfig, SearchHit
from basic_memory_benchmarks.providers.base import BenchmarkProvider
from basic_memory_benchmarks.utils import run_command


class BasicMemoryLocalProvider(BenchmarkProvider):
    name = "bm-local"

    def _project_name(self, run_config: RunConfig) -> str:
        return f"bm-bench-{run_config.run_id}"

    def ingest(self, corpus_path: Path, run_config: RunConfig) -> None:
        project_name = self._project_name(run_config)

        # Trigger: benchmark corpus needs indexing in BM
        # Why: keep provider path external to BM internals
        # Outcome: project exists and search index is up to date
        add_args = ["bm", "project", "add", project_name, str(corpus_path)]
        try:
            run_command(add_args)
        except subprocess.CalledProcessError as exc:
            merged = (exc.stdout or "") + "\n" + (exc.stderr or "")
            if "already exists" not in merged.lower():
                raise

        try:
            run_command(["bm", "reindex", "--search", "--embeddings", "-p", project_name])
        except subprocess.CalledProcessError:
            run_command(["bm", "reindex", "--search", "-p", project_name])

    @staticmethod
    def _doc_id_from_item(item: dict) -> str | None:
        raw = item.get("source_doc_id") or item.get("permalink") or item.get("file_path")
        if not raw:
            return None
        name = str(raw).rstrip("/").split("/")[-1]
        if name.endswith(".md"):
            name = name[:-3]
        return name

    def search(self, query: str, limit: int, run_config: RunConfig) -> list[SearchHit]:
        project_name = self._project_name(run_config)
        args = [
            "bm",
            "tool",
            "search-notes",
            query,
            "--project",
            project_name,
            "--page-size",
            str(limit),
            "--hybrid",
            "--local",
        ]

        try:
            completed = run_command(args)
        except subprocess.CalledProcessError:
            completed = run_command(
                [
                    "bm",
                    "tool",
                    "search-notes",
                    query,
                    "--project",
                    project_name,
                    "--page-size",
                    str(limit),
                    "--local",
                ]
            )

        payload = json.loads(completed.stdout.strip() or "{}")
        rows = payload.get("results") if isinstance(payload, dict) else []
        hits: list[SearchHit] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            metadata_raw = row.get("metadata")
            metadata: dict[str, Any]
            if isinstance(metadata_raw, dict):
                metadata = cast(dict[str, Any], metadata_raw)
            else:
                metadata = {}
            hits.append(
                SearchHit(
                    id=str(row.get("entity_id") or row.get("observation_id") or row.get("relation_id") or ""),
                    source_doc_id=self._doc_id_from_item(row),
                    source_path=row.get("file_path") or row.get("permalink"),
                    text=row.get("matched_chunk") or row.get("content"),
                    score=float(row.get("score", 0.0) or 0.0),
                    metadata=metadata,
                )
            )
        return hits

    def cleanup(self, run_config: RunConfig) -> None:
        _ = run_config
        # Intentionally no-op to avoid deleting user configuration accidentally.

    def version_info(self) -> dict[str, str]:
        try:
            result = run_command(["bm", "--version"])
            return {"bm_version": result.stdout.strip()}
        except Exception:
            return {}
