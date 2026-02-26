"""Mem0 local provider using the mem0ai package."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import frontmatter

from basic_memory_benchmarks.exceptions import ProviderSkippedError
from basic_memory_benchmarks.models import RunConfig, SearchHit
from basic_memory_benchmarks.providers.base import BenchmarkProvider


class Mem0LocalProvider(BenchmarkProvider):
    name = "mem0-local"

    def __init__(self) -> None:
        self._memory = None

    def _user_id(self, run_config: RunConfig) -> str:
        return f"bm-bench-{run_config.run_id}-mem0"

    def _ensure_memory(self):
        if self._memory is not None:
            return self._memory
        if not os.getenv("OPENAI_API_KEY"):
            raise ProviderSkippedError("OPENAI_API_KEY missing for mem0-local provider")

        from mem0 import Memory  # Deferred import to keep startup lightweight

        self._memory = Memory()
        return self._memory

    @staticmethod
    def _doc_id_from_path(path: Path) -> str:
        if path.suffix == ".md":
            return path.name[:-3]
        return path.name

    @staticmethod
    def _conversation_id(path: Path) -> str:
        parts = list(path.parts)
        for part in parts:
            if part.startswith("locomo-c"):
                return part
        return "unknown"

    def ingest(self, corpus_path: Path, run_config: RunConfig) -> None:
        memory = self._ensure_memory()
        user_id = self._user_id(run_config)

        for note_path in sorted(corpus_path.rglob("*.md")):
            rel_path = note_path.relative_to(corpus_path).as_posix()
            with note_path.open("r", encoding="utf-8") as handle:
                parsed = frontmatter.load(handle)
            doc_id = str(parsed.get("source_doc_id") or self._doc_id_from_path(note_path))
            conversation_id = str(parsed.get("conversation_id") or self._conversation_id(note_path))
            metadata = {
                "source_doc_id": doc_id,
                "source_path": rel_path,
                "conversation_id": conversation_id,
                "dataset_id": run_config.dataset_id,
            }
            memory.add(parsed.content, user_id=user_id, metadata=metadata, infer=False)

    @staticmethod
    def _normalize_item(item: dict) -> SearchHit:
        metadata_raw = item.get("metadata")
        metadata: dict[str, Any]
        if isinstance(metadata_raw, dict):
            metadata = cast(dict[str, Any], metadata_raw)
        else:
            metadata = {}
        source_doc_id = item.get("source_doc_id") or metadata.get("source_doc_id")
        source_path = item.get("source_path") or metadata.get("source_path")
        score_raw = item.get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None

        return SearchHit(
            id=str(item.get("id") or ""),
            source_doc_id=source_doc_id,
            source_path=source_path,
            text=item.get("memory") or item.get("text"),
            score=score,
            metadata=metadata,
        )

    def search(self, query: str, limit: int, run_config: RunConfig) -> list[SearchHit]:
        memory = self._ensure_memory()
        user_id = self._user_id(run_config)
        payload = memory.search(query=query, user_id=user_id, limit=limit)
        rows = payload.get("results") if isinstance(payload, dict) else []

        hits: list[SearchHit] = []
        for item in rows or []:
            if isinstance(item, dict):
                hits.append(self._normalize_item(item))
        return hits

    def cleanup(self, run_config: RunConfig) -> None:
        if self._memory is None:
            return
        try:
            self._memory.delete_all(user_id=self._user_id(run_config))
        except Exception:
            # Cleanup should never break the main benchmark flow.
            return

    def version_info(self) -> dict[str, str]:
        try:
            return {"mem0ai": version("mem0ai")}
        except Exception:
            return {}
