"""Provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from basic_memory_benchmarks.models import RunConfig, SearchHit


class BenchmarkProvider(ABC):
    name: str

    @abstractmethod
    def ingest(self, corpus_path: Path, run_config: RunConfig) -> None:
        """Ingest corpus into provider-specific memory store."""

    @abstractmethod
    def search(self, query: str, limit: int, run_config: RunConfig) -> list[SearchHit]:
        """Search for relevant memories and return normalized hits."""

    @abstractmethod
    def cleanup(self, run_config: RunConfig) -> None:
        """Clean up any provider-scoped state."""

    def version_info(self) -> dict[str, str]:
        return {}
