from basic_memory_benchmarks.providers.mem0_local import Mem0LocalProvider


def test_mem0_normalization_prefers_nested_metadata() -> None:
    item = {
        "id": "abc",
        "memory": "hello",
        "score": 0.42,
        "metadata": {
            "source_doc_id": "doc-1",
            "source_path": "docs/doc-1.md",
        },
    }
    hit = Mem0LocalProvider._normalize_item(item)
    assert hit.source_doc_id == "doc-1"
    assert hit.source_path == "docs/doc-1.md"
    assert hit.score == 0.42
