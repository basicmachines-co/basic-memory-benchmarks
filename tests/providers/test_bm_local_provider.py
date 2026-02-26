from basic_memory_benchmarks.providers.bm_local import BasicMemoryLocalProvider


def test_extract_existing_project_name_from_bm_error() -> None:
    message = (
        "Error adding project: Cannot create project at '/tmp/docs': "
        "path is nested within existing project 'bm-bench-abc123' at '/tmp/docs'."
    )
    assert BasicMemoryLocalProvider._extract_existing_project_name(message) == "bm-bench-abc123"


def test_extract_existing_project_name_none_without_match() -> None:
    message = "Error adding project: unknown failure"
    assert BasicMemoryLocalProvider._extract_existing_project_name(message) is None


def test_extract_existing_project_name_from_wrapped_bm_error() -> None:
    message = (
        "Error adding project: path is nested within existing project \n"
        "'bm-bench-wrap999' at '/tmp/docs'."
    )
    assert BasicMemoryLocalProvider._extract_existing_project_name(message) == "bm-bench-wrap999"
