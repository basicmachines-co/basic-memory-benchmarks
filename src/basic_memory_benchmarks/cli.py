"""CLI entrypoint for benchmark operations."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import typer
from rich.console import Console

from basic_memory_benchmarks.converters.locomo_to_corpus import convert_locomo_to_corpus
from basic_memory_benchmarks.datasets.locomo import LOCOMO_URL, fetch_locomo_dataset
from basic_memory_benchmarks.models import DatasetProvenance, RunConfig
from basic_memory_benchmarks.reporting.compare import compare_provider_metric, load_retrieval_summary
from basic_memory_benchmarks.runner import run_judge, run_qa_stage, run_retrieval
from basic_memory_benchmarks.utils import sha256_file

app = typer.Typer(help="Basic Memory benchmark suite")
console = Console()

datasets_app = typer.Typer(help="Dataset management commands")
convert_app = typer.Typer(help="Dataset conversion commands")
run_app = typer.Typer(help="Benchmark execution commands")

app.add_typer(datasets_app, name="datasets")
app.add_typer(convert_app, name="convert")
app.add_typer(run_app, name="run")


@datasets_app.command("fetch")
def datasets_fetch(
    dataset: str = typer.Option("locomo", "--dataset"),
    output: Path = typer.Option(Path("benchmarks/datasets/locomo/locomo10.json"), "--output"),
    url: str = typer.Option(LOCOMO_URL, "--url"),
) -> None:
    if dataset != "locomo":
        raise typer.BadParameter("Only locomo is supported in v1")

    provenance = fetch_locomo_dataset(output_path=output, url=url)
    console.print(f"Downloaded {dataset} to [cyan]{output}[/cyan]")
    console.print(f"SHA256: [green]{provenance.checksum_sha256}[/green]")


@convert_app.command("locomo")
def convert_locomo(
    dataset_path: Path = typer.Option(Path("benchmarks/datasets/locomo/locomo10.json"), "--dataset-path"),
    output_dir: Path = typer.Option(Path("benchmarks/generated/locomo"), "--output-dir"),
    max_conversations: int | None = typer.Option(None, "--max-conversations"),
) -> None:
    docs_dir, queries_path, doc_count, query_count = convert_locomo_to_corpus(
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_conversations=max_conversations,
    )
    console.print(f"Docs: [cyan]{docs_dir}[/cyan] ({doc_count})")
    console.print(f"Queries: [cyan]{queries_path}[/cyan] ({query_count})")


@run_app.command("retrieval")
def run_retrieval_command(
    providers: str = typer.Option("bm-local,mem0-local", "--providers"),
    dataset_id: str = typer.Option("locomo", "--dataset-id"),
    dataset_path: Path = typer.Option(Path("benchmarks/datasets/locomo/locomo10.json"), "--dataset-path"),
    corpus_dir: Path = typer.Option(Path("benchmarks/generated/locomo/docs"), "--corpus-dir"),
    queries_path: Path = typer.Option(Path("benchmarks/generated/locomo/queries.json"), "--queries-path"),
    output_root: Path = typer.Option(Path("benchmarks/runs"), "--output-root"),
    run_id: str | None = typer.Option(None, "--run-id"),
    top_k: int = typer.Option(10, "--top-k"),
    bm_source: str = typer.Option(
        "github:basicmachines-co/basic-memory@main",
        "--bm-source",
    ),
    bm_local_path: str | None = typer.Option(None, "--bm-local-path"),
    allow_provider_skip: bool = typer.Option(True, "--allow-provider-skip/--strict-providers"),
) -> None:
    resolved_run_id = run_id or uuid.uuid4().hex[:12]
    provider_list = [item.strip() for item in providers.split(",") if item.strip()]

    if not dataset_path.exists():
        raise typer.BadParameter(f"Dataset path not found: {dataset_path}")
    if not corpus_dir.exists():
        raise typer.BadParameter(f"Corpus dir not found: {corpus_dir}")
    if not queries_path.exists():
        raise typer.BadParameter(f"Queries file not found: {queries_path}")

    provenance = DatasetProvenance(
        dataset_id=dataset_id,
        source_url=str(dataset_path),
        checksum_sha256=sha256_file(dataset_path),
        license_note="See dataset source/license terms",
        fetched_at_utc="unknown",
    )

    config = RunConfig(
        run_id=resolved_run_id,
        dataset_id=dataset_id,
        dataset_path=str(dataset_path),
        corpus_dir=str(corpus_dir),
        queries_path=str(queries_path),
        output_root=str(output_root),
        providers=provider_list,
        top_k=top_k,
        bm_source=bm_source,
        bm_local_path=bm_local_path,
        allow_provider_skip=allow_provider_skip,
    )

    run_dir = run_retrieval(run_config=config, dataset=provenance)
    console.print(f"Retrieval run complete: [green]{run_dir}[/green]")


@run_app.command("qa")
def run_qa_command(
    run_dir: Path = typer.Option(..., "--run-dir"),
    answerer: str = typer.Option(
        "claude:claude-haiku-4-5",
        "--answerer",
        help="Runner spec: claude:<model> or openai-compat:<model>@<base_url>",
    ),
    judge: str = typer.Option(
        "claude:claude-sonnet-4-6",
        "--judge",
        help="Runner spec: claude:<model> or openai-compat:<model>@<base_url>",
    ),
    max_workers: int = typer.Option(4, "--max-workers"),
) -> None:
    out = run_qa_stage(
        run_dir=run_dir,
        answerer_spec=answerer,
        judge_spec=judge,
        max_workers=max_workers,
    )
    console.print(f"QA run complete: [green]{out}[/green]")
    console.print(f"See [cyan]{out / 'qa-summary.json'}[/cyan]")


@run_app.command("judge")
def run_judge_command(
    run_dir: Path = typer.Option(..., "--run-dir"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
) -> None:
    out = run_judge(run_dir=run_dir, model=model)
    console.print(f"Judge run complete: [green]{out}[/green]")


@run_app.command("full")
def run_full_command(
    providers: str = typer.Option("bm-local,mem0-local", "--providers"),
    dataset_id: str = typer.Option("locomo", "--dataset-id"),
    dataset_path: Path = typer.Option(Path("benchmarks/datasets/locomo/locomo10.json"), "--dataset-path"),
    corpus_dir: Path = typer.Option(Path("benchmarks/generated/locomo/docs"), "--corpus-dir"),
    queries_path: Path = typer.Option(Path("benchmarks/generated/locomo/queries.json"), "--queries-path"),
    output_root: Path = typer.Option(Path("benchmarks/runs"), "--output-root"),
    run_id: str | None = typer.Option(None, "--run-id"),
    top_k: int = typer.Option(10, "--top-k"),
    bm_source: str = typer.Option("github:basicmachines-co/basic-memory@main", "--bm-source"),
    bm_local_path: str | None = typer.Option(None, "--bm-local-path"),
    allow_provider_skip: bool = typer.Option(True, "--allow-provider-skip/--strict-providers"),
    judge: bool = typer.Option(False, "--judge"),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model"),
) -> None:
    run_retrieval_command(
        providers=providers,
        dataset_id=dataset_id,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        queries_path=queries_path,
        output_root=output_root,
        run_id=run_id,
        top_k=top_k,
        bm_source=bm_source,
        bm_local_path=bm_local_path,
        allow_provider_skip=allow_provider_skip,
    )

    if judge:
        resolved_run_id = run_id
        if resolved_run_id is None:
            # run_retrieval_command generated uuid when run_id is None. infer by latest dir.
            run_dirs = sorted(Path(output_root).glob("*"), key=lambda path: path.stat().st_mtime)
            if not run_dirs:
                raise RuntimeError("Unable to locate run directory for judge step")
            run_dir = run_dirs[-1]
        else:
            run_dir = Path(output_root) / resolved_run_id
        run_judge_command(run_dir=run_dir, model=judge_model)


@app.command("compare")
def compare_runs(
    baseline: Path = typer.Argument(..., help="Path to baseline retrieval-summary.json"),
    candidate: Path = typer.Argument(..., help="Path to candidate retrieval-summary.json"),
    provider: str = typer.Option("bm-local", "--provider"),
    metric: str = typer.Option("recall_at_5", "--metric"),
) -> None:
    baseline_payload = load_retrieval_summary(baseline)
    candidate_payload = load_retrieval_summary(candidate)
    b, c, delta = compare_provider_metric(baseline_payload, candidate_payload, provider, metric)
    console.print(f"provider={provider} metric={metric}")
    console.print(f"baseline={b}")
    console.print(f"candidate={c}")
    console.print(f"delta={delta}")


@app.command("publish")
def publish_run(
    run_dir: Path = typer.Option(..., "--run-dir"),
    destination: Path = typer.Option(Path("benchmarks/results/public"), "--destination"),
) -> None:
    if not run_dir.exists():
        raise typer.BadParameter(f"Run directory does not exist: {run_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / run_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir, target)
    console.print(f"Published run to [green]{target}[/green]")


@app.command("validate-artifacts")
def validate_artifacts(
    run_dir: Path = typer.Option(..., "--run-dir"),
) -> None:
    expected = [
        "manifest.json",
        "provider-status.json",
        "per-query-retrieval.jsonl",
        "retrieval-summary.json",
        "summary.md",
    ]
    missing = [name for name in expected if not (run_dir / name).exists()]
    if missing:
        raise typer.BadParameter(f"Missing artifacts: {missing}")
    console.print("Artifacts look complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
