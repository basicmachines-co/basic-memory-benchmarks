# basic-memory-benchmarks command runner

bench-smoke:
    uv run bm-bench run retrieval \
      --dataset-id synthetic \
      --dataset-path benchmarks/synthetic/queries.json \
      --corpus-dir benchmarks/synthetic/docs \
      --queries-path benchmarks/synthetic/queries.json \
      --providers bm-local,mem0-local \
      --allow-provider-skip

bench-fetch-locomo:
    uv run bm-bench datasets fetch --dataset locomo

bench-convert-locomo:
    uv run bm-bench convert locomo

bench-run-bm-local:
    uv run bm-bench run retrieval --providers bm-local

bench-run-mem0-local:
    uv run bm-bench run retrieval --providers mem0-local --allow-provider-skip

bench-run-full:
    uv run bm-bench run full --providers bm-local,mem0-local --allow-provider-skip

bench-judge RUN_DIR:
    uv run bm-bench run judge --run-dir {{RUN_DIR}}

bench-publish RUN_DIR:
    uv run bm-bench publish --run-dir {{RUN_DIR}}
