# Concurrent Embedding Evaluation — Final Benchmark Data

This directory contains the final, completed CSV results underlying the
Concurrent Embedding Evaluation section of the report for
`ConcurrentEmbeddingEngine`. All files were produced by
`benchmarks/embedding_concurrency_benchmark.py` and are committed here
unmodified, exactly as generated. Across the four experiments below, the
completed run executed a total of 508,800 embedding requests.

Each experiment has a `_trials.csv` file (one row per individual repeated
trial) and a `_summary.csv` file (one row per configuration, aggregating
the trials with mean/std/95% CI, plus `speedup_vs_baseline`,
`pvalue_vs_baseline`, and `significant_vs_baseline` computed against the
`original` scenario measured in the same experiment run).

## Files

- **`exp1_concurrency_sweep_warm_trials.csv` / `exp1_concurrency_sweep_warm_summary.csv`**
  Experiment 1: synthetic CPU-bound engine, concurrency swept over
  {1, 2, 4, 8, 16, 32} at a fixed duplicate ratio of 0.5, with the
  thread/process pool created once and reused across trials (warm pool,
  steady-state).

- **`exp1_concurrency_sweep_cold_trials.csv` / `exp1_concurrency_sweep_cold_summary.csv`**
  Same sweep as above, but with the thread/process pool constructed fresh
  for every trial (cold pool), isolating pool start-up/tear-down overhead.

- **`exp2_duplicate_ratio_sweep_trials.csv` / `exp2_duplicate_ratio_sweep_summary.csv`**
  Experiment 2: synthetic CPU-bound engine, concurrency fixed at 16,
  duplicate ratio swept over {0.0, 0.2, 0.5, 0.8}, warm pool. Isolates the
  effect of in-flight request overlap/duplication on batching and
  deduplication.

- **`exp3_real_model_warm_trials.csv` / `exp3_real_model_warm_summary.csv`**
  Experiment 3: real embedding model
  (`sentence-transformers/all-MiniLM-L6-v2` via `LangChainEmbeddingEngine`),
  concurrency swept over {1, 2, 4, 8, 16, 32} at a fixed duplicate ratio of
  0.5, warm pool. Validates whether the synthetic-benchmark findings
  transfer to a real model.

## Excluded from this directory

Smoke-test runs, the superseded ~48-request preliminary benchmark, and an
incomplete/failed Experiment 3 cold-pool run are intentionally not
included here, as they are not part of the final, completed benchmark.
