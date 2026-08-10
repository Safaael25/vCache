"""
Rigorous benchmark: original synchronous embedding path vs. `ConcurrentEmbeddingEngine`
(THREAD / PROCESS modes) under concurrent load.

METHODOLOGY VERSION 2. This supersedes the original single-sample, 48-request
benchmark used during initial development (see
`benchmarks/embedding_concurrency_results_v1_preliminary.md` for that
preliminary run, explicitly retired - do not cite its headline numbers).
This version addresses every gap identified in the review of v1:

  - Workload size scales with concurrency: `max(2000, concurrency * 100)`
    requests per configuration, instead of a fixed 48.
  - Every configuration is repeated (default 5x) and reported as
    mean / standard deviation / 95% confidence interval (Student's t,
    appropriate for small samples), not a single wall-clock sample.
  - An untimed warm-up phase precedes every timed measurement.
  - Real prompts, sampled from `playground/sample_data.parquet` (1000 rows,
    realistic length variance - 25 to ~13,800 characters), are used instead
    of 6 fixed 18-character synthetic strings.
  - A real embedding model (`LangChainEmbeddingEngine` /
    `sentence-transformers/all-MiniLM-L6-v2`) is exercised in a dedicated
    experiment, not just the synthetic CPU-bound stand-in.
  - Duplicate ratio (0% / 20% / 50% / 80%) is an explicit, controlled
    variable, not an incidental byproduct of a fixed 6-unique/48-request
    workload.
  - Every scenario is instrumented (via `InstrumentedEmbeddingEngine`, which
    is multiprocessing-safe) to report the number of underlying embedding
    computations actually performed, the number of requests satisfied via
    deduplication, and the batch sizes ConcurrentEmbeddingEngine formed -
    not just wall-clock timing.
  - A Welch's t-test against the baseline's per-trial timings is reported
    per configuration, so speedup claims are labeled as statistically
    significant (p < 0.05) or not, rather than asserted from a single run.

This still does NOT run through the full vCache pipeline (LLM calls, vector
DB, Bayesian policy) - it isolates the embedding step in isolation, since
that is the component `ConcurrentEmbeddingEngine` optimizes.

Three experiments, run independently (not a full cross-product, which would
take many hours) - see module docstrings on each `run_experiment_*` function:
  1. Concurrency sweep (1,2,4,8,16,32) at a fixed 50% duplicate ratio.
  2. Duplicate-ratio sweep (0/20/50/80%) at a fixed concurrency=16.
  3. Real-model validation (LangChainEmbeddingEngine / all-MiniLM-L6-v2).

Run:
    python benchmarks/embedding_concurrency_benchmark.py --experiment concurrency
    python benchmarks/embedding_concurrency_benchmark.py --experiment duplicate_ratio
    python benchmarks/embedding_concurrency_benchmark.py --experiment real_model
    python benchmarks/embedding_concurrency_benchmark.py --experiment all
"""

import argparse
import csv
import functools
import hashlib
import json
import math
import multiprocessing
import os
import queue as queue_module
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from vcache.vcache_core.cache.embedding_engine.concurrent_embedding_engine import (
    ConcurrentEmbeddingEngine,
    EmbeddingExecutionMode,
)
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine

try:
    import psutil
except ImportError:
    psutil = None

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

RESULTS_DIR = Path(__file__).parent / "results" / "embedding_concurrency_v2"
PLAYGROUND_PARQUET = Path(__file__).parent.parent / "playground" / "sample_data.parquet"

_EMBEDDING_DIM = 384
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
DUPLICATE_RATIOS = [0.0, 0.2, 0.5, 0.8]


# --------------------------------------------------------------------------
# Synthetic CPU-bound embedding engine (calibrated to ~4-5ms/call so that
# 2000-3200-request x 5-repeat sweeps stay tractable, while remaining
# genuinely GIL-holding CPU work, not I/O).
# --------------------------------------------------------------------------


def _cpu_bound_fake_embedding(text: str, work_units: int) -> List[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    x = seed
    acc = 0
    for _ in range(work_units):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        acc += x % 1000
    return [((acc + i) % 997) / 997.0 for i in range(_EMBEDDING_DIM)]


class SimulatedLocalEmbeddingEngine(EmbeddingEngine):
    """Deterministic, picklable, CPU-bound stand-in for a local embedding model."""

    def __init__(self, work_units: int = 30_000):
        self.work_units = work_units

    def get_embedding(self, text: str) -> List[float]:
        return _cpu_bound_fake_embedding(text, self.work_units)

    def get_engine_factory(self) -> Callable[[], "EmbeddingEngine"]:
        return functools.partial(
            SimulatedLocalEmbeddingEngine, work_units=self.work_units
        )


# --------------------------------------------------------------------------
# Real-prompt workload construction
# --------------------------------------------------------------------------


def load_real_prompt_pool(seed: int = 42) -> List[str]:
    """Loads and shuffles the 'text' column of playground/sample_data.parquet."""
    import pandas as pd

    df = pd.read_parquet(PLAYGROUND_PARQUET)
    texts = df["text"].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts


def make_workload(
    num_requests: int,
    duplicate_ratio: float,
    prompt_pool: List[str],
    seed: int = 0,
) -> Tuple[List[str], int]:
    """
    Builds a request list of length `num_requests` where approximately
    `duplicate_ratio` of requests are repeats of an earlier prompt in the
    same list.

    num_unique = max(1, round(num_requests * (1 - duplicate_ratio))). The
    unique prompts are drawn from `prompt_pool` (real text from
    sample_data.parquet); if more unique prompts are needed than the pool
    provides (1000 rows), the pool is cycled with a short disambiguating
    suffix appended so requested uniqueness is still honored while every
    prompt's core content and length remain realistic.

    The resulting list is shuffled (not left as clean round-robin blocks) so
    that duplicate arrivals are genuinely interleaved - this matters because
    ConcurrentEmbeddingEngine can only deduplicate/batch requests that
    arrive close together in time, not ones separated by many distinct
    intervening requests.

    Returns (workload, num_unique_actually_used).
    """
    num_unique = max(1, round(num_requests * (1 - duplicate_ratio)))
    pool_size = len(prompt_pool)
    unique_texts = []
    for i in range(num_unique):
        base = prompt_pool[i % pool_size]
        cycle = i // pool_size
        unique_texts.append(base if cycle == 0 else f"{base}\n[variant {cycle}]")

    workload = [unique_texts[i % num_unique] for i in range(num_requests)]
    random.Random(seed).shuffle(workload)
    return workload, num_unique


# --------------------------------------------------------------------------
# Instrumentation: records actual underlying computations + batch sizes,
# even across ProcessPoolExecutor worker processes (via a multiprocessing
# Queue, since worker processes don't share the parent's in-memory counters).
# --------------------------------------------------------------------------


class InstrumentedEmbeddingEngine(EmbeddingEngine):
    """
    Wraps an inner engine and records, via a multiprocessing-safe Queue, the
    size of every batch actually computed - i.e. every call to
    `get_embeddings()`. This is the single choke point
    `ConcurrentEmbeddingEngine` dispatches through in both THREAD and
    PROCESS mode (see `_dispatch_batch` in concurrent_embedding_engine.py),
    so counting there gives an exact measurement of underlying computations
    regardless of mode, including inside spawned worker processes.

    `get_engine_factory` deliberately delegates to the wrapped inner
    engine's own factory (not to the `inner_factory` this instance was built
    from), so the picklable callable shipped to a spawned worker process is
    always built from simple, importable pieces (a class + primitive kwargs)
    regardless of how the enclosing benchmark code originally constructed
    this instance.
    """

    def __init__(self, inner_factory: Callable[[], EmbeddingEngine], count_queue):
        self._inner = inner_factory()
        self._count_queue = count_queue

    def get_embedding(self, text: str) -> List[float]:
        result = self._inner.get_embedding(text)
        self._count_queue.put(1)
        return result

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = self._inner.get_embeddings(texts)
        self._count_queue.put(len(texts))
        return results

    def get_engine_factory(self) -> Optional[Callable[[], EmbeddingEngine]]:
        inner_worker_factory = self._inner.get_engine_factory()
        if inner_worker_factory is None:
            return None
        return functools.partial(
            InstrumentedEmbeddingEngine,
            inner_factory=inner_worker_factory,
            count_queue=self._count_queue,
        )


def _drain_counts(q) -> List[int]:
    batch_sizes = []
    while True:
        try:
            batch_sizes.append(q.get_nowait())
        except queue_module.Empty:
            break
    return batch_sizes


def build_engine_and_queue(
    base_engine_factory: Callable[[], EmbeddingEngine],
    mode: EmbeddingExecutionMode,
    num_workers: int,
    max_batch_size: int,
) -> Tuple[ConcurrentEmbeddingEngine, "multiprocessing.Queue"]:
    """Builds an instrumented ConcurrentEmbeddingEngine + its count queue."""
    q = multiprocessing.Queue()
    instrumented = InstrumentedEmbeddingEngine(
        inner_factory=base_engine_factory, count_queue=q
    )
    engine = ConcurrentEmbeddingEngine(
        engine=instrumented, mode=mode, num_workers=num_workers,
        max_batch_size=max_batch_size,
    )
    return engine, q


# --------------------------------------------------------------------------
# Resource sampling (CPU% / RSS memory), fixed to reuse psutil.Process
# objects across samples (a fresh Process() always reports 0.0% CPU on its
# first call, since cpu_percent() reports a delta since that object's last
# call - this bug was present and fixed during v1 validation).
# --------------------------------------------------------------------------


class _ResourceSampler:
    def __init__(
        self,
        extra_pids_fn: Optional[Callable[[], List[int]]] = None,
        interval_s: float = 0.08,
    ):
        self._extra_pids_fn = extra_pids_fn or (lambda: [])
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu_samples: List[float] = []
        self._peak_rss_bytes = 0
        self._processes: Dict[int, "psutil.Process"] = {}

    def _refresh_tracked_processes(self) -> None:
        if os.getpid() not in self._processes:
            self._processes[os.getpid()] = psutil.Process(os.getpid())
        for pid in self._extra_pids_fn():
            if pid not in self._processes:
                try:
                    proc = psutil.Process(pid)
                    proc.cpu_percent(interval=None)
                    self._processes[pid] = proc
                except psutil.Error:
                    continue

    def start(self) -> None:
        if psutil is None:
            return
        self._refresh_tracked_processes()
        for proc in self._processes.values():
            try:
                proc.cpu_percent(interval=None)
            except psutil.Error:
                continue
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            self._refresh_tracked_processes()
            total_cpu = 0.0
            total_rss = 0
            dead_pids = []
            for pid, proc in self._processes.items():
                try:
                    total_cpu += proc.cpu_percent(interval=None)
                    total_rss += proc.memory_info().rss
                except psutil.Error:
                    dead_pids.append(pid)
            for pid in dead_pids:
                del self._processes[pid]
            self._cpu_samples.append(total_cpu)
            self._peak_rss_bytes = max(self._peak_rss_bytes, total_rss)

    def stop(self):
        if psutil is None:
            return None, None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        avg_cpu = statistics.mean(self._cpu_samples) if self._cpu_samples else None
        peak_mb = self._peak_rss_bytes / (1024 * 1024) if self._peak_rss_bytes else None
        return avg_cpu, peak_mb


def _process_pool_pids(engine: ConcurrentEmbeddingEngine) -> List[int]:
    pool = getattr(engine, "_process_pool", None)
    if pool is None:
        return []
    processes = getattr(pool, "_processes", None) or {}
    return list(processes.keys())


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


# --------------------------------------------------------------------------
# Single-trial execution
# --------------------------------------------------------------------------


@dataclass
class TrialResult:
    scenario: str
    concurrency: int
    duplicate_ratio: float
    pool_mode: str
    trial_index: int
    num_requests: int
    num_unique: int
    total_time_s: float
    avg_latency_ms: float
    p50_ms: float
    p95_ms: float
    throughput_rps: float
    cpu_percent_avg: Optional[float]
    memory_rss_peak_mb: Optional[float]
    actual_computations: int
    num_batches: int
    mean_batch_size: float
    max_batch_size: int
    dedup_count: int


def run_trial(
    scenario: str,
    engine: EmbeddingEngine,
    workload: List[str],
    concurrency: int,
    duplicate_ratio: float,
    pool_mode: str,
    trial_index: int,
    num_unique: int,
    count_queue=None,
    extra_pids_fn: Optional[Callable[[], List[int]]] = None,
) -> TrialResult:
    latencies: List[float] = []
    lat_lock = threading.Lock()

    def timed_call(text: str) -> None:
        t0 = time.perf_counter()
        engine.get_embedding(text)
        dt = time.perf_counter() - t0
        with lat_lock:
            latencies.append(dt)

    sampler = _ResourceSampler(extra_pids_fn=extra_pids_fn)
    sampler.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(timed_call, workload))
    total_time = time.perf_counter() - t0
    cpu_avg, mem_peak_mb = sampler.stop()

    if count_queue is not None:
        batch_sizes = _drain_counts(count_queue)
        actual_computations = sum(batch_sizes)
        num_batches = len(batch_sizes)
        mean_batch_size = actual_computations / num_batches if num_batches else 0.0
        max_batch = max(batch_sizes) if batch_sizes else 0
    else:
        actual_computations = len(workload)
        num_batches = len(workload)
        mean_batch_size = 1.0
        max_batch = 1

    dedup_count = max(0, len(workload) - actual_computations)

    return TrialResult(
        scenario=scenario,
        concurrency=concurrency,
        duplicate_ratio=duplicate_ratio,
        pool_mode=pool_mode,
        trial_index=trial_index,
        num_requests=len(workload),
        num_unique=num_unique,
        total_time_s=total_time,
        avg_latency_ms=(statistics.mean(latencies) * 1000) if latencies else 0.0,
        p50_ms=_percentile(latencies, 50) * 1000,
        p95_ms=_percentile(latencies, 95) * 1000,
        throughput_rps=(len(workload) / total_time) if total_time > 0 else 0.0,
        cpu_percent_avg=cpu_avg,
        memory_rss_peak_mb=mem_peak_mb,
        actual_computations=actual_computations,
        num_batches=num_batches,
        mean_batch_size=mean_batch_size,
        max_batch_size=max_batch,
        dedup_count=dedup_count,
    )


# --------------------------------------------------------------------------
# Aggregation across repeated trials: mean / std / 95% CI (Student's t)
# --------------------------------------------------------------------------


def mean_std_ci(
    values: List[Optional[float]], confidence: float = 0.95
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    clean = [v for v in values if v is not None]
    if not clean:
        return (None, None, None, None)
    n = len(clean)
    m = statistics.mean(clean)
    if n < 2:
        return (m, 0.0, m, m)
    sd = statistics.stdev(clean)
    if sd == 0 or scipy_stats is None:
        return (m, sd, m, m)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sd / math.sqrt(n)
    return (m, sd, m - margin, m + margin)


def welch_t_test_pvalue(a: List[float], b: List[float]) -> Optional[float]:
    if scipy_stats is None or len(a) < 2 or len(b) < 2:
        return None
    _, pvalue = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(pvalue)


@dataclass
class AggregatedResult:
    scenario: str
    concurrency: int
    duplicate_ratio: float
    pool_mode: str
    num_requests: int
    num_unique: int
    repeats: int
    total_time_s_mean: float
    total_time_s_std: float
    total_time_s_ci_low: float
    total_time_s_ci_high: float
    avg_latency_ms_mean: float
    avg_latency_ms_std: float
    p50_ms_mean: float
    p50_ms_std: float
    p95_ms_mean: float
    p95_ms_std: float
    throughput_rps_mean: float
    throughput_rps_std: float
    cpu_percent_mean: Optional[float]
    memory_mb_mean: Optional[float]
    actual_computations_mean: float
    dedup_count_mean: float
    dedup_fraction: float
    mean_batch_size_mean: float
    max_batch_size: int
    speedup_vs_baseline: Optional[float]
    pvalue_vs_baseline: Optional[float]
    significant_vs_baseline: Optional[bool]


def aggregate_trials(
    trials: List[TrialResult], baseline_trials: Optional[List[TrialResult]]
) -> AggregatedResult:
    first = trials[0]
    tt_m, tt_sd, tt_lo, tt_hi = mean_std_ci([t.total_time_s for t in trials])
    al_m, al_sd, _, _ = mean_std_ci([t.avg_latency_ms for t in trials])
    p50_m, p50_sd, _, _ = mean_std_ci([t.p50_ms for t in trials])
    p95_m, p95_sd, _, _ = mean_std_ci([t.p95_ms for t in trials])
    th_m, th_sd, _, _ = mean_std_ci([t.throughput_rps for t in trials])
    cpu_m, _, _, _ = mean_std_ci([t.cpu_percent_avg for t in trials])
    mem_m, _, _, _ = mean_std_ci([t.memory_rss_peak_mb for t in trials])
    comp_m, _, _, _ = mean_std_ci([float(t.actual_computations) for t in trials])
    dedup_m, _, _, _ = mean_std_ci([float(t.dedup_count) for t in trials])
    batch_m, _, _, _ = mean_std_ci([t.mean_batch_size for t in trials])
    max_batch = max(t.max_batch_size for t in trials)

    speedup = None
    pvalue = None
    significant = None
    if baseline_trials is not None and baseline_trials:
        baseline_mean = statistics.mean(t.total_time_s for t in baseline_trials)
        if tt_m and tt_m > 0:
            speedup = baseline_mean / tt_m
        pvalue = welch_t_test_pvalue(
            [t.total_time_s for t in baseline_trials], [t.total_time_s for t in trials]
        )
        if pvalue is not None:
            significant = pvalue < 0.05

    return AggregatedResult(
        scenario=first.scenario,
        concurrency=first.concurrency,
        duplicate_ratio=first.duplicate_ratio,
        pool_mode=first.pool_mode,
        num_requests=first.num_requests,
        num_unique=first.num_unique,
        repeats=len(trials),
        total_time_s_mean=tt_m,
        total_time_s_std=tt_sd,
        total_time_s_ci_low=tt_lo,
        total_time_s_ci_high=tt_hi,
        avg_latency_ms_mean=al_m,
        avg_latency_ms_std=al_sd,
        p50_ms_mean=p50_m,
        p50_ms_std=p50_sd,
        p95_ms_mean=p95_m,
        p95_ms_std=p95_sd,
        throughput_rps_mean=th_m,
        throughput_rps_std=th_sd,
        cpu_percent_mean=cpu_m,
        memory_mb_mean=mem_m,
        actual_computations_mean=comp_m,
        dedup_count_mean=dedup_m,
        dedup_fraction=(dedup_m / first.num_requests) if first.num_requests else 0.0,
        mean_batch_size_mean=batch_m,
        max_batch_size=max_batch,
        speedup_vs_baseline=speedup,
        pvalue_vs_baseline=pvalue,
        significant_vs_baseline=significant,
    )


# --------------------------------------------------------------------------
# Repeated-trial runners.
#
# run_repeats_cold: builds a brand-new engine/pool for EVERY trial, with NO
#   warm-up (warming it up would hide exactly the cold-start cost this mode
#   exists to measure).
# run_repeats_warm: reuses ONE already-built engine/pool across every trial,
#   performing a single untimed warm-up phase before the first measured
#   trial (unless the engine was already warmed by a previous call, e.g.
#   when the same pool is being reused across multiple concurrency levels).
# --------------------------------------------------------------------------


def run_repeats_cold(
    scenario: str,
    build_fresh_engine_fn: Callable[[], Tuple[EmbeddingEngine, object, Optional[Callable]]],
    concurrency: int,
    duplicate_ratio: float,
    repeats: int,
    prompt_pool: List[str],
    base_seed: int,
    requests_fn: Callable[[int], int],
) -> List[TrialResult]:
    trials = []
    for i in range(repeats):
        engine, q, pids_fn = build_fresh_engine_fn()
        num_requests = requests_fn(concurrency)
        workload, num_unique = make_workload(
            num_requests, duplicate_ratio, prompt_pool, seed=base_seed + i
        )
        trial = run_trial(
            scenario, engine, workload, concurrency, duplicate_ratio, "cold",
            i, num_unique, q, pids_fn,
        )
        trials.append(trial)
        _print_trial(trial, i, repeats)
        if hasattr(engine, "shutdown"):
            engine.shutdown()
    return trials


def run_repeats_warm(
    scenario: str,
    engine: EmbeddingEngine,
    count_queue,
    pids_fn: Optional[Callable[[], List[int]]],
    concurrency: int,
    duplicate_ratio: float,
    repeats: int,
    warmup_requests: int,
    prompt_pool: List[str],
    base_seed: int,
    already_warmed: bool,
    requests_fn: Callable[[int], int] = lambda c: max(2000, c * 100),
) -> List[TrialResult]:
    if not already_warmed:
        warm_wl, _ = make_workload(
            warmup_requests, duplicate_ratio, prompt_pool, seed=base_seed - 1
        )
        for text in warm_wl:
            engine.get_embedding(text)
        if count_queue is not None:
            _drain_counts(count_queue)

    trials = []
    for i in range(repeats):
        num_requests = requests_fn(concurrency)
        workload, num_unique = make_workload(
            num_requests, duplicate_ratio, prompt_pool, seed=base_seed + i
        )
        trial = run_trial(
            scenario, engine, workload, concurrency, duplicate_ratio, "warm",
            i, num_unique, count_queue, pids_fn,
        )
        trials.append(trial)
        _print_trial(trial, i, repeats)
    return trials


def _print_trial(trial: TrialResult, i: int, repeats: int) -> None:
    print(
        f"    [{trial.scenario:<8}] trial {i + 1}/{repeats}: "
        f"total={trial.total_time_s:7.2f}s avg={trial.avg_latency_ms:8.1f}ms "
        f"p95={trial.p95_ms:8.1f}ms thr={trial.throughput_rps:7.1f}req/s "
        f"computations={trial.actual_computations}/{trial.num_requests} "
        f"dedup={trial.dedup_count} batches={trial.num_batches} "
        f"mean_batch={trial.mean_batch_size:.2f}"
    )


# --------------------------------------------------------------------------
# CSV / JSON output helpers
# --------------------------------------------------------------------------


def write_trials_csv(path: Path, trials: List[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trials:
        return
    fieldnames = list(trials[0].__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            writer.writerow(t.__dict__)
    print(f"  wrote {path} ({len(trials)} rows)")


def write_summary_csv(path: Path, aggregates: List[AggregatedResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not aggregates:
        return
    fieldnames = list(aggregates[0].__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in aggregates:
            writer.writerow(a.__dict__)
    print(f"  wrote {path} ({len(aggregates)} rows)")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: getattr(o, "__dict__", str(o)))
    print(f"  wrote {path}")


# --------------------------------------------------------------------------
# Experiment 1: concurrency sweep at a fixed duplicate ratio.
# --------------------------------------------------------------------------


def run_experiment_concurrency_sweep(
    base_engine_factory: Callable[[], EmbeddingEngine],
    prompt_pool: List[str],
    pool_mode: str,
    repeats: int,
    num_workers: int,
    max_batch_size: int,
    duplicate_ratio: float = 0.5,
    warmup_requests: int = 150,
    requests_fn: Callable[[int], int] = lambda c: max(2000, c * 100),
    tag: str = "exp1_concurrency_sweep",
) -> Tuple[List[TrialResult], List[AggregatedResult]]:
    print(
        f"\n{'#' * 70}\n# EXPERIMENT 1 ({pool_mode} pool): concurrency sweep, "
        f"duplicate_ratio={duplicate_ratio}, repeats={repeats}\n{'#' * 70}"
    )
    all_trials: List[TrialResult] = []
    all_aggregates: List[AggregatedResult] = []

    thread_engine = process_engine = thread_q = process_q = process_pids_fn = None
    warmed = False
    if pool_mode == "warm":
        thread_engine, thread_q = build_engine_and_queue(
            base_engine_factory, EmbeddingExecutionMode.THREAD, num_workers, max_batch_size
        )
        process_engine, process_q = build_engine_and_queue(
            base_engine_factory, EmbeddingExecutionMode.PROCESS, num_workers, max_batch_size
        )
        process_pids_fn = lambda: _process_pool_pids(process_engine)  # noqa: E731

    try:
        for concurrency in CONCURRENCY_LEVELS:
            num_requests = requests_fn(concurrency)
            print(f"\n-- concurrency={concurrency} (num_requests={num_requests}) --")

            baseline_trials = run_repeats_cold(
                "original", lambda: (base_engine_factory(), None, None),
                concurrency, duplicate_ratio, repeats, prompt_pool,
                base_seed=1000 * concurrency, requests_fn=requests_fn,
            )
            all_trials += baseline_trials
            all_aggregates.append(aggregate_trials(baseline_trials, None))

            if pool_mode == "warm":
                thread_trials = run_repeats_warm(
                    "thread", thread_engine, thread_q, None, concurrency,
                    duplicate_ratio, repeats, warmup_requests, prompt_pool,
                    base_seed=1000 * concurrency + 1, already_warmed=warmed,
                    requests_fn=requests_fn,
                )
                process_trials = run_repeats_warm(
                    "process", process_engine, process_q, process_pids_fn, concurrency,
                    duplicate_ratio, repeats, warmup_requests, prompt_pool,
                    base_seed=1000 * concurrency + 2, already_warmed=warmed,
                    requests_fn=requests_fn,
                )
                warmed = True
            else:
                thread_trials = run_repeats_cold(
                    "thread",
                    lambda: (*build_engine_and_queue(
                        base_engine_factory, EmbeddingExecutionMode.THREAD,
                        num_workers, max_batch_size,
                    ), None),
                    concurrency, duplicate_ratio, repeats, prompt_pool,
                    base_seed=1000 * concurrency + 1, requests_fn=requests_fn,
                )
                process_trials = run_repeats_cold(
                    "process",
                    lambda: _build_process_triple(
                        base_engine_factory, num_workers, max_batch_size
                    ),
                    concurrency, duplicate_ratio, repeats, prompt_pool,
                    base_seed=1000 * concurrency + 2, requests_fn=requests_fn,
                )

            all_trials += thread_trials
            all_aggregates.append(aggregate_trials(thread_trials, baseline_trials))
            all_trials += process_trials
            all_aggregates.append(aggregate_trials(process_trials, baseline_trials))
    finally:
        if pool_mode == "warm":
            thread_engine.shutdown()
            process_engine.shutdown()

    write_trials_csv(RESULTS_DIR / f"{tag}_{pool_mode}_trials.csv", all_trials)
    write_summary_csv(RESULTS_DIR / f"{tag}_{pool_mode}_summary.csv", all_aggregates)
    return all_trials, all_aggregates


def _build_process_triple(base_engine_factory, num_workers, max_batch_size):
    engine, q = build_engine_and_queue(
        base_engine_factory, EmbeddingExecutionMode.PROCESS, num_workers, max_batch_size
    )
    return engine, q, (lambda: _process_pool_pids(engine))


# --------------------------------------------------------------------------
# Experiment 2: duplicate-ratio sweep at a fixed concurrency, warm-pool only
# (the ratio's effect on steady-state dedup/batching behavior is the point;
# cold-start cost is already characterized in Experiment 1).
# --------------------------------------------------------------------------


def run_experiment_duplicate_ratio_sweep(
    base_engine_factory: Callable[[], EmbeddingEngine],
    prompt_pool: List[str],
    repeats: int,
    num_workers: int,
    max_batch_size: int,
    concurrency: int = 16,
    warmup_requests: int = 150,
    tag: str = "exp2_duplicate_ratio_sweep",
) -> Tuple[List[TrialResult], List[AggregatedResult]]:
    print(
        f"\n{'#' * 70}\n# EXPERIMENT 2 (warm pool): duplicate-ratio sweep, "
        f"concurrency={concurrency}, repeats={repeats}\n{'#' * 70}"
    )
    all_trials: List[TrialResult] = []
    all_aggregates: List[AggregatedResult] = []

    thread_engine, thread_q = build_engine_and_queue(
        base_engine_factory, EmbeddingExecutionMode.THREAD, num_workers, max_batch_size
    )
    process_engine, process_q = build_engine_and_queue(
        base_engine_factory, EmbeddingExecutionMode.PROCESS, num_workers, max_batch_size
    )
    process_pids_fn = lambda: _process_pool_pids(process_engine)  # noqa: E731
    warmed = False

    try:
        for ratio in DUPLICATE_RATIOS:
            print(f"\n-- duplicate_ratio={ratio} --")
            seed_base = int(ratio * 10000) + 1

            baseline_trials = run_repeats_cold(
                "original", lambda: (base_engine_factory(), None, None),
                concurrency, ratio, repeats, prompt_pool,
                base_seed=seed_base, requests_fn=lambda c: max(2000, c * 100),
            )
            all_trials += baseline_trials
            all_aggregates.append(aggregate_trials(baseline_trials, None))

            thread_trials = run_repeats_warm(
                "thread", thread_engine, thread_q, None, concurrency, ratio,
                repeats, warmup_requests, prompt_pool,
                base_seed=seed_base + 1, already_warmed=warmed,
            )
            process_trials = run_repeats_warm(
                "process", process_engine, process_q, process_pids_fn, concurrency,
                ratio, repeats, warmup_requests, prompt_pool,
                base_seed=seed_base + 2, already_warmed=warmed,
            )
            warmed = True

            all_trials += thread_trials
            all_aggregates.append(aggregate_trials(thread_trials, baseline_trials))
            all_trials += process_trials
            all_aggregates.append(aggregate_trials(process_trials, baseline_trials))
    finally:
        thread_engine.shutdown()
        process_engine.shutdown()

    write_trials_csv(RESULTS_DIR / f"{tag}_trials.csv", all_trials)
    write_summary_csv(RESULTS_DIR / f"{tag}_summary.csv", all_aggregates)
    return all_trials, all_aggregates


# --------------------------------------------------------------------------
# Experiment 3: real embedding model validation.
# --------------------------------------------------------------------------


def run_experiment_real_model(
    prompt_pool: List[str],
    repeats_warm: int,
    repeats_cold: int,
    num_workers: int,
    max_batch_size: int,
    model_name: str,
    duplicate_ratio: float = 0.5,
    fixed_num_requests: int = 800,
    warmup_requests: int = 60,
    tag: str = "exp3_real_model",
) -> Dict[str, Tuple[List[TrialResult], List[AggregatedResult]]]:
    from vcache.vcache_core.cache.embedding_engine.strategies.lang_chain import (
        LangChainEmbeddingEngine,
    )

    def base_engine_factory():
        return LangChainEmbeddingEngine(model_name=model_name)

    print(
        f"\n{'#' * 70}\n# EXPERIMENT 3: real model ({model_name}), "
        f"duplicate_ratio={duplicate_ratio}, N={fixed_num_requests} (fixed, "
        f"not scaled with concurrency - real-model inference is far slower "
        f"per call than the synthetic engine, so N is intentionally smaller "
        f"here to keep runtime tractable; this is a disclosed scope "
        f"reduction for this experiment only)\n{'#' * 70}"
    )

    requests_fn = lambda c: fixed_num_requests  # noqa: E731

    warm_trials, warm_aggregates = run_experiment_concurrency_sweep(
        base_engine_factory=base_engine_factory,
        prompt_pool=prompt_pool,
        pool_mode="warm",
        repeats=repeats_warm,
        num_workers=num_workers,
        max_batch_size=max_batch_size,
        duplicate_ratio=duplicate_ratio,
        warmup_requests=warmup_requests,
        requests_fn=requests_fn,
        tag=tag,
    )

    cold_trials, cold_aggregates = run_experiment_concurrency_sweep(
        base_engine_factory=base_engine_factory,
        prompt_pool=prompt_pool,
        pool_mode="cold",
        repeats=repeats_cold,
        num_workers=num_workers,
        max_batch_size=max_batch_size,
        duplicate_ratio=duplicate_ratio,
        warmup_requests=warmup_requests,
        requests_fn=requests_fn,
        tag=tag,
    )

    return {
        "warm": (warm_trials, warm_aggregates),
        "cold": (cold_trials, cold_aggregates),
    }


# --------------------------------------------------------------------------
# Console summary table
# --------------------------------------------------------------------------


def print_summary_table(aggregates: List[AggregatedResult], title: str) -> None:
    header = (
        f"{'ratio':>6} | {'conc':>5} | {'mode':<6} | {'pool':<5} | {'scenario':<8} | "
        f"{'reqs':>5} | {'total(s)':>16} | {'p95(ms)':>10} | {'thr(req/s)':>11} | "
        f"{'comp':>6} | {'dedup':>6} | {'batch':>6} | {'speedup':>9} | {'p-value':>9} | {'sig?':>4}"
    )
    print(f"\n{'=' * len(header)}\n{title}\n{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for a in aggregates:
        tt = f"{a.total_time_s_mean:6.2f}±{a.total_time_s_std:5.2f}"
        speedup_str = f"{a.speedup_vs_baseline:8.2f}x" if a.speedup_vs_baseline else "      n/a"
        pvalue_str = f"{a.pvalue_vs_baseline:9.4f}" if a.pvalue_vs_baseline is not None else "      n/a"
        sig_str = (
            "YES" if a.significant_vs_baseline else ("no" if a.significant_vs_baseline is not None else "n/a")
        )
        print(
            f"{a.duplicate_ratio:6.1f} | {a.concurrency:5d} | {a.pool_mode:<6} | "
            f"n={a.repeats:<3} | {a.scenario:<8} | {a.num_requests:5d} | "
            f"{tt:>16} | {a.p95_ms_mean:10.1f} | {a.throughput_rps_mean:11.1f} | "
            f"{a.actual_computations_mean:6.0f} | {a.dedup_count_mean:6.0f} | "
            f"{a.mean_batch_size_mean:6.2f} | {speedup_str} | {pvalue_str} | {sig_str:>4}"
        )
    print("=" * len(header))


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["concurrency", "duplicate_ratio", "real_model", "all"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repeats-cold", type=int, default=3)
    parser.add_argument("--repeats-real-model-cold", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--work-units", type=int, default=30_000)
    parser.add_argument("--warmup-requests", type=int, default=150)
    parser.add_argument(
        "--model-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--real-model-num-requests", type=int, default=800)
    parser.add_argument(
        "--skip-cold", action="store_true", help="Skip cold-pool legs (faster, less complete)."
    )
    args = parser.parse_args()

    if psutil is None:
        print("NOTE: psutil not installed - CPU/memory will be n/a.")
    if scipy_stats is None:
        print("NOTE: scipy not installed - CI/p-values will be degenerate (std=0).")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    num_workers = args.num_workers or min(4, os.cpu_count() or 4)
    print(f"Loading real prompt pool from {PLAYGROUND_PARQUET} ...")
    prompt_pool = load_real_prompt_pool()
    print(f"Loaded {len(prompt_pool)} real prompts.")

    def synthetic_engine_factory():
        return SimulatedLocalEmbeddingEngine(work_units=args.work_units)

    all_output = {}

    if args.experiment in ("concurrency", "all"):
        warm_trials, warm_agg = run_experiment_concurrency_sweep(
            synthetic_engine_factory, prompt_pool, "warm", args.repeats,
            num_workers, args.max_batch_size, warmup_requests=args.warmup_requests,
        )
        print_summary_table(warm_agg, "EXPERIMENT 1 (warm pool) - concurrency sweep")
        all_output["exp1_warm"] = warm_agg

        if not args.skip_cold:
            cold_trials, cold_agg = run_experiment_concurrency_sweep(
                synthetic_engine_factory, prompt_pool, "cold", args.repeats_cold,
                num_workers, args.max_batch_size, warmup_requests=args.warmup_requests,
            )
            print_summary_table(cold_agg, "EXPERIMENT 1 (cold pool) - concurrency sweep")
            all_output["exp1_cold"] = cold_agg

    if args.experiment in ("duplicate_ratio", "all"):
        ratio_trials, ratio_agg = run_experiment_duplicate_ratio_sweep(
            synthetic_engine_factory, prompt_pool, args.repeats,
            num_workers, args.max_batch_size,
        )
        print_summary_table(ratio_agg, "EXPERIMENT 2 (warm pool) - duplicate-ratio sweep")
        all_output["exp2"] = ratio_agg

    if args.experiment in ("real_model", "all"):
        real_model_results = run_experiment_real_model(
            prompt_pool, args.repeats, args.repeats_real_model_cold,
            num_workers, args.max_batch_size, args.model_name,
            fixed_num_requests=args.real_model_num_requests,
        )
        print_summary_table(real_model_results["warm"][1], "EXPERIMENT 3 (warm pool) - real model")
        print_summary_table(real_model_results["cold"][1], "EXPERIMENT 3 (cold pool) - real model")
        all_output["exp3_warm"] = real_model_results["warm"][1]
        all_output["exp3_cold"] = real_model_results["cold"][1]

    write_json(RESULTS_DIR / "all_summaries.json", all_output)
    print(f"\nAll results written under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
