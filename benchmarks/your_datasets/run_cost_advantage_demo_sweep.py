"""
Same design as run_cost_advantage_demo.py, scaled to cost_sensitive_retention_large.json
(200 queries, 40 unique clusters: 10 expensive / 30 cheap) and swept across
cache sizes 10/20/40/80/160. watermark and eviction_percentage are held
IDENTICAL (0.75 / 0.25) at every cache size and for every policy -- only
max_size changes -- so the sweep isolates cache-size pressure, not policy
tuning.

Working set size is 40 unique clusters, so max_size=40 is the point where
the cache can hold everything at once (no pressure left), and max_size=80/160
are expected to converge to ~trivial, near-identical results across policies
by construction -- that convergence is itself the finding, not a bug: it
shows the cache-size regime where cost-aware eviction's advantage matters
(tight cache) vs. where it stops mattering (cache >= working set).
"""

import json
import os
import random

import numpy as np
import pandas as pd

from benchmarks.benchmark import Benchmark, EmbeddingModel, LargeLanguageModel
from vcache.config import VCacheConfig
from vcache.inference_engine.strategies.benchmark import BenchmarkInferenceEngine
from vcache.main import VCache
from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.vector_db import (
    HNSWLibVectorDB,
    SimilarityMetricType,
)
from vcache.vcache_core.cache.eviction_policy.strategies.arc import ARCEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.cost_aware import (
    CostAwareEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.fifo import FIFOEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.gpca import GPCAEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.lru import LRUEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.mru import MRUEvictionPolicy
from vcache.vcache_core.cache.eviction_policy.strategies.scu import SCUEvictionPolicy
from vcache.vcache_core.similarity_evaluator.strategies.benchmark_comparison import (
    BenchmarkComparisonSimilarityEvaluator,
)
from vcache.vcache_policy.strategies.verified import VerifiedDecisionPolicy

DELTA = 0.02
WATERMARK = 0.75
EVICTION_PCT = 0.25
CACHE_SIZES = [10, 20, 40, 80, 160]
COST_FIELD = "response_gpt-4o-mini_lat"

dataset_path = os.path.join(os.path.dirname(__file__), "cost_sensitive_retention_large.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    workload = json.load(f)
print(f"Loaded {len(workload)} rows from {dataset_path}\n")

total_cost_no_cache_sec = sum(row[COST_FIELD] for row in workload)
replay_phases = {p for p in {row["phase"] for row in workload} if p.startswith("replay_")}
replay_indices = [i for i, row in enumerate(workload) if row["phase"] in replay_phases]
print(f"Replay checkpoints: {sorted(replay_phases)} ({len(replay_indices)} total replay queries)\n")


def make_config(eviction_policy):
    return VCacheConfig(
        inference_engine=BenchmarkInferenceEngine(),
        embedding_engine=BenchmarkEmbeddingEngine(),
        vector_db=HNSWLibVectorDB(
            similarity_metric_type=SimilarityMetricType.COSINE,
            max_capacity=len(workload) + 10,
        ),
        embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
        similarity_evaluator=BenchmarkComparisonSimilarityEvaluator(),
        eviction_policy=eviction_policy,
    )


def make_policies(max_size):
    return {
        "FIFO": lambda: FIFOEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
        "LRU": lambda: LRUEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
        "MRU": lambda: MRUEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
        "SCU": lambda: SCUEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
        "ARC": lambda: ARCEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
        "GPCA": lambda: GPCAEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
        "CostAware": lambda: CostAwareEvictionPolicy(max_size=max_size, watermark=WATERMARK, eviction_percentage=EVICTION_PCT, cost_weight=0.5),
    }


results_dir = os.path.join(os.path.dirname(__file__), "cost_sensitive_retention_large_results")
os.makedirs(results_dir, exist_ok=True)

rows = []
for max_size in CACHE_SIZES:
    policies = make_policies(max_size)
    for policy_name, make_policy in policies.items():
        print(f"=== max_size={max_size} / {policy_name} ===")
        random.seed(42)  # neutralize VerifiedDecisionPolicy's unseeded explore/exploit draw
        eviction_policy = make_policy()
        vcache_config = make_config(eviction_policy)
        vcache = VCache(vcache_config, VerifiedDecisionPolicy(delta=DELTA))

        benchmark = Benchmark(vcache)
        benchmark.filepath = "cost_sensitive_retention_large"
        benchmark.embedding_model = EmbeddingModel.E5_LARGE_V2.value
        benchmark.llm_model = LargeLanguageModel.GPT_4O_MINI.value
        benchmark.timestamp = f"{max_size}_{policy_name}"
        benchmark.threshold = None
        benchmark.delta = DELTA
        benchmark.is_static_threshold = False
        benchmark.output_folder_path = os.path.join(results_dir, str(max_size), policy_name)
        benchmark.eviction_policy = eviction_policy
        benchmark.is_custom_dataset = False
        benchmark.stats_set_up()

        benchmark.run_benchmark_loop(workload, len(workload))
        hits = benchmark.cache_hit_list

        cost_saved_sec = sum(row[COST_FIELD] for row, hit in zip(workload, hits) if hit)
        cost_incurred_sec = sum(row[COST_FIELD] for row, hit in zip(workload, hits) if not hit)
        replay_hits = [hits[i] for i in replay_indices]

        vcache_lat = benchmark.latency_vcache_list
        direct_lat = benchmark.latency_direct_list
        lat_stats = Benchmark._latency_stats(vcache_lat)
        total_vcache_time = sum(vcache_lat) if vcache_lat else None
        throughput_qps = len(hits) / total_vcache_time if total_vcache_time else None

        tp, fp, tn, fn = sum(benchmark.tp_list), sum(benchmark.fp_list), sum(benchmark.tn_list), sum(benchmark.fn_list)
        expected_hit_ratio_pct = 100 * tp / (tp + fp) if (tp + fp) > 0 else None

        mean_cpu = float(np.mean(benchmark.cpu_percent_list)) if benchmark.cpu_percent_list else None
        mean_mem_mb = float(np.mean(benchmark.memory_mb_list)) if benchmark.memory_mb_list else None

        row_result = {
            "max_size": max_size,
            "policy": policy_name,
            "hit_rate_pct": round(100 * sum(hits) / len(hits), 1),
            "expensive_replay_hit_rate_pct": round(100 * sum(replay_hits) / len(replay_hits), 1),
            "expensive_replay_hits": f"{sum(replay_hits)}/{len(replay_hits)}",
            "cost_saved_sec": round(cost_saved_sec, 2),
            "cost_incurred_sec": round(cost_incurred_sec, 2),
            "cost_savings_ratio_pct": round(100 * cost_saved_sec / total_cost_no_cache_sec, 1),
            "expected_hit_ratio_pct": round(expected_hit_ratio_pct, 1) if expected_hit_ratio_pct is not None else None,
            "latency_vcache_mean_ms": round(lat_stats["mean"] * 1000, 3) if lat_stats["mean"] is not None else None,
            "latency_vcache_p95_ms": round(lat_stats["p95"] * 1000, 3) if lat_stats["p95"] is not None else None,
            "latency_vcache_p99_ms": round(lat_stats["p99"] * 1000, 3) if lat_stats["p99"] is not None else None,
            "latency_direct_mean_ms": round(1000 * float(np.mean(direct_lat)), 3) if direct_lat else None,
            "throughput_qps": round(throughput_qps, 2) if throughput_qps is not None else None,
            "mean_cpu_pct": round(mean_cpu, 2) if mean_cpu is not None else None,
            "mean_memory_mb": round(mean_mem_mb, 2) if mean_mem_mb is not None else None,
        }
        rows.append(row_result)
        print(f"  {row_result}")

        vcache.vcache_policy.shutdown()
        eviction_policy.shutdown()

result_df = pd.DataFrame(rows)
print(f"\ntotal_cost_no_cache_sec (baseline, zero caching): {round(total_cost_no_cache_sec, 2)}\n")
print("=== FULL SWEEP RESULTS: cost_sensitive_retention_large.json ===")
for max_size in CACHE_SIZES:
    sub = result_df[result_df["max_size"] == max_size].sort_values("cost_saved_sec", ascending=False)
    print(f"\n--- max_size={max_size} ---")
    print(sub.drop(columns=["max_size"]).to_string(index=False))
result_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
