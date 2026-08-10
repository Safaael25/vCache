"""
Runs hot_cold_repeat_workload.json (see
../dataset_generators/generate_hot_cold_repeat_workload.py) across
three cache sizes -- 10, 15, 20 entries, against a 30-cluster working set --
and all seven eviction policies. Same watermark/eviction_percentage held
constant throughout, matching the convention from the other two experiments.

Reports, per policy/size:
  - overall hit rate (meaningful here, unlike Exp 1/2, since every query can
    hit in principle -- no structural one-offs)
  - hit rate specifically on COLD-EXPENSIVE clusters (CE1-CE5): the key
    comparison. These are exactly as rare as COLD-CHEAP (2 reps, same wide
    spacing) but expensive to regenerate -- a policy blind to cost should
    treat them identically to COLD-CHEAP and lose both; CostAware should
    protect them specifically.
  - hit rate on COLD-CHEAP (CC1-CC10), as the "should lose these, that's
    fine" control -- if a policy protects these just as well as CE, cost
    isn't actually driving its behavior.
  - cost saved / cost savings ratio, same definition as Experiment 2.
"""

import json
import os

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
CACHE_SIZES = [10, 15, 20]
COST_FIELD = "response_gpt-4o-mini_lat"

dataset_path = os.path.join(os.path.dirname(__file__), "hot_cold_repeat_workload.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    workload = json.load(f)
print(f"Loaded {len(workload)} rows from {dataset_path}\n")

total_cost_no_cache_sec = sum(row[COST_FIELD] for row in workload)
ce_indices = [i for i, row in enumerate(workload) if row["cluster_type"] == "cold_expensive"]
cc_indices = [i for i, row in enumerate(workload) if row["cluster_type"] == "cold_cheap"]
he_indices = [i for i, row in enumerate(workload) if row["cluster_type"] == "hot_expensive"]


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


results_dir = os.path.join(os.path.dirname(__file__), "hot_cold_repeat_sweep_results")
os.makedirs(results_dir, exist_ok=True)


def hit_rate(hits, indices):
    if not indices:
        return None
    sub = [hits[i] for i in indices]
    return 100 * sum(sub) / len(sub)


rows = []
for max_size in CACHE_SIZES:
    for policy_name, make_policy in make_policies(max_size).items():
        print(f"=== max_size={max_size} / {policy_name} ===")
        eviction_policy = make_policy()
        vcache_config = make_config(eviction_policy)
        vcache = VCache(vcache_config, VerifiedDecisionPolicy(delta=DELTA))

        benchmark = Benchmark(vcache)
        benchmark.filepath = "hot_cold_repeat_workload"
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

        row_result = {
            "max_size": max_size,
            "policy": policy_name,
            "overall_hit_rate_pct": round(100 * sum(hits) / len(hits), 1),
            "cold_expensive_hit_rate_pct": round(hit_rate(hits, ce_indices), 1),
            "cold_cheap_hit_rate_pct": round(hit_rate(hits, cc_indices), 1),
            "hot_expensive_hit_rate_pct": round(hit_rate(hits, he_indices), 1),
            "cost_saved_sec": round(cost_saved_sec, 2),
            "cost_savings_ratio_pct": round(100 * cost_saved_sec / total_cost_no_cache_sec, 1),
        }
        rows.append(row_result)
        print(f"  {row_result}")

        vcache.vcache_policy.shutdown()
        eviction_policy.shutdown()

result_df = pd.DataFrame(rows)
print(f"\ntotal_cost_no_cache_sec: {round(total_cost_no_cache_sec, 2)}\n")
print("=== FULL RESULTS: hot_cold_repeat_workload.json ===")
for max_size in CACHE_SIZES:
    sub = result_df[result_df["max_size"] == max_size].sort_values("cold_expensive_hit_rate_pct", ascending=False)
    print(f"\n--- max_size={max_size} ---")
    print(sub.drop(columns=["max_size"]).to_string(index=False))
result_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
