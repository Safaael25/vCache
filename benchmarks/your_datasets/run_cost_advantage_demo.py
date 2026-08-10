"""
Runs cost_sensitive_retention.json (see
../dataset_generators/generate_cost_sensitive_retention.py) against
every eviction policy in this branch (FIFO, LRU, MRU, SCU, ARC, GPCA,
CostAware), all with the SAME max_size/watermark/eviction_percentage and the
SAME fixed query order, and reports the metric that actually demonstrates
cost-aware eviction's advantage.

Why not just report overall hit_rate: CostAwareEvictionPolicy deliberately
evicts some recently-used CHEAP items to protect rarely-used EXPENSIVE ones,
so its overall hit rate can be equal to or even lower than plain LRU's. The
benefit only shows up as reduced total regeneration cost. So this script
reports, per policy:
  - overall hit_rate (for reference / sanity check)
  - expensive_replay_hit_rate: hit rate specifically on the 3 "replay_N"
    checkpoints where E1-E4 are re-queried after being aged out by cheap-item
    churn -- this isolates the exact behavior under test.
  - cost_saved_sec: sum of response_gpt-4o-mini_lat for every query that was
    a cache HIT (cost avoided by not regenerating).
  - cost_incurred_sec: sum of response_gpt-4o-mini_lat for every query that
    was a cache MISS (cost actually paid).
  - cost_savings_ratio: cost_saved_sec / total_cost_no_cache_sec, where
    total_cost_no_cache_sec is the cost of answering every query from
    scratch with no cache at all (the standard cost-aware-eviction metric).
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
MAX_SIZE = 8
WATERMARK = 0.75
EVICTION_PCT = 0.25
COST_FIELD = "response_gpt-4o-mini_lat"

dataset_path = os.path.join(os.path.dirname(__file__), "cost_sensitive_retention.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    workload = json.load(f)
print(f"Loaded {len(workload)} rows from {dataset_path}\n")

total_cost_no_cache_sec = sum(row[COST_FIELD] for row in workload)
replay_phases = {"replay_1", "replay_2", "replay_3"}
replay_indices = [i for i, row in enumerate(workload) if row["phase"] in replay_phases]


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


POLICIES = {
    "FIFO": lambda: FIFOEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
    "LRU": lambda: LRUEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
    "MRU": lambda: MRUEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
    "SCU": lambda: SCUEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
    "ARC": lambda: ARCEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
    "GPCA": lambda: GPCAEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT),
    "CostAware": lambda: CostAwareEvictionPolicy(max_size=MAX_SIZE, watermark=WATERMARK, eviction_percentage=EVICTION_PCT, cost_weight=0.5),
}

results_dir = os.path.join(os.path.dirname(__file__), "cost_sensitive_retention_results")
os.makedirs(results_dir, exist_ok=True)

rows = []
for policy_name, make_policy in POLICIES.items():
    print(f"=== {policy_name} ===")
    eviction_policy = make_policy()
    vcache_config = make_config(eviction_policy)
    vcache = VCache(vcache_config, VerifiedDecisionPolicy(delta=DELTA))

    benchmark = Benchmark(vcache)
    benchmark.filepath = "cost_sensitive_retention"
    benchmark.embedding_model = EmbeddingModel.E5_LARGE_V2.value
    benchmark.llm_model = LargeLanguageModel.GPT_4O_MINI.value
    benchmark.timestamp = policy_name
    benchmark.threshold = None
    benchmark.delta = DELTA
    benchmark.is_static_threshold = False
    benchmark.output_folder_path = os.path.join(results_dir, policy_name)
    benchmark.eviction_policy = eviction_policy
    benchmark.is_custom_dataset = False
    benchmark.stats_set_up()

    benchmark.run_benchmark_loop(workload, len(workload))
    hits = benchmark.cache_hit_list

    cost_saved_sec = sum(row[COST_FIELD] for row, hit in zip(workload, hits) if hit)
    cost_incurred_sec = sum(row[COST_FIELD] for row, hit in zip(workload, hits) if not hit)
    replay_hits = [hits[i] for i in replay_indices]

    row_result = {
        "policy": policy_name,
        "hit_rate_pct": round(100 * sum(hits) / len(hits), 1),
        "expensive_replay_hit_rate_pct": round(100 * sum(replay_hits) / len(replay_hits), 1),
        "expensive_replay_hits": f"{sum(replay_hits)}/{len(replay_hits)}",
        "cost_saved_sec": round(cost_saved_sec, 2),
        "cost_incurred_sec": round(cost_incurred_sec, 2),
        "cost_savings_ratio_pct": round(100 * cost_saved_sec / total_cost_no_cache_sec, 1),
    }
    rows.append(row_result)
    print(f"  {row_result}\n")

    vcache.vcache_policy.shutdown()
    eviction_policy.shutdown()

result_df = pd.DataFrame(rows).sort_values("cost_saved_sec", ascending=False)
print(f"\ntotal_cost_no_cache_sec (baseline, zero caching): {round(total_cost_no_cache_sec, 2)}\n")
print("=== FULL RESULTS: cost_sensitive_retention.json ===")
print(result_df.to_string(index=False))
result_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
