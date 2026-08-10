"""
v2 of the dual-advantage dataset. v1 used continuous fractional-position
scheduling for a "bursty" ARC-target tier, merged into one global timeline
with everything else -- the burst locality got diluted by interleaving with
other clusters, and sessions landed before the >=5-occurrence floor needed
for VerifiedDecisionPolicy to ever exploit. Result: ARC's target tier got
0% hits at every cache size. Root cause: continuous scheduling, not enough
signal.

v2 uses the ROUND-BASED CHECKPOINT structure that has actually worked every
other time in this project (Experiment 1's real warmup->scan->recovery, and
the successful Experiment 2 cost-advantage sweep): insert once, then
alternate disruption rounds with explicit replay-checkpoints of the
protected items. Fully repeated -- zero one-off queries anywhere, every
cluster >= 5 reps (clears VerifiedDecisionPolicy's observation floor).

Three cluster groups, 20 total, all repeated identically (1 insert + 4
replay checkpoints = 5 reps each) so the ONLY thing that differs between
groups is the eviction policy's response to their cost, not their access
pattern:
  - EXPENSIVE (4 clusters, cost 40-90s): CostAware's target. Cheap-blind
    baselines should lose these to churn; CostAware should protect them via
    cost regardless of being touched no more often than anyone else.
  - ARC_TARGET (4 clusters, cost 2-6s, cheap): ARC's target. CostAware gives
    these no special credit (they're cheap) -- but they get replayed at the
    exact same checkpoints as EXPENSIVE, so ARC's T1->T2 promotion (any item
    hit even once graduates to frequency-based protection) should protect
    them purely from being repeatedly touched, something no baseline tracks.
  - CHURN (12 clusters, cost 0.3-2s): replayed once per round, providing
    real disruption pressure. Repeated (4 reps: 1 insert + 3 more churn
    passes) so nothing in this dataset is a true one-off.

Structure: insert all 20 once (20 queries), then 4 rounds of [churn: replay
all 12 churn clusters once (12) -> checkpoint: replay all 4 expensive + all
4 arc_target clusters (8)] = 20 * 4 = 80. Total: 20 + 80 = 100 queries.
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015
N_ROUNDS = 4

EXPENSIVE = [(f"EXP{i}", 700 + i, float(40 + 16.7 * (i - 1))) for i in range(1, 5)]  # 40,56.7,73.3,90
ARC_TARGET = [(f"ARCT{i}", 710 + i, round(2 + 1.33 * (i - 1), 2)) for i in range(1, 5)]  # 2,3.33,4.67,6
CHURN = [(f"CHU{i}", 720 + i, round(0.3 + 0.155 * (i - 1), 2)) for i in range(1, 13)]  # 0.3..2.0

ALL_CLUSTERS = EXPENSIVE + ARC_TARGET + CHURN
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}
expensive_names = [n for n, _, _ in EXPENSIVE]
arc_target_names = [n for n, _, _ in ARC_TARGET]
churn_names = [n for n, _, _ in CHURN]
checkpoint_names = expensive_names + arc_target_names

rng = np.random.default_rng(SEED)
base_vectors = {}
for name, _id_set, _cost in ALL_CLUSTERS:
    v = rng.standard_normal(EMBED_DIM)
    base_vectors[name] = v / np.linalg.norm(v)


def make_embedding(cluster_name: str) -> list:
    base = base_vectors[cluster_name]
    noisy = base + NOISE_SCALE * rng.standard_normal(EMBED_DIM)
    noisy = noisy / np.linalg.norm(noisy)
    return noisy.tolist()


# 1) Fixed query order: fill once, then N_ROUNDS x (churn + checkpoint).
query_plan = []
query_plan += [(n, "fill") for n in checkpoint_names + churn_names]
for r in range(1, N_ROUNDS + 1):
    query_plan += [(n, "churn") for n in churn_names]
    query_plan += [(n, f"checkpoint_{r}") for n in checkpoint_names]

expected_total = len(ALL_CLUSTERS) + N_ROUNDS * (len(churn_names) + len(checkpoint_names))
assert len(query_plan) == expected_total == 100, len(query_plan)

# 2) Materialize.
dataset = []
occurrence_count = {name: 0 for name, _, _ in ALL_CLUSTERS}
for cluster_name, phase in query_plan:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    occurrence_count[cluster_name] += 1
    if cluster_name in expensive_names:
        tier = "expensive"
    elif cluster_name in arc_target_names:
        tier = "arc_target"
    else:
        tier = "churn"
    row = {
        "prompt": f"[{cluster_name}] occurrence #{occurrence_count[cluster_name]} ({phase})",
        "ID_Set": id_set,
        "cluster": cluster_name,
        "tier": tier,
        "phase": phase,
        "emb_e5_large_v2": make_embedding(cluster_name),
        "emb_e5_large_v2_lat": 0.01,
        "response_gpt-4o-mini": f"Answer for {cluster_name}",
        "response_gpt-4o-mini_lat": cost,
    }
    dataset.append(row)

# 3) Sanity checks: every cluster >= 4 reps (fill + N_ROUNDS or fill + all
# churn passes), no one-offs anywhere.
from collections import Counter

counts = Counter(r["cluster"] for r in dataset)
assert min(counts.values()) >= 4, f"Found a near-one-off cluster: {counts}"
assert len(counts) == 20, len(counts)
for n in checkpoint_names:
    assert counts[n] == 1 + N_ROUNDS, (n, counts[n])  # 1 fill + 4 checkpoints = 5
for n in churn_names:
    assert counts[n] == 1 + N_ROUNDS, (n, counts[n])  # 1 fill + 4 churn passes = 5

same_cluster_sims = []
for name in counts:
    occs = [np.array(r["emb_e5_large_v2"]) for r in dataset if r["cluster"] == name]
    for i in range(len(occs)):
        for j in range(i + 1, len(occs)):
            same_cluster_sims.append(float(np.dot(occs[i], occs[j])))
other_bases = list(base_vectors.values())
cross_cluster_sims = []
for i in range(len(other_bases)):
    for j in range(i + 1, len(other_bases)):
        cross_cluster_sims.append(float(np.dot(other_bases[i], other_bases[j])))

print(f"Total queries: {len(dataset)}, unique clusters: {len(counts)}, min reps: {min(counts.values())}")
print(f"Same-cluster cosine similarity: mean={np.mean(same_cluster_sims):.4f}, min={np.min(same_cluster_sims):.4f}")
print(f"Cross-cluster cosine similarity: mean={np.mean(cross_cluster_sims):.4f}, max={np.max(cross_cluster_sims):.4f}")
assert np.min(same_cluster_sims) > np.max(cross_cluster_sims)

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "dual_advantage_dataset_v2.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows to {out_path}")
print(f"Expensive (CostAware target): {expensive_names} cost 40-90s")
print(f"ARC target (cheap, repeatedly touched): {arc_target_names} cost 2-6s")
print(f"Churn (disruption, also repeated): {churn_names} cost 0.3-2s")
