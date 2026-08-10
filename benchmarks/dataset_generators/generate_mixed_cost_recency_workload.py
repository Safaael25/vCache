"""
A 150-query, fully-repeated synthetic dataset deliberately engineered so
that CostAware and ARC each have a distinct mechanism to win on, since code
review shows they cover different, non-overlapping blind spots in the four
baselines (FIFO/LRU/MRU/SCU) -- see the conversation for the full weakness
table. No baseline uses BOTH cost and frequency signal; CostAware only uses
cost, ARC only uses frequency. So the workload needs two separate tiers:

  - EXPENSIVE tier (5 clusters, cost 40-90s, 7 reps each, WIDE evenly-spread
    gaps): tests whether a policy protects a costly item regardless of
    access pattern. CostAware's specific target -- it doesn't care about
    gap width, only cost. Baselines (blind to cost) should lose these to
    generic staleness-based churn.
  - BURSTY tier (5 clusters, cost 3-8s, 3 "sessions" of 3 closely-spaced
    references each = 9 reps, sessions spread across the run): tests
    whether a policy notices an item is being *repeatedly returned to*,
    not just recently touched once. ARC's specific target -- its T1->T2
    promotion signal rewards sustained re-reference across sessions in a
    way plain LRU (which only remembers the single last access) cannot.
    CostAware gets little credit here since these items are cheap.
  - FILLER tier (10 clusters, cost 0.3-2s, 7 reps each, evenly spread):
    general churn/pressure, not meant to favor any policy.

20 unique clusters total (5+5+10), 150 queries (35+45+70), every cluster
repeats >=7 times (clears VerifiedDecisionPolicy's >=6-observation floor
with margin -- see the earlier non-determinism/cold-start investigation).
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015
TOTAL_QUERIES = 150

EXPENSIVE = [(f"EXP{i}", 600 + i, float(40 + 12.5 * (i - 1))) for i in range(1, 6)]  # 40,52.5,65,77.5,90
BURSTY = [(f"BUR{i}", 610 + i, round(3 + 1.25 * (i - 1), 2)) for i in range(1, 6)]  # 3,4.25,5.5,6.75,8
FILLER = [(f"FIL{i}", 620 + i, round(0.3 + 0.17 * (i - 1), 2)) for i in range(1, 11)]  # 0.3..1.83

ALL_CLUSTERS = EXPENSIVE + BURSTY + FILLER
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}
expensive_names = [n for n, _, _ in EXPENSIVE]
bursty_names = [n for n, _, _ in BURSTY]
filler_names = [n for n, _, _ in FILLER]

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


# 1) Schedule every occurrence as (position, cluster, occurrence_index),
# then merge and sort into one global deterministic sequence.
scheduled = []

# EXPENSIVE: 7 reps, evenly spread across the full timeline.
for name in expensive_names:
    reps = 7
    for occ in range(reps):
        pos = occ / (reps - 1) * (TOTAL_QUERIES - 1)
        scheduled.append((pos, name, occ, "expensive"))

# BURSTY: 3 sessions of 3 closely-spaced references each (9 reps total).
# Sessions spread evenly across the timeline; within a session, references
# land 1 slot apart.
for name in bursty_names:
    n_sessions = 3
    per_session = 3
    for s in range(n_sessions):
        session_pos = s / (n_sessions - 1) * (TOTAL_QUERIES - 1 - per_session)
        for k in range(per_session):
            occ = s * per_session + k
            scheduled.append((session_pos + k, name, occ, "bursty"))

# FILLER: 7 reps, evenly spread across the full timeline.
for name in filler_names:
    reps = 7
    for occ in range(reps):
        pos = occ / (reps - 1) * (TOTAL_QUERIES - 1)
        scheduled.append((pos, name, occ, "filler"))

# Deterministic tie-break: position, then cluster name, then occurrence.
scheduled.sort(key=lambda x: (x[0], x[1], x[2]))

# 2) Materialize into full dataset rows. Total slots may slightly exceed
# TOTAL_QUERIES due to rounding; trim/pad is unnecessary since we just use
# the natural count.
dataset = []
occurrence_count = {name: 0 for name, _, _ in ALL_CLUSTERS}
for _pos, cluster_name, _occ, tier in scheduled:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    occurrence_count[cluster_name] += 1
    row = {
        "prompt": f"[{cluster_name}] occurrence #{occurrence_count[cluster_name]} ({tier})",
        "ID_Set": id_set,
        "cluster": cluster_name,
        "tier": tier,
        "emb_e5_large_v2": make_embedding(cluster_name),
        "emb_e5_large_v2_lat": 0.01,
        "response_gpt-4o-mini": f"Answer for {cluster_name}",
        "response_gpt-4o-mini_lat": cost,
    }
    dataset.append(row)

# 3) Sanity checks.
from collections import Counter

counts = Counter(r["cluster"] for r in dataset)
assert min(counts.values()) >= 7, f"Some cluster has < 7 reps: {counts}"
assert len(counts) == 20, len(counts)

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

print(f"Total queries: {len(dataset)}, unique clusters: {len(counts)}")
print(f"Same-cluster cosine similarity: mean={np.mean(same_cluster_sims):.4f}, min={np.min(same_cluster_sims):.4f}")
print(f"Cross-cluster cosine similarity: mean={np.mean(cross_cluster_sims):.4f}, max={np.max(cross_cluster_sims):.4f}")
assert np.min(same_cluster_sims) > np.max(cross_cluster_sims)

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "mixed_cost_recency_workload.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows to {out_path}")
print(f"Expensive tier: {expensive_names} (cost 40-90s, 7 reps each, wide spread)")
print(f"Bursty tier: {bursty_names} (cost 3-8s, 3 sessions x 3 reps, sessions spread)")
print(f"Filler tier: {filler_names} (cost 0.3-1.83s, 7 reps each, wide spread)")
