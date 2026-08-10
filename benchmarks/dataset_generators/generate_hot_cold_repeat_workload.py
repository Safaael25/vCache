"""
Builds a third, idealized "fully-repeated best-case" dataset: every cluster
repeats at least twice, so no query is a structural one-off that can never
hit under any policy. This removes the dilution that made overall hit rates
low in cost_sensitive_retention_large.json (Experiment 2) and workload_profile_arc.json
(Experiment 1) -- there, most requests genuinely never repeat, by design, to
create eviction pressure. Here, eviction pressure instead comes entirely from
CACHE CAPACITY and REPEAT SPACING, not from unrepeatable filler traffic --
isolating pure eviction-policy quality from cold-start noise, and giving an
upper-bound hit-rate reference to read the partial-repetition experiments
against.

Design (see conversation for the three decisions this resolves):
  - Repeat spacing: SPREAD, not adjacent. Each cluster's N occurrences are
    scheduled at evenly-spaced fractional positions across the whole run
    (occurrence i at position i/(N-1) of the timeline), then every cluster's
    scheduled occurrences are merged and sorted into one global sequence.
    A cluster with only 2 repeats gets one occurrence near the very start
    and one near the very end -- the widest possible gap, so "did the cache
    still have it" is a real question, not a trivial adjacent-repeat freebie.
  - Cluster distribution: SKEWED. 15 "hot" clusters repeat 8 times each;
    15 "cold" clusters repeat exactly twice (the minimum for "fully
    repeated"). This is more realistic than uniform repetition and gives
    eviction policies something to actually differentiate on.
  - Cost structure: reused from cost_sensitive_retention_large.json's cheap/
    expensive split (same 40-94s / 0.3-2.2s ranges) so this dataset stays
    directly comparable to Experiment 2, and so CostAware has a real signal
    to act on -- with uniform cost, CostAware degenerates toward recency/
    frequency-only behavior and there'd be nothing to isolate.

Four cluster groups (30 unique clusters, 150 total queries, minimum repeat
count is 2 -- no one-offs anywhere):
  - HE1-HE5  hot,  expensive (cost 40-64s), 8 reps each   -> 40 queries
  - CE1-CE5  cold, expensive (cost 46-94s), 2 reps each   -> 10 queries
  - HC1-HC10 hot,  cheap (cost 0.3-1.2s),   8 reps each   -> 80 queries
  - CC1-CC10 cold, cheap (cost 1.3-2.2s),   2 reps each   -> 20 queries
                                                    total: 150 queries

The critical comparison is CE (cold + expensive): these clusters are exactly
as rare as CC (cold + cheap), but expensive to regenerate. A policy that
tracks only recency/frequency should treat CE and CC identically and lose
both between their two widely-spaced occurrences; CostAware should protect
CE specifically because of its cost; ARC should do better than plain
recency/frequency baselines on both HE and CE by virtue of adapting to the
mix of one-time-then-repeated items across the whole run.
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015
TOTAL_QUERIES = 150

HOT_REPS = 8
COLD_REPS = 2

HOT_EXPENSIVE = [(f"HE{i}", 500 + i, float(40 + 6 * (i - 1))) for i in range(1, 6)]
COLD_EXPENSIVE = [(f"CE{i}", 510 + i, float(46 + 12 * (i - 1))) for i in range(1, 6)]
HOT_CHEAP = [(f"HC{i}", 520 + i, round(0.3 + 0.1 * (i - 1), 2)) for i in range(1, 11)]
COLD_CHEAP = [(f"CC{i}", 530 + i, round(1.3 + 0.1 * (i - 1), 2)) for i in range(1, 11)]

ALL_CLUSTERS = HOT_EXPENSIVE + COLD_EXPENSIVE + HOT_CHEAP + COLD_CHEAP
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}
hot_names = {n for n, _, _ in HOT_EXPENSIVE + HOT_CHEAP}
expensive_names = {n for n, _, _ in HOT_EXPENSIVE + COLD_EXPENSIVE}

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


# 1) Schedule every cluster's occurrences at evenly-spaced fractional
# positions, then merge and sort into one global, deterministic order.
scheduled = []  # (position, cluster_name, occurrence_index)
for name, _id_set, _cost in ALL_CLUSTERS:
    reps = HOT_REPS if name in hot_names else COLD_REPS
    for occ in range(reps):
        position = occ / (reps - 1) * (TOTAL_QUERIES - 1) if reps > 1 else 0.0
        scheduled.append((position, name, occ))

# Deterministic tie-break: position, then cluster name, then occurrence index.
scheduled.sort(key=lambda x: (x[0], x[1], x[2]))
assert len(scheduled) == TOTAL_QUERIES, len(scheduled)

# 2) Materialize each scheduled slot into a full dataset row.
dataset = []
for position, cluster_name, occ in scheduled:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    is_hot = cluster_name in hot_names
    is_expensive = cluster_name in expensive_names
    row = {
        "prompt": f"[{cluster_name}] occurrence #{occ + 1}",
        "ID_Set": id_set,
        "cluster": cluster_name,
        "cluster_type": ("hot" if is_hot else "cold")
        + "_"
        + ("expensive" if is_expensive else "cheap"),
        "emb_e5_large_v2": make_embedding(cluster_name),
        "emb_e5_large_v2_lat": 0.01,
        "response_gpt-4o-mini": f"Answer for {cluster_name}",
        "response_gpt-4o-mini_lat": cost,
    }
    dataset.append(row)

# 3) Sanity checks: every cluster repeats >= 2 times (no one-offs), and
# embeddings stay well-separated.
from collections import Counter

counts = Counter(r["cluster"] for r in dataset)
assert min(counts.values()) >= 2, f"Found a one-off cluster: {counts}"
assert len(counts) == 30, len(counts)

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

print(f"Same-cluster cosine similarity: mean={np.mean(same_cluster_sims):.4f}, min={np.min(same_cluster_sims):.4f}")
print(f"Cross-cluster cosine similarity: mean={np.mean(cross_cluster_sims):.4f}, max={np.max(cross_cluster_sims):.4f}")
assert np.min(same_cluster_sims) > np.max(cross_cluster_sims)
print(f"Repeat counts: {sorted(set(counts.values()))} -- min repeat = {min(counts.values())} (no one-offs)")

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "hot_cold_repeat_workload.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows, {len(counts)} unique clusters, to {out_path}")
print(f"Hot clusters (8 reps): {sorted(hot_names)}")
print(f"Cold clusters (2 reps): {sorted(n for n in counts if n not in hot_names)}")
