"""
High-temporal-locality workload, per the paper's own "Future Work" framing:
prior experiments deliberately used high eviction pressure (large gaps
between repeats, many competing clusters, tight caches), which is exactly
why ARC never got a fair test -- its T1->T2 promotion needs *sustained,
uninterrupted* presence to build hit_count, and every eviction resets an
item's observation history to zero on re-insertion. A workload with
substantially higher locality -- short, dense gaps between repeats -- is
the complementary regime where ARC's mechanism can actually function, and
where we can ask a different question than before: not "does it survive
brutal pressure" but "does it still improve cache efficiency when the
baseline hit rate is already decent."

Structure: ROUND-ROBIN, not sparse checkpoints. 15 unique clusters, each
touched exactly once per round, 10 rounds -> every cluster's repeat gap is
exactly 14 (one round length), the same for everyone, dense and short
compared to the previous sparse designs (which had gaps of 20-30+).
150 queries total (15 x 10), fully repeated, zero one-offs.

Same three groups as the v2 dual-advantage dataset (comparable cost
structure), but now identically dense repetition for all three:
  - EXPENSIVE (4, cost 40-90s): CostAware's target.
  - ARC_TARGET (4, cost 2-6s, cheap): ARC's target -- now with a real shot,
    since short gaps mean far fewer resets before hit_count can build up.
  - CHURN (7, cost 0.3-2s): still creates real pressure (see cache-size
    sweep below), just denser/shorter-range than before.
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015
N_ROUNDS = 10

EXPENSIVE = [(f"EXP{i}", 800 + i, float(40 + 16.7 * (i - 1))) for i in range(1, 5)]  # 40,56.7,73.3,90
ARC_TARGET = [(f"ARCT{i}", 810 + i, round(2 + 1.33 * (i - 1), 2)) for i in range(1, 5)]  # 2,3.33,4.67,6
CHURN = [(f"CHU{i}", 820 + i, round(0.3 + 0.28 * (i - 1), 2)) for i in range(1, 8)]  # 0.3..2.0

ALL_CLUSTERS = EXPENSIVE + ARC_TARGET + CHURN
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}
expensive_names = [n for n, _, _ in EXPENSIVE]
arc_target_names = [n for n, _, _ in ARC_TARGET]
churn_names = [n for n, _, _ in CHURN]
ALL_NAMES = expensive_names + arc_target_names + churn_names
assert len(ALL_NAMES) == 15

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


# 1) Round-robin sequence: every cluster once per round, deterministic
# per-round shuffle, no self-adjacency across round boundaries.
import random as _random

seq_rng = _random.Random(42)
while True:
    rounds = []
    for _ in range(N_ROUNDS):
        r = ALL_NAMES[:]
        seq_rng.shuffle(r)
        rounds.append(r)
    sequence = [x for r in rounds for x in r]
    if all(sequence[i] != sequence[i + 1] for i in range(len(sequence) - 1)):
        break

assert len(sequence) == 150

# 2) Materialize.
dataset = []
occurrence_count = {name: 0 for name in ALL_NAMES}
for cluster_name in sequence:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    occurrence_count[cluster_name] += 1
    if cluster_name in expensive_names:
        tier = "expensive"
    elif cluster_name in arc_target_names:
        tier = "arc_target"
    else:
        tier = "churn"
    row = {
        "prompt": f"[{cluster_name}] occurrence #{occurrence_count[cluster_name]}",
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
assert all(c == N_ROUNDS for c in counts.values()), counts
assert len(counts) == 15

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

print(f"Total queries: {len(dataset)}, unique clusters: {len(counts)}, reps each: {N_ROUNDS}, gap: {len(ALL_NAMES)-1}")
print(f"Same-cluster cosine similarity: mean={np.mean(same_cluster_sims):.4f}, min={np.min(same_cluster_sims):.4f}")
print(f"Cross-cluster cosine similarity: mean={np.mean(cross_cluster_sims):.4f}, max={np.max(cross_cluster_sims):.4f}")
assert np.min(same_cluster_sims) > np.max(cross_cluster_sims)

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "high_locality_dataset.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows to {out_path}")
print(f"Expensive: {expensive_names} cost 40-90s")
print(f"ARC target: {arc_target_names} cost 2-6s")
print(f"Churn: {churn_names} cost 0.3-2s")
