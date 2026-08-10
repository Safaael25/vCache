"""
v2 of the fully-repeated best-case dataset. v1 (generate_hot_cold_repeat_workload.py)
spread every cluster's repeats across the ENTIRE 150-query timeline, which
made even the gap between a hot cluster's own repeats (~21 queries) too wide
for a 10-20 entry cache to survive against 30 competing clusters -- overall
hit rates came out low (0-8.7%), not the "high hit ratio" this dataset is
supposed to demonstrate.

v2 fixes this with ROUND-based scheduling instead of full-timeline spread:
  - 4 rounds. Hot clusters (15: HE1-5, HC1-10) appear in every round (4 reps
    each). Cold clusters (15: CE1-5, CC1-10) appear only in round 1 and
    round 4 (2 reps each -- still fully repeated, no one-offs).
  - Round 1 and 4: 15 hot + 15 cold = 30 queries each.
    Round 2 and 3: 15 hot only = 15 queries each.
    Total: 30 + 15 + 15 + 30 = 90 queries.
  - This bounds every gap: consecutive hot repeats are ~15-30 queries apart
    (survivable by a reasonably-sized cache), and the cold-cluster gap
    (round 1 -> round 4) is a fixed ~60 queries (still a real test, not an
    impossible one).

Same cost structure as v1 (comparable to Experiment 2): HE 40-64s, CE
46-94s, HC 0.3-1.2s, CC 1.3-2.2s. Same 30-cluster working set, so cache
sizes can be read the same way: 20 (below working set, real pressure),
40 (at/near working set, pressure eases), 80 (well above working set,
should approach ceiling hit rate for every policy).
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015
HOT_REPS = 4
COLD_REPS = 2

HOT_EXPENSIVE = [(f"HE{i}", 500 + i, float(40 + 6 * (i - 1))) for i in range(1, 6)]
COLD_EXPENSIVE = [(f"CE{i}", 510 + i, float(46 + 12 * (i - 1))) for i in range(1, 6)]
HOT_CHEAP = [(f"HC{i}", 520 + i, round(0.3 + 0.1 * (i - 1), 2)) for i in range(1, 11)]
COLD_CHEAP = [(f"CC{i}", 530 + i, round(1.3 + 0.1 * (i - 1), 2)) for i in range(1, 11)]

ALL_CLUSTERS = HOT_EXPENSIVE + COLD_EXPENSIVE + HOT_CHEAP + COLD_CHEAP
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}
hot_names = [n for n, _, _ in HOT_EXPENSIVE + HOT_CHEAP]
cold_names = [n for n, _, _ in COLD_EXPENSIVE + COLD_CHEAP]
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


# Fixed, varied (but deterministic) within-round order: interleave the four
# groups rather than blocking them, so same-cost/same-hotness items aren't
# bunched together in a way that could artificially help or hurt any policy.
def interleaved(*groups):
    out = []
    iters = [list(g) for g in groups]
    while any(iters):
        for it in iters:
            if it:
                out.append(it.pop(0))
    return out


hot_expensive_names = [n for n, _, _ in HOT_EXPENSIVE]
hot_cheap_names = [n for n, _, _ in HOT_CHEAP]
cold_expensive_names = [n for n, _, _ in COLD_EXPENSIVE]
cold_cheap_names = [n for n, _, _ in COLD_CHEAP]

hot_round_order = interleaved(hot_expensive_names, hot_cheap_names)
cold_round_order = interleaved(cold_expensive_names, cold_cheap_names)

round1 = interleaved(cold_round_order, hot_round_order)
round2 = list(hot_round_order)
round3 = list(reversed(hot_round_order))  # vary order so it's not identical to round2
round4 = interleaved(hot_round_order, cold_round_order)

query_plan = round1 + round2 + round3 + round4
assert len(query_plan) == 90, len(query_plan)

occurrence_count = {name: 0 for name, _, _ in ALL_CLUSTERS}
dataset = []
for cluster_name in query_plan:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    occurrence_count[cluster_name] += 1
    is_hot = cluster_name in hot_names
    is_expensive = cluster_name in expensive_names
    row = {
        "prompt": f"[{cluster_name}] occurrence #{occurrence_count[cluster_name]}",
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

from collections import Counter

counts = Counter(r["cluster"] for r in dataset)
assert min(counts.values()) >= 2, f"Found a one-off cluster: {counts}"
assert len(counts) == 30, len(counts)
for n in hot_names:
    assert counts[n] == HOT_REPS, (n, counts[n])
for n in cold_names:
    assert counts[n] == COLD_REPS, (n, counts[n])

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

# Report gaps for sanity.
def gaps_for(name):
    idxs = [i for i, r in enumerate(dataset) if r["cluster"] == name]
    return [b - a for a, b in zip(idxs, idxs[1:])]

print(f"Hot-cluster gap example (HE1): {gaps_for('HE1')}")
print(f"Cold-cluster gap example (CE1): {gaps_for('CE1')}")

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "hot_cold_repeat_workload_v2.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows, {len(counts)} unique clusters, to {out_path}")
