"""
Builds cost_advantage_demo.json: a small, fully deterministic synthetic
workload designed to isolate the one property that distinguishes
CostAwareEvictionPolicy from plain LRU: whether a policy protects
expensive-to-regenerate items even when they are NOT the most recently or
most frequently reused ones.

Structure (16 unique semantic clusters, fixed query order, no runtime
randomness once generated -- this file is a one-time generator, the output
JSON is the deterministic artifact actually used by run_cost_advantage_demo.py):

  - 4 "expensive" clusters (E1-E4, cost 40-70s -- simulating a slow LLM call)
  - 12 "cheap" clusters (C1-C12, cost 0.3-1.95s)

Query order (fixed list, not shuffled):
  1) Fill: insert all 16 clusters once (E1-E4, then C1-C12).
  2) Churn round 1: replay C1-C12 once (12 queries) -- recency pressure
     that touches only cheap items, so E1-E4 age untouched.
  3) Replay check 1: re-query E1-E4 (4 queries) -- do they still hit?
  4) Churn round 2: replay C1-C12 twice (24 queries) -- more pressure.
  5) Replay check 2: re-query E1-E4 (4 queries).
  6) Churn round 3: replay C1-C12 twice (24 queries) -- more pressure again.
  7) Replay check 3: re-query E1-E4 (4 queries).

Total: 88 queries. Every entry is tagged with "phase" and "cluster_type" so
the runner can break down hit rate specifically for the expensive-replay
checkpoints, which is the metric that actually demonstrates the policy's
advantage (overall hit rate does not, by design -- see the docstring in
run_cost_advantage_demo.py).

Embeddings are 512-dim (matching EmbeddingModel.E5_LARGE_V2) and synthetic:
each cluster gets a random unit base vector (fixed seed), and each query
occurrence is that base vector plus small Gaussian noise, renormalized. In
512 dimensions two independent random unit vectors are nearly orthogonal
(cosine similarity ~0), so different clusters are trivially well-separated
while same-cluster occurrences stay highly similar -- this is what makes
the nearest-neighbor cache lookup behave like real semantic paraphrases
without needing a real embedding model.
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015

EXPENSIVE_CLUSTERS = [
    ("E1", 101, 40.0),
    ("E2", 102, 50.0),
    ("E3", 103, 60.0),
    ("E4", 104, 70.0),
]
CHEAP_CLUSTERS = [
    (f"C{i}", 200 + i, round(0.30 + 0.15 * (i - 1), 2)) for i in range(1, 13)
]
ALL_CLUSTERS = EXPENSIVE_CLUSTERS + CHEAP_CLUSTERS
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}

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


# 1) Build the fixed query order as a list of (cluster_name, phase) pairs.
cheap_names = [name for name, _, _ in CHEAP_CLUSTERS]
expensive_names = [name for name, _, _ in EXPENSIVE_CLUSTERS]

query_plan = []
query_plan += [(n, "fill") for n in expensive_names]
query_plan += [(n, "fill") for n in cheap_names]

query_plan += [(n, "churn") for n in cheap_names]  # churn round 1 (12)
query_plan += [(n, "replay_1") for n in expensive_names]  # replay check 1 (4)

query_plan += [(n, "churn") for n in cheap_names] * 2  # churn round 2 (24)
query_plan += [(n, "replay_2") for n in expensive_names]  # replay check 2 (4)

query_plan += [(n, "churn") for n in cheap_names] * 2  # churn round 3 (24)
query_plan += [(n, "replay_3") for n in expensive_names]  # replay check 3 (4)

assert len(query_plan) == 88, len(query_plan)

# 2) Materialize each query into a full dataset row.
dataset = []
occurrence_count = {name: 0 for name, _, _ in ALL_CLUSTERS}
for cluster_name, phase in query_plan:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    occurrence_count[cluster_name] += 1
    cluster_type = "expensive" if cluster_name in expensive_names else "cheap"
    row = {
        "prompt": f"[{cluster_name}] occurrence #{occurrence_count[cluster_name]} ({phase})",
        "ID_Set": id_set,
        "phase": phase,
        "cluster": cluster_name,
        "cluster_type": cluster_type,
        "emb_e5_large_v2": make_embedding(cluster_name),
        "emb_e5_large_v2_lat": 0.01,
        "response_gpt-4o-mini": f"Answer for {cluster_name}",
        "response_gpt-4o-mini_lat": cost,
    }
    dataset.append(row)

# 3) Sanity-check embedding separation before writing anything out.
same_cluster_sims = []
cross_cluster_sims = []
for name in [n for n, _, _ in ALL_CLUSTERS]:
    occs = [np.array(r["emb_e5_large_v2"]) for r in dataset if r["cluster"] == name]
    for i in range(len(occs)):
        for j in range(i + 1, len(occs)):
            same_cluster_sims.append(float(np.dot(occs[i], occs[j])))
other_bases = list(base_vectors.values())
for i in range(len(other_bases)):
    for j in range(i + 1, len(other_bases)):
        cross_cluster_sims.append(float(np.dot(other_bases[i], other_bases[j])))

print(f"Same-cluster cosine similarity: mean={np.mean(same_cluster_sims):.4f}, min={np.min(same_cluster_sims):.4f}")
print(f"Cross-cluster cosine similarity: mean={np.mean(cross_cluster_sims):.4f}, max={np.max(cross_cluster_sims):.4f}")
assert np.min(same_cluster_sims) > np.max(cross_cluster_sims), (
    "Embedding separation too weak -- same-cluster similarity must exceed "
    "cross-cluster similarity for the nearest-neighbor lookup to behave "
    "like real semantic paraphrases."
)

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "cost_advantage_demo.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows to {out_path}")
print(f"Expensive clusters: {[(n, c) for n, _, c in EXPENSIVE_CLUSTERS]}")
print(f"Cheap clusters: {[(n, c) for n, _, c in CHEAP_CLUSTERS]}")
