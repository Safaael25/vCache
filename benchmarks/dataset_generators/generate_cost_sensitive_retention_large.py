"""
Scaled-up version of generate_cost_sensitive_retention.py: same fill -> churn ->
replay design, same principle (expensive items must survive churn from
unrelated cheap items to be worth anything), but 200 queries over a larger
40-cluster working set (10 expensive, 30 cheap) so that a cache-size sweep
of 10/20/40/80/160 has room to show both "tight cache, real pressure" and
"cache >= working set, no pressure left" regimes.

Structure:
  1) Fill: insert all 40 clusters once (10 expensive, then 30 cheap) -- 40 queries.
  2) Four repeated rounds of:
       - Churn: replay all 30 cheap clusters once (30 queries) -- recency
         pressure that never touches the expensive clusters.
       - Replay check: re-query all 10 expensive clusters (10 queries) --
         do they still hit after that round of churn?
     4 rounds x 40 queries = 160 queries.
  Total: 40 + 160 = 200 queries, fully deterministic (fixed seed, fixed
  hand-written order, no runtime randomness).

This gives 4 replay checkpoints x 10 expensive clusters = 40 replay data
points (vs. 12 in the original 88-query version), for more stable stats
across the cache-size sweep.
"""

import json
import os

import numpy as np

SEED = 42
EMBED_DIM = 512
NOISE_SCALE = 0.015

N_EXPENSIVE = 10
N_CHEAP = 30
N_CHURN_ROUNDS = 4

EXPENSIVE_CLUSTERS = [
    (f"E{i}", 300 + i, float(40 + 6 * (i - 1))) for i in range(1, N_EXPENSIVE + 1)
]  # cost 40, 46, 52, ..., 94
CHEAP_CLUSTERS = [
    (f"C{i}", 400 + i, round(0.30 + 0.10 * (i - 1), 2)) for i in range(1, N_CHEAP + 1)
]  # cost 0.30, 0.40, ..., 3.20
ALL_CLUSTERS = EXPENSIVE_CLUSTERS + CHEAP_CLUSTERS
CLUSTER_BY_NAME = {name: (id_set, cost) for name, id_set, cost in ALL_CLUSTERS}
expensive_names = [name for name, _, _ in EXPENSIVE_CLUSTERS]
cheap_names = [name for name, _, _ in CHEAP_CLUSTERS]

# Real-language stand-in for each cluster's underlying "question" -- purely
# for human readability in the dataset/logs. The benchmark itself never reads
# this text: BenchmarkInferenceEngine returns a pre-set response regardless
# of prompt content, and correctness is judged by ID_Set, not by text
# similarity. Expensive clusters read as long-form/multi-step generation
# tasks (plausibly slow); cheap clusters read as short factual lookups
# (plausibly fast) -- matching the cost each cluster is assigned above.
PROMPT_TEXT = {
    "E1": "Write a detailed 2,000-word essay comparing string theory and loop quantum gravity, including their experimental testability.",
    "E2": "Generate a complete REST API in Python with FastAPI, including authentication, rate limiting, and full test coverage.",
    "E3": "Summarize and cross-reference the key arguments from these five philosophy papers on personal identity, noting where they conflict.",
    "E4": "Draft a full business plan for a solar panel recycling startup, including market analysis, financial projections, and a 3-year roadmap.",
    "E5": "Translate this 10-page legal contract from German to English while preserving all clause numbering and legal terminology.",
    "E6": "Write a step-by-step proof of the Cayley-Hamilton theorem for n x n matrices, then extend it to the infinite-dimensional case.",
    "E7": "Design a complete microservices architecture for a ride-sharing app, with diagrams, data flow, and failure-mode analysis.",
    "E8": "Analyze this 50,000-row sales dataset and produce a full report with trend analysis, seasonality decomposition, and forecasts.",
    "E9": "Write a 15-chapter outline for a fantasy novel, including character arcs, world-building rules, and a three-act structure.",
    "E10": "Perform a full code review of this 5,000-line codebase, flagging security issues, performance bottlenecks, and architectural concerns.",
    "C1": "What's the capital of France?",
    "C2": "Convert 100 Fahrenheit to Celsius.",
    "C3": "Is 17 a prime number?",
    "C4": "What year did World War II end?",
    "C5": "Spell 'necessary' correctly.",
    "C6": "What's 15% of 200?",
    "C7": "Name the largest planet in our solar system.",
    "C8": "How many continents are there?",
    "C9": "What's the boiling point of water at sea level?",
    "C10": "Translate 'hello' to Spanish.",
    "C11": "What's the chemical symbol for gold?",
    "C12": "How many days are in a leap year?",
    "C13": "What's the square root of 144?",
    "C14": "Who wrote 'Romeo and Juliet'?",
    "C15": "What's the currency of Japan?",
    "C16": "How many sides does a hexagon have?",
    "C17": "What's the freezing point of water in Fahrenheit?",
    "C18": "Name the smallest country in the world.",
    "C19": "What's 7 times 8?",
    "C20": "What does 'HTTP' stand for?",
    "C21": "How many bones are in the human body?",
    "C22": "What's the tallest mountain in the world?",
    "C23": "Convert 5 kilometers to miles.",
    "C24": "What's the plural of 'cactus'?",
    "C25": "Who painted the Mona Lisa?",
    "C26": "What's the speed of light in a vacuum?",
    "C27": "How many minutes are in a day?",
    "C28": "What's the opposite of 'ubiquitous'?",
    "C29": "Name the first president of the United States.",
    "C30": "What's 2 to the power of 10?",
}
assert set(PROMPT_TEXT) == {name for name, _, _ in ALL_CLUSTERS}

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


# 1) Fixed query order.
query_plan = []
query_plan += [(n, "fill") for n in expensive_names]
query_plan += [(n, "fill") for n in cheap_names]

for round_idx in range(1, N_CHURN_ROUNDS + 1):
    query_plan += [(n, "churn") for n in cheap_names]
    query_plan += [(n, f"replay_{round_idx}") for n in expensive_names]

expected_total = N_EXPENSIVE + N_CHEAP + N_CHURN_ROUNDS * (N_CHEAP + N_EXPENSIVE)
assert len(query_plan) == expected_total == 200, len(query_plan)

# 2) Materialize each query into a full dataset row.
dataset = []
occurrence_count = {name: 0 for name, _, _ in ALL_CLUSTERS}
for cluster_name, phase in query_plan:
    id_set, cost = CLUSTER_BY_NAME[cluster_name]
    occurrence_count[cluster_name] += 1
    cluster_type = "expensive" if cluster_name in expensive_names else "cheap"
    row = {
        "prompt": PROMPT_TEXT[cluster_name],
        "occurrence_label": f"[{cluster_name}] occurrence #{occurrence_count[cluster_name]} ({phase})",
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
for name in [n for n, _, _ in ALL_CLUSTERS]:
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
assert np.min(same_cluster_sims) > np.max(cross_cluster_sims), (
    "Embedding separation too weak for the nearest-neighbor lookup to behave "
    "like real semantic paraphrases."
)

out_path = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "cost_sensitive_retention_large.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"\nWrote {len(dataset)} rows to {out_path}")
print(f"{N_EXPENSIVE} expensive clusters, cost {EXPENSIVE_CLUSTERS[0][2]}-{EXPENSIVE_CLUSTERS[-1][2]}s")
print(f"{N_CHEAP} cheap clusters, cost {CHEAP_CLUSTERS[0][2]}-{CHEAP_CLUSTERS[-1][2]}s")
print(f"Total unique clusters (working set size): {N_EXPENSIVE + N_CHEAP}")
