"""
Builds a real-data 800-row workload actually capable of producing hits.

Why this is needed: the plain first-800-rows slice of Arena
(real_samples/SemBenchmarkLmArena.json) has 707 distinct topics out of 800
rows -- almost everything is a one-off. But VerifiedDecisionPolicy has a
hard cold-start gate (verified.py: `if len(observations) < 6: EXPLORE`) --
any cached item needs at least 6 real observations before a hit is even
possible, regardless of how similar a query is. A workload dominated by
one-off topics is structurally incapable of producing many hits, no matter
how good the eviction policy is.

This script instead selects N real topics that already have a genuine
duplicate ID_Set (2-3 real occurrences each, found in the existing 800-row
sample -- no new downloads needed) and replays each one R times, cycling
through its real occurrences, round-robin interleaved with the other
topics -- so every included topic clears the 6-observation floor early and
has many remaining real chances to actually hit.
"""

import json
import os
from collections import defaultdict

N_TOPICS = 25
REPS_PER_TOPIC = 32  # N_TOPICS * REPS_PER_TOPIC = 800

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "real_samples")
src_path = os.path.join(SAMPLES_DIR, "SemBenchmarkLmArena.json")
with open(src_path, "r", encoding="utf-8") as f:
    rows = json.load(f)

by_id = defaultdict(list)
for r in rows:
    by_id[r["ID_Set"]].append(r)

dup_clusters = {k: v for k, v in by_id.items() if len(v) > 1}
# Prefer clusters with more real occurrences (more paraphrase variety) first.
ordered_clusters = sorted(dup_clusters.items(), key=lambda kv: -len(kv[1]))
selected = ordered_clusters[:N_TOPICS]
print(f"Selected {len(selected)} real topics (cluster sizes: {[len(v) for _, v in selected]})")

# Build each topic's replay sequence by cycling through its real occurrences.
topic_sequences = []
for id_set, occurrences in selected:
    seq = [occurrences[i % len(occurrences)] for i in range(REPS_PER_TOPIC)]
    topic_sequences.append(seq)

# Round-robin interleave across topics.
workload = []
for rep_idx in range(REPS_PER_TOPIC):
    for topic_idx in range(N_TOPICS):
        workload.append(topic_sequences[topic_idx][rep_idx])

assert len(workload) == N_TOPICS * REPS_PER_TOPIC, len(workload)

out_path = os.path.join(SAMPLES_DIR, "SemBenchmarkLmArena_repeats.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(workload, f)

print(f"Wrote {len(workload)} rows to {out_path}")
print(f"{N_TOPICS} real topics x {REPS_PER_TOPIC} reps each, round-robin interleaved")
