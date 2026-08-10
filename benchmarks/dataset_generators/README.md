# Dataset generators

One-time scripts that build the synthetic/real workload JSON files consumed
by the `run_*.py` benchmarks in [`../your_datasets/`](../your_datasets/).
Each script writes its output there; the JSON files themselves are gitignored
(`benchmarks/your_datasets/*.json`) and are not committed — run the script to
regenerate them.

All scripts below are deterministic (fixed `random.seed` / `np.random.default_rng`
seed) and reproduce byte-identical output on every run, given the same
inputs. The exception is `fetch_real_sample.py`, which pulls live data from
the HuggingFace `datasets-server` API and is not reproducible byte-for-byte
(though it skips re-fetching if 800+ rows are already cached locally).

Run any script from this directory (or anywhere — paths are resolved
relative to the script file, not the cwd):

```
python3 build_temporal_locality_workload.py
```

## Scripts

- **`build_temporal_locality_workload.py`** — builds the round-robin,
  dense-repeat high-temporal-locality workload (150 rows, 15 clusters, gap
  14) used to give ARC's T1->T2 promotion a fair test. Generates cost, tier
  (expensive/arc_target/churn), the interleaved request sequence, and
  prompt text in one pass.
  `python3 build_temporal_locality_workload.py`

- **`generate_cost_sensitive_retention.py`** — builds
  `cost_sensitive_retention.json` (88 rows, 16 clusters: 4 expensive + 12
  cheap), a fixed churn/replay sequence isolating whether a policy protects
  expensive-to-regenerate items under recency pressure.
  `python3 generate_cost_sensitive_retention.py`

- **`generate_cost_sensitive_retention_large.py`** — larger variant of the
  above (200 rows, 40 clusters: 10 expensive + 30 cheap).
  `python3 generate_cost_sensitive_retention_large.py`

- **`generate_mixed_cost_recency_workload.py`** — builds
  `mixed_cost_recency_workload.json` (150 rows): expensive, bursty, and
  filler tiers combined to exercise both cost-awareness and recency
  dynamics together.
  `python3 generate_mixed_cost_recency_workload.py`

- **`generate_mixed_cost_recency_workload_v2.py`** — revised, denser
  variant of the mixed cost/recency workload (100 rows: expensive /
  arc_target / churn tiers).
  `python3 generate_mixed_cost_recency_workload_v2.py`

- **`generate_hot_cold_repeat_workload.py`** — builds
  `hot_cold_repeat_workload.json` (150 rows, 30 clusters, all clusters
  repeated at least twice: 15 "hot" clusters at 8 reps, 15 "cold" clusters
  at 2 reps).
  `python3 generate_hot_cold_repeat_workload.py`

- **`generate_hot_cold_repeat_workload_v2.py`** — revised hot/cold repeat
  workload with tighter hot-cluster gaps (90 rows, 30 clusters).
  `python3 generate_hot_cold_repeat_workload_v2.py`

- **`fetch_real_sample.py`** — pulls an 800-row real-data sample from each
  of the 4 vCache SemBenchmark HuggingFace datasets via the datasets-server
  `/rows` API, writing to `../your_datasets/real_samples/`. Not seeded
  (live network fetch); re-run is skipped automatically once 800+ rows are
  already cached for a dataset.
  `python3 fetch_real_sample.py`

- **`generate_real_samples_repeats.py`** — post-processes the real
  `SemBenchmarkLmArena` sample (requires `fetch_real_sample.py` to have run
  first) into an 800-row workload of 25 real topics x 32 reps each,
  round-robin interleaved, so every topic clears VerifiedDecisionPolicy's
  6-observation cold-start floor.
  `python3 generate_real_samples_repeats.py`
