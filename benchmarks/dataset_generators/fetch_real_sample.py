"""
Pulls an 800-row real-data sample from each of the 4 vCache SemBenchmark
HuggingFace datasets via the HF datasets-server REST API (`/rows`), instead
of reading the parquet files directly.

Why: each dataset's parquet file is written as a single row group covering
every row, so parquet's only skip mechanism (row-group pruning) can't help
-- any direct read (streaming `datasets.load_dataset`, raw pyarrow, DuckDB
with LIMIT) has to decode the *entire* embedding column (~1-1.4GB) before
it can return even 1 row, which OOMs/thrashes on this machine. The
datasets-server `/rows` endpoint is a separate, pre-indexed HF service that
serves an arbitrary offset/length row range directly, without touching the
parquet file's column-chunk boundaries at all -- so it's the one access
path that's actually offset/length-bounded here. It caps `length` at 100
per request, so this paginates 8 requests per dataset to reach 800 rows.

Each row is parsed down to only the 6 fields the benchmark harness needs
(prompt, ID_Set, embedding, embedding latency, response, response latency)
immediately on receipt -- the other 5 embedding variants each dataset row
also carries are dropped right away rather than held in memory.
"""

import json
import os
import time

import requests

TARGET_ROWS = 800
PAGE_SIZE = 100  # datasets-server hard cap

# Each dataset only carries a subset of embedding models / LLM responses --
# picked per dataset to match what's actually present (confirmed via a
# 1-row schema probe), matching benchmarks/benchmark.py's EmbeddingModel /
# LargeLanguageModel column-name conventions so the saved JSON is a drop-in
# HuggingFace-shaped row for the existing Benchmark harness.
DATASET_CONFIGS = [
    {
        "repo_id": "vCache/SemBenchmarkLmArena",
        "embed_col": "emb_e5_large_v2",
        "response_col": "response_gpt-4o-mini",
    },
    {
        "repo_id": "vCache/SemBenchmarkSearchQueries",
        "embed_col": "emb_gte",
        "response_col": "response_llama_3_8b",
    },
    {
        "repo_id": "vCache/SemBenchmarkClassification",
        "embed_col": "emb_e5_large_v2",
        "response_col": "response_llama_3_8b",
    },
    {
        "repo_id": "vCache/SemBenchmarkCombo",
        "embed_col": "emb_gte",
        "response_col": "response_llama_3_8b",
    },
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "your_datasets", "real_samples")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_dataset(cfg: dict, target_rows: int) -> list:
    repo_id = cfg["repo_id"]
    embed_col = cfg["embed_col"]
    embed_lat_col = embed_col + "_lat"
    response_col = cfg["response_col"]
    response_lat_col = response_col + "_lat"

    rows = []
    offset = 0
    while len(rows) < target_rows:
        length = min(PAGE_SIZE, target_rows - len(rows))
        for attempt in range(4):
            try:
                resp = requests.get(
                    "https://datasets-server.huggingface.co/rows",
                    params={
                        "dataset": repo_id,
                        "config": "default",
                        "split": "train",
                        "offset": offset,
                        "length": length,
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                break
            except (requests.exceptions.RequestException,) as e:
                if attempt == 3:
                    raise
                wait = 5 * (attempt + 1)
                print(f"  [{repo_id}] request failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        page = resp.json()
        page_rows = page.get("rows", [])
        if not page_rows:
            print(f"  [{repo_id}] no more rows at offset {offset}, stopping early")
            break
        for entry in page_rows:
            r = entry["row"]
            embedding_raw = r[embed_col]
            embedding = (
                json.loads(embedding_raw)
                if isinstance(embedding_raw, str)
                else embedding_raw
            )
            rows.append(
                {
                    "prompt": r["prompt"],
                    "ID_Set": r.get("ID_Set", r.get("id_set", -1)),
                    embed_col: embedding,
                    embed_lat_col: float(r[embed_lat_col]),
                    response_col: r[response_col],
                    response_lat_col: float(r[response_lat_col]),
                }
            )
        offset += length
        print(f"  [{repo_id}] fetched {len(rows)}/{target_rows}")
    return rows


for cfg in DATASET_CONFIGS:
    repo_id = cfg["repo_id"]
    out_name = repo_id.split("/")[-1] + ".json"
    out_path = os.path.join(OUT_DIR, out_name)

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if len(existing) >= TARGET_ROWS:
            print(f"=== {repo_id}: already have {len(existing)} rows, skipping ===\n")
            continue

    print(f"=== {repo_id} (embed={cfg['embed_col']}, response={cfg['response_col']}) ===")
    t0 = time.time()
    rows = fetch_dataset(cfg, TARGET_ROWS)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print(f"  wrote {len(rows)} rows to {out_path} in {round(time.time() - t0, 1)}s\n")

print("Done.")
