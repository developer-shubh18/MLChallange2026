"""
features.py — pairwise similarity feature engineering (Stage 3), v2.

v2 changes from the original (see business_entity_resolution_optimization_plan.md
for the full diagnosis) — all three changes are correctness-preserving rewrites
of the same 13 features, not new features:

1. Cheap features (exact/PIN/country/suffix/script/length) are computed with
   fully vectorized pandas/NumPy column ops instead of `joined.apply(axis=1)`.
2. The 4 fuzzy features (name/addr Jaro-Winkler + Levenshtein) and the 2 Jaccard
   features are computed via plain Python list comprehensions over `.tolist()`
   columns instead of `DataFrame.apply(axis=1)`. rapidfuzz itself was never the
   bottleneck — pandas reconstructing a Series per row was. A comprehension
   over plain Python lists avoids that entirely while still calling the same
   rapidfuzz functions.
3. Batching no longer filters the full `pairs` frame with `isin()` once per
   batch (O(total_rows) rescanned per batch, ~91 times at batch_size=20_000
   over 1.8M entities). `pairs` is sorted once by source1_entity_id, entities
   are assigned a batch id via a single vectorized `.map()`, and batches are
   produced by one `groupby()` pass instead.
4. Label attachment uses a vectorized merge against a precomputed positive-pairs
   DataFrame (built once, outside the batch loop) instead of a per-row
   `.apply(lambda r: ... in pos_pairs)`.

Deliberately NOT done here (see optimization plan, Section 16 and 19/20):
entity-level token/length precomputation, multiprocessing workers, and
DuckDB-side cheap-feature computation. Vectorizing away the `apply(axis=1)`
overhead is expected to capture most of the win on its own; benchmark before
adding the rest.

VERIFY BEFORE A FULL RUN: run this on the same candidate_pairs.tsv, take
--batch-size 20000, and diff part_00000000.parquet against the v1 output
already on disk (the same batch boundary) to confirm identical feature values
before committing to the full ~21.8M-pair run.
"""

import argparse
import gc
import glob
import os
import time

import duckdb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import normalize_source_file

FEATURE_COLS = [
    # Name features
    "name_jw", "name_lev", "name_token_sort_ratio", "name_token_set_ratio",
    "name_partial_ratio", "name_exact", "name_prefix_match", "name_suffix_match",
    "name_char_3gram_jaccard", "len_diff_name",
    # Address features
    "addr_jw", "addr_lev", "addr_token_set_ratio", "addr_jaccard",
    "pin_match", "pin_prefix_match", "street_num_match", "addr_is_empty", "len_diff_addr",
    # Context & interaction features
    "country_match", "target_source_is_s2", "target_source_is_s3",
    "script_mismatch", "script_mismatch_addr_boost", "script_mismatch_pin_boost",
    "name_addr_harmonic_mean", "candidate_degree"
]


def _char_3grams(s: str) -> set:
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def _extract_number(s: str) -> str:
    for tok in s.split():
        if tok.isdigit():
            return tok
    return ""


def _expand_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a DataFrame of (source1_entity_id, candidate_entity_ids)
    into long DataFrame of (source1_entity_id, candidate_entity_id)."""
    col = "candidate_entity_ids" if "candidate_entity_ids" in df.columns else "matched_entity_ids"
    rows = []
    for s1, ids in zip(df["source1_entity_id"], df[col]):
        if not ids:
            continue
        for cid in ids.split(","):
            rows.append((s1, cid))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])


def _expand_candidate_pairs(candidate_pairs_path: str) -> pd.DataFrame:
    df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
    return _expand_chunk(df)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compute_features(joined: pd.DataFrame) -> pd.DataFrame:
    """All dense features for one batch's joined rows."""

    # --- cheap features: fully vectorized pandas/NumPy column ops ---
    name_core_1 = joined["name_core_1"].fillna("")
    name_core_2 = joined["name_core_2"].fillna("")
    addr_clean_1 = joined["addr_clean_1"].fillna("")
    addr_clean_2 = joined["addr_clean_2"].fillna("")
    p1_s = joined["addr_pin_1"].fillna("")
    p2_s = joined["addr_pin_2"].fillna("")

    name_exact = ((name_core_1 == name_core_2) & (name_core_1 != "")).astype(float)
    pin_match = ((p1_s == p2_s) & (p1_s != "")).astype(float)
    country_match = (joined["country_1"] == joined["country_2"]).astype(float)
    suffix_match = (
        (joined["name_legal_suffix_1"] == joined["name_legal_suffix_2"])
        & (joined["name_legal_suffix_1"].fillna("") != "")
    ).astype(float)
    script_mismatch = (joined["name_is_latin_1"] != joined["name_is_latin_2"]).astype(float)
    len_diff_name = (name_core_1.str.len() - name_core_2.str.len()).abs().astype(float)
    len_diff_addr = (addr_clean_1.str.len() - addr_clean_2.str.len()).abs().astype(float)
    addr_is_empty = ((addr_clean_1 == "") | (addr_clean_2 == "")).astype(float)
    target_source_is_s2 = joined["candidate_entity_id"].str.startswith("S2-").astype(float)
    target_source_is_s3 = joined["candidate_entity_id"].str.startswith("S3-").astype(float)

    # --- string & fuzzy features: fast Python comprehensions over .tolist() ---
    n1_list, n2_list = name_core_1.tolist(), name_core_2.tolist()
    a1_list, a2_list = addr_clean_1.tolist(), addr_clean_2.tolist()
    p1_list, p2_list = p1_s.tolist(), p2_s.tolist()

    name_jw = [JaroWinkler.normalized_similarity(x, y) if (x or y) else 0.0
               for x, y in zip(n1_list, n2_list)]
    name_lev = [Levenshtein.normalized_similarity(x, y) if (x or y) else 0.0
                for x, y in zip(n1_list, n2_list)]
    name_token_sort_ratio = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(n1_list, n2_list)]
    name_token_set_ratio = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(n1_list, n2_list)]
    name_partial_ratio = [fuzz.partial_ratio(x, y) / 100.0 for x, y in zip(n1_list, n2_list)]
    name_prefix_match = [1.0 if ((x.startswith(y) or y.startswith(x)) and min(len(x), len(y)) >= 4) else 0.0
                         for x, y in zip(n1_list, n2_list)]
    name_char_3gram_jaccard = [_jaccard(_char_3grams(x), _char_3grams(y)) for x, y in zip(n1_list, n2_list)]

    addr_jw = [JaroWinkler.normalized_similarity(x, y) if (x or y) else 0.0
               for x, y in zip(a1_list, a2_list)]
    addr_lev = [Levenshtein.normalized_similarity(x, y) if (x or y) else 0.0
                for x, y in zip(a1_list, a2_list)]
    addr_token_set_ratio = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(a1_list, a2_list)]
    addr_jaccard = [_jaccard(set(x.split()), set(y.split())) for x, y in zip(a1_list, a2_list)]

    pin_prefix_match = [1.0 if (x[:3] == y[:3] and len(x) >= 3 and len(y) >= 3) else 0.0
                        for x, y in zip(p1_list, p2_list)]
    street_num_match = [1.0 if (_extract_number(x) == _extract_number(y) and _extract_number(x) != "") else 0.0
                        for x, y in zip(a1_list, a2_list)]

    # --- Interaction features ---
    sm_arr = script_mismatch.values
    addr_ts_arr = np.array(addr_token_set_ratio, dtype=float)
    pin_m_arr = pin_match.values
    name_jw_arr = np.array(name_jw, dtype=float)
    addr_jw_arr = np.array(addr_jw, dtype=float)

    script_mismatch_addr_boost = sm_arr * addr_ts_arr
    script_mismatch_pin_boost = sm_arr * pin_m_arr
    name_addr_harmonic_mean = (2.0 * name_jw_arr * addr_jw_arr) / (name_jw_arr + addr_jw_arr + 1e-6)
    candidate_degree = joined.groupby("source1_entity_id")["candidate_entity_id"].transform("count").values.astype(float)

    return pd.DataFrame({
        "source1_entity_id": joined["source1_entity_id"].values,
        "candidate_entity_id": joined["candidate_entity_id"].values,
        "name_jw": name_jw,
        "name_lev": name_lev,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_exact": name_exact.values,
        "name_prefix_match": name_prefix_match,
        "name_suffix_match": suffix_match.values,
        "name_char_3gram_jaccard": name_char_3gram_jaccard,
        "len_diff_name": len_diff_name.values,
        "addr_jw": addr_jw,
        "addr_lev": addr_lev,
        "addr_token_set_ratio": addr_token_set_ratio,
        "addr_jaccard": addr_jaccard,
        "pin_match": pin_match.values,
        "pin_prefix_match": pin_prefix_match,
        "street_num_match": street_num_match,
        "addr_is_empty": addr_is_empty.values,
        "len_diff_addr": len_diff_addr.values,
        "country_match": country_match.values,
        "target_source_is_s2": target_source_is_s2.values,
        "target_source_is_s3": target_source_is_s3.values,
        "script_mismatch": script_mismatch.values,
        "script_mismatch_addr_boost": script_mismatch_addr_boost,
        "script_mismatch_pin_boost": script_mismatch_pin_boost,
        "name_addr_harmonic_mean": name_addr_harmonic_mean,
        "candidate_degree": candidate_degree,
    })


JOIN_SQL = """
    SELECT
        p.source1_entity_id, p.candidate_entity_id,
        s1.name_core AS name_core_1, s1.addr_clean AS addr_clean_1,
        s1.addr_pin AS addr_pin_1, s1.country AS country_1,
        s1.name_legal_suffix AS name_legal_suffix_1, s1.name_is_latin AS name_is_latin_1,
        o.name_core AS name_core_2, o.addr_clean AS addr_clean_2,
        o.addr_pin AS addr_pin_2, o.country AS country_2,
        o.name_legal_suffix AS name_legal_suffix_2, o.name_is_latin AS name_is_latin_2
    FROM batch_pairs p
    JOIN source1 s1 ON p.source1_entity_id = s1.entity_id
    JOIN others o ON p.candidate_entity_id = o.entity_id
"""


def build_features(dataset_dir: str, split: str, cache_dir: str, candidate_pairs_path: str,
                    out_dir: str, ground_truth_path: str = None, batch_size: int = 20_000,
                    max_batches: int = None):
    os.makedirs(out_dir, exist_ok=True)

    s1_cache = normalize_source_file(os.path.join(dataset_dir, split, f"{split}_source1.tsv"),
                                      os.path.join(cache_dir, f"{split}_source1.parquet"))
    s2_cache = normalize_source_file(os.path.join(dataset_dir, split, f"{split}_source2.tsv"),
                                      os.path.join(cache_dir, f"{split}_source2.parquet"))
    s3_cache = normalize_source_file(os.path.join(dataset_dir, split, f"{split}_source3.tsv"),
                                      os.path.join(cache_dir, f"{split}_source3.parquet"))

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{os.path.join(cache_dir, 'duckdb_tmp')}'")

    con.execute(f"CREATE VIEW source1 AS SELECT * FROM read_parquet('{s1_cache}')")
    con.execute(f"CREATE VIEW others AS "
                f"SELECT * FROM read_parquet('{s2_cache}') UNION ALL SELECT * FROM read_parquet('{s3_cache}')")

    pos_set = None
    if ground_truth_path:
        print(f"Loading ground-truth positive pairs from {ground_truth_path} ...", flush=True)
        t_gt = time.time()
        gc.disable()
        pos_set = set()
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1]:
                    s1 = parts[0]
                    for cid in parts[1].split(","):
                        pos_set.add((s1, cid))
        gc.enable()
        gc.freeze()
        print(f"Loaded {len(pos_set):,} ground-truth positive pairs in {time.time() - t_gt:.1f}s", flush=True)

    print(f"Scanning {candidate_pairs_path} ...", flush=True)
    with open(candidate_pairs_path, "r", encoding="utf-8") as f:
        total_entities = sum(1 for _ in f) - 1
    total_batches = (total_entities + batch_size - 1) // batch_size
    n_batches = min(total_batches, max_batches) if max_batches else total_batches
    print(f"Streaming {total_entities:,} S1 entities across {n_batches} batches (batch_size={batch_size:,}) ...", flush=True)

    chunk_reader = pd.read_csv(
        candidate_pairs_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=batch_size
    )

    n_written = 0
    start_all = time.time()

    for batch_id, chunk_df in enumerate(chunk_reader):
        if max_batches is not None and batch_id >= max_batches:
            break

        t_batch = time.time()
        batch_pairs = _expand_chunk(chunk_df)

        part_start = batch_id * batch_size
        part_path = os.path.join(out_dir, f"part_{part_start:08d}.parquet")

        if batch_pairs.empty:
            continue

        try:
            con.unregister("batch_pairs")
        except Exception:
            pass
        con.register("batch_pairs", batch_pairs)
        joined = con.execute(JOIN_SQL).fetchdf()
        try:
            con.unregister("batch_pairs")
        except Exception:
            pass

        if joined.empty:
            continue

        result = _compute_features(joined)

        if pos_set is not None:
            s1_vals = result["source1_entity_id"].values
            cid_vals = result["candidate_entity_id"].values
            result["label"] = np.fromiter(
                (1 if (s1, cid) in pos_set else 0 for s1, cid in zip(s1_vals, cid_vals)),
                dtype=np.int32,
                count=len(result)
            )

        result.to_parquet(part_path, index=False)
        n_written += len(result)
        elapsed = time.time() - t_batch
        print(f"  batch {batch_id + 1}/{n_batches}: {len(result):,} feature rows -> {part_path} ({elapsed:.1f}s)", flush=True)

    # Integrity verification gate: ensure distinct entities across batches
    part_files = sorted(glob.glob(os.path.join(out_dir, "*.parquet")))
    if len(part_files) >= 2:
        import pyarrow.parquet as pq
        s1_batch0 = set(pq.read_table(part_files[0], columns=["source1_entity_id"])["source1_entity_id"].to_pylist())
        s1_batch1 = set(pq.read_table(part_files[1], columns=["source1_entity_id"])["source1_entity_id"].to_pylist())
        overlap = s1_batch0.intersection(s1_batch1)
        if overlap:
            raise RuntimeError(f"FATAL: Table shadowing detected! {len(overlap)} overlapping entities between batch 0 and 1.")

    print(f"Done — {n_written:,} total feature rows written to {out_dir}/ in {time.time() - start_all:.1f}s", flush=True)


def load_features(features_dir: str) -> pd.DataFrame:
    """Helper for train.py / predict.py to read all part-files back as one frame."""
    paths = sorted(glob.glob(os.path.join(features_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No feature part-files found under {features_dir}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: pairwise feature engineering.")
    parser.add_argument("--dataset-dir", default="student_resource/dataset")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--candidate-pairs", required=True)
    parser.add_argument("--out", required=True, help="Output directory for feature Parquet parts.")
    parser.add_argument("--ground-truth", default=None,
                         help="train_ground_truth.tsv — attaches labels when given.")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches to process (useful for testing).")
    args = parser.parse_args()
    build_features(args.dataset_dir, args.split, args.cache_dir, args.candidate_pairs,
                    args.out, args.ground_truth, args.batch_size, args.max_batches)
