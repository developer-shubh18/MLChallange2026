"""
blocking.py — candidate generation (Stage 2).

Pipeline:
  1. Normalize each raw source file once, cached to Parquet (via normalize.py).
  2. Load the normalized sources into DuckDB.
  3. Build candidates from three unioned strategies:
       a) exact match on the normalized, suffix-stripped, token-sorted name key
          (name_core)                                    — cheap, high precision
       b) exact match on extracted PIN/ZIP code (addr_pin) — catches same-location
          matches even with a badly-mangled name, and is the main signal for
          non-Latin-script names (addresses are usually still Latin/roman script
          even when the business name isn't — see the India Source-2/3 samples)
       c) shared rare/discriminating token match — catches fuzzy variants (typos,
          abbreviation differences, word-order transpositions) that (a) and (b)
          miss. For each record we keep only its TOP_K_TOKENS least-frequent
          tokens (name + address tokens combined) as blocking keys; rare tokens
          are the most discriminating, and keeping only a few per record bounds
          candidate-set size even for records built from generic words. Tokens
          that are globally too common (> MAX_TOKEN_DF occurrences) are dropped
          entirely as stopword-like noise.
  4. Emit candidate_pairs.tsv in the exact submission format: one row per
     Source-1 entity_id, comma-separated S2-/S3- IDs, empty when none found.

Tune TOP_K_TOKENS / MAX_TOKEN_DF against your measured blocking recall (see the
project README for how to check recall on a held-out validation split) — if
recall against S2/S3 is low for a source, first check whether MAX_TOKEN_DF is
dropping too many tokens, or raise TOP_K_TOKENS.
"""

import argparse
import os

# pyrefly: ignore [missing-import]
import duckdb

from normalize import normalize_source_file

TOP_K_TOKENS = 2
MAX_TOKEN_DF = 2_000     # tokens appearing in >2K records are stopword-like noise
MAX_BLOCK_FREQ = 20      # max records per blocking key per source side
MAX_NAME_FREQ = 100      # relaxed frequency cap for exact name matches
MIN_PIN_LEN = 4          # ignore very short PINs


def _tokenize_sql(con, table):
    """Explode name_core + addr_clean into (entity_id, country, token) rows."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table}_tokens AS
        SELECT entity_id, country,
               unnest(
                   list_filter(
                       list_distinct(
                           list_concat(
                               string_split(name_core, ' '),
                               string_split(addr_clean, ' ')
                           )
                       ),
                       x -> x != ''
                   )
               ) AS token
        FROM {table}
    """)


def _pick_blocking_tokens(con, table):
    """Keep only the TOP_K_TOKENS rarest tokens per record as blocking keys."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table}_df AS
        SELECT token, COUNT(*) AS df FROM {table}_tokens GROUP BY token
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table}_blockkeys AS
        SELECT entity_id, country, token FROM (
            SELECT t.entity_id, t.country, t.token,
                   ROW_NUMBER() OVER (PARTITION BY t.entity_id ORDER BY d.df ASC) AS rn
            FROM {table}_tokens t
            JOIN {table}_df d USING (token)
            WHERE d.df <= {MAX_TOKEN_DF}
        )
        WHERE rn <= {TOP_K_TOKENS}
    """)
    con.execute(f"DROP TABLE IF EXISTS {table}_tokens")
    con.execute(f"DROP TABLE IF EXISTS {table}_df")


def build_candidates_to_parquet(con, s1_table, other_table, out_parquet, tmp_dir):
    """Build candidate pairs and write deduplicated result to a parquet file.

    Everything is disk-based — each strategy writes to parquet via COPY,
    the final UNION+DISTINCT reads from those parquet files and writes a
    combined result.  Nothing large ever enters pandas."""
    os.makedirs(tmp_dir, exist_ok=True)

    _tokenize_sql(con, s1_table)
    _tokenize_sql(con, other_table)
    _pick_blocking_tokens(con, s1_table)
    _pick_blocking_tokens(con, other_table)

    tag = f"{s1_table}_{other_table}"
    name_pq = os.path.join(tmp_dir, f"cand_name_{tag}.parquet")
    pin_pq  = os.path.join(tmp_dir, f"cand_pin_{tag}.parquet")
    tok_pq  = os.path.join(tmp_dir, f"cand_tok_{tag}.parquet")

    # --- Strategy 1: exact normalized-name match (country-gated, frequency-capped) ---
    con.execute(f"""
        COPY (
            SELECT DISTINCT a.entity_id AS source1_entity_id,
                            b.entity_id AS candidate_entity_id
            FROM {s1_table} a
            JOIN {other_table} b ON a.name_core = b.name_core AND a.country = b.country
            WHERE a.name_core != ''
              AND a.name_core IN (
                  SELECT name_core FROM {s1_table}
                  WHERE name_core != ''
                  GROUP BY name_core HAVING COUNT(*) <= {MAX_NAME_FREQ}
              )
              AND a.name_core IN (
                  SELECT name_core FROM {other_table}
                  WHERE name_core != ''
                  GROUP BY name_core HAVING COUNT(*) <= {MAX_NAME_FREQ}
              )
        ) TO '{name_pq}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{name_pq}')").fetchone()[0]
    print(f"    name_core  : {cnt:>10,} pairs")

    # --- Strategy 2: PIN / ZIP match (country-gated, frequency-capped, min-length) ---
    con.execute(f"""
        COPY (
            SELECT DISTINCT a.entity_id AS source1_entity_id,
                            b.entity_id AS candidate_entity_id
            FROM {s1_table} a
            JOIN {other_table} b ON a.addr_pin = b.addr_pin AND a.country = b.country
            WHERE a.addr_pin != ''
              AND length(a.addr_pin) >= {MIN_PIN_LEN}
              AND a.addr_pin IN (
                  SELECT addr_pin FROM {s1_table}
                  WHERE addr_pin != '' AND length(addr_pin) >= {MIN_PIN_LEN}
                  GROUP BY addr_pin HAVING COUNT(*) <= {MAX_BLOCK_FREQ}
              )
              AND a.addr_pin IN (
                  SELECT addr_pin FROM {other_table}
                  WHERE addr_pin != '' AND length(addr_pin) >= {MIN_PIN_LEN}
                  GROUP BY addr_pin HAVING COUNT(*) <= {MAX_BLOCK_FREQ}
              )
        ) TO '{pin_pq}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pin_pq}')").fetchone()[0]
    print(f"    addr_pin   : {cnt:>10,} pairs")

    # --- Strategy 3: rare-token match (country-gated, frequency-capped) ---
    con.execute(f"""
        COPY (
            SELECT DISTINCT a.entity_id AS source1_entity_id,
                            b.entity_id AS candidate_entity_id
            FROM {s1_table}_blockkeys a
            JOIN {other_table}_blockkeys b ON a.token = b.token AND a.country = b.country
            WHERE a.token IN (
                SELECT token FROM {s1_table}_blockkeys
                GROUP BY token HAVING COUNT(*) <= {MAX_BLOCK_FREQ}
            )
            AND a.token IN (
                SELECT token FROM {other_table}_blockkeys
                GROUP BY token HAVING COUNT(*) <= {MAX_BLOCK_FREQ}
            )
        ) TO '{tok_pq}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tok_pq}')").fetchone()[0]
    print(f"    token_match: {cnt:>10,} pairs")

    # Clean up blockkeys
    con.execute(f"DROP TABLE IF EXISTS {s1_table}_blockkeys")
    con.execute(f"DROP TABLE IF EXISTS {other_table}_blockkeys")

    # --- Combine & deduplicate from parquet → parquet ---
    con.execute(f"""
        COPY (
            SELECT DISTINCT source1_entity_id, candidate_entity_id FROM (
                SELECT * FROM read_parquet('{name_pq}')
                UNION ALL SELECT * FROM read_parquet('{pin_pq}')
                UNION ALL SELECT * FROM read_parquet('{tok_pq}')
            )
        ) TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_parquet}')").fetchone()[0]
    print(f"    TOTAL      : {total:>10,} pairs (deduplicated)")

    # Clean up intermediate parquet files
    for f in [name_pq, pin_pq, tok_pq]:
        if os.path.exists(f):
            os.remove(f)

    return total

def write_candidate_pairs_tsv(out_path, s1_cache, s2_parquet, s3_parquet):
    """Write the final candidate_pairs.tsv using pure Python + PyArrow.

    DuckDB's STRING_AGG is too memory-hungry for this machine.  Instead we:
      1. Stream-read candidate parquets in batches via PyArrow
      2. Accumulate candidates in a plain dict (~1.5 GB)
      3. Read source1 entity IDs and write TSV line-by-line"""
    import pyarrow.parquet as pq
    from collections import defaultdict

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- Build candidate dict by streaming parquet batches ---
    candidates = defaultdict(list)
    for pq_path in [s2_parquet, s3_parquet]:
        pf = pq.ParquetFile(pq_path)
        for batch in pf.iter_batches(
            batch_size=500_000,
            columns=["source1_entity_id", "candidate_entity_id"],
        ):
            s1_ids = batch.column("source1_entity_id").to_pylist()
            cand_ids = batch.column("candidate_entity_id").to_pylist()
            for s1_id, cand_id in zip(s1_ids, cand_ids):
                candidates[s1_id].append(cand_id)

    n_with = len(candidates)
    print(f"    {n_with:,} S1 entities have candidates")

    # --- Read source1 entity IDs and write TSV ---
    s1_table = pq.read_table(s1_cache, columns=["entity_id"])
    s1_ids = s1_table.column("entity_id").to_pylist()
    n_total = len(s1_ids)

    with open(out_path, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cands = candidates.get(s1_id)
            if cands:
                cand_str = ",".join(sorted(set(cands)))
            else:
                cand_str = ""
            f.write(f"{s1_id}\t{cand_str}\n")

    print(f"Wrote {out_path} — {n_with:,} / {n_total:,} S1 entities have at least one candidate")


def run(dataset_dir: str, split: str, cache_dir: str, out_path: str):
    os.makedirs(cache_dir, exist_ok=True)
    tmp_dir = os.path.join(cache_dir, "blocking_tmp")

    s1_path = os.path.join(dataset_dir, split, f"{split}_source1.tsv")
    s2_path = os.path.join(dataset_dir, split, f"{split}_source2.tsv")
    s3_path = os.path.join(dataset_dir, split, f"{split}_source3.tsv")

    print("Normalizing sources (cached — skipped on re-run)...")
    s1_cache = normalize_source_file(s1_path, os.path.join(cache_dir, f"{split}_source1.parquet"))
    s2_cache = normalize_source_file(s2_path, os.path.join(cache_dir, f"{split}_source2.parquet"))
    s3_cache = normalize_source_file(s3_path, os.path.join(cache_dir, f"{split}_source3.parquet"))

    # --- Blocking phase: needs source tables in DuckDB ---
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{os.path.join(cache_dir, 'duckdb_tmp')}'")

    con.execute(f"CREATE TABLE source1 AS SELECT * FROM read_parquet('{s1_cache}')")
    con.execute(f"CREATE TABLE source2 AS SELECT * FROM read_parquet('{s2_cache}')")
    con.execute(f"CREATE TABLE source3 AS SELECT * FROM read_parquet('{s3_cache}')")

    s2_parquet = os.path.join(cache_dir, f"{split}_candidates_s2.parquet")
    s3_parquet = os.path.join(cache_dir, f"{split}_candidates_s3.parquet")

    print("Blocking against Source 2 ...")
    build_candidates_to_parquet(con, "source1", "source2", s2_parquet, tmp_dir)
    con.execute("DROP TABLE IF EXISTS source2")

    print("Blocking against Source 3 ...")
    build_candidates_to_parquet(con, "source1", "source3", s3_parquet, tmp_dir)
    con.execute("DROP TABLE IF EXISTS source3")

    # Close blocking connection — free ALL DuckDB memory
    con.execute("DROP TABLE IF EXISTS source1")
    con.close()
    print("Blocking complete. Freed DuckDB memory.")

    # --- TSV output phase: fresh connection, no tables loaded ---
    print("Writing candidate_pairs.tsv ...")
    write_candidate_pairs_tsv(out_path, s1_cache, s2_parquet, s3_parquet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: candidate generation (blocking).")
    parser.add_argument("--dataset-dir", default="student_resource/dataset",
                         help="Folder containing train/ and test/ subfolders.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--cache-dir", default="cache",
                         help="Where normalized Parquet caches are stored/read.")
    parser.add_argument("--out", required=True, help="Output candidate_pairs.tsv path.")
    args = parser.parse_args()
    run(args.dataset_dir, args.split, args.cache_dir, args.out)

