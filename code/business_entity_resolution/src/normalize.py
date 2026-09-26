"""
normalize.py — shared name/address normalization used identically by blocking.py
and features.py, so blocking keys and similarity features are computed on the
exact same cleaned text.

Design notes based on inspecting the actual data:
- business_name sometimes appears in non-Latin scripts (Devanagari, Kannada, etc.)
  for India records, while the Source-1 reference is transliterated Latin script.
  We do NOT call any external transliteration API/service (disallowed by the
  challenge rules — no external lookups). Instead we detect a non-Latin string
  and fall back to address-driven blocking/features for that record; a rule-based,
  offline transliteration pass can be layered in later as a pure local library
  (no network calls) if it measurably helps recall.
- Addresses are unstructured and reordered (component order varies), sometimes
  empty, and contain landmark phrases ("Near X"), unit/apartment markers, and
  PIN/ZIP codes buried anywhere in the string.
- Legal suffixes (Pvt Ltd, LLC, Inc, Corp...) are noisy signal for equality but
  useful as a separate flag — we strip them into `legal_suffix` rather than
  discarding them.
"""

import os
import re
import time
import unicodedata

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# --- Static lookup tables -----------------------------------------------

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "private", "limited",
    "ltd", "llp", "llc", "l.l.c", "inc", "incorporated", "corp", "corporation",
    "co", "company", "enterprises", "enterprise", "associates", "group",
    "services", "solutions", "ventures",
]

ADDRESS_ABBREV = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard", r"\bdr\b": "drive", r"\bln\b": "lane",
    r"\bct\b": "court", r"\bapt\b": "apartment", r"\bste\b": "suite",
    r"\bunit\b": "unit", r"\bno\b": "number", r"\bhwy\b": "highway",
    r"\bpo box\b": "po box",
}

NAME_ABBREV = {
    r"&": "and",
    r"\bcorp\b": "corporation",
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
}

PIN_RE = re.compile(r"\b(\d{5,6}(?:-\d{4})?)\b")  # US ZIP or India PIN
LANDMARK_RE = re.compile(r"\bnear\b.*", re.IGNORECASE)


def is_latin_script(text: str) -> bool:
    """True if text is (almost) entirely Latin-script / ASCII-punctuation."""
    if not text:
        return True
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters) > 0.8


def _basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s&/-]", " ", text)   # drop stray punctuation, keep & / -
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(raw: str) -> dict:
    """Return normalized name fields used for both blocking and features.

    Keys:
      clean        - lowercased, punctuation-stripped name (suffix intact)
      core         - clean name with legal suffixes stripped
      tokens       - sorted token list of `core` (order-invariant compare)
      legal_suffix - the suffix removed, if any (else "")
      is_latin     - whether the raw string is Latin-script
    """
    is_latin = is_latin_script(raw)
    clean = _basic_clean(raw)
    for pat, repl in NAME_ABBREV.items():
        clean = re.sub(pat, repl, clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    core = clean
    found_suffix = ""
    for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
        pat = rf"\b{re.escape(suf)}\b"
        if re.search(pat, core):
            core = re.sub(pat, "", core).strip()
            found_suffix = found_suffix or suf
    core = re.sub(r"\s+", " ", core).strip()

    tokens = sorted(core.split())
    return {
        "clean": clean,
        "core": core,
        "tokens": tokens,
        "sorted_core": " ".join(tokens),
        "legal_suffix": found_suffix,
        "is_latin": is_latin,
    }


def normalize_address(raw: str) -> dict:
    """Return normalized address fields.

    Keys:
      clean     - lowercased, abbreviation-expanded, landmark-stripped address
      tokens    - sorted token list (order-invariant compare)
      pin       - extracted postal/ZIP code if any (else "")
      has_landmark - whether a "near X" style landmark phrase was present
    """
    if not isinstance(raw, str) or not raw.strip():
        return {"clean": "", "tokens": [], "sorted_core": "", "pin": "", "has_landmark": False}

    has_landmark = bool(LANDMARK_RE.search(raw))
    text = LANDMARK_RE.sub("", raw)

    pin_match = PIN_RE.search(text)
    pin = pin_match.group(1) if pin_match else ""

    clean = _basic_clean(text)
    for pat, repl in ADDRESS_ABBREV.items():
        clean = re.sub(pat, repl, clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    tokens = sorted(clean.split())
    return {
        "clean": clean,
        "tokens": tokens,
        "sorted_core": " ".join(tokens),
        "pin": pin,
        "has_landmark": has_landmark,
    }


def normalize_record(business_name: str, business_address: str, country: str) -> dict:
    """Normalize one full record. `country` is passed through untouched as a
    plain string field — never hard-coded/one-hot to a fixed set, since the
    test set introduces a country (France) absent from training."""
    name_feats = normalize_name(business_name)
    addr_feats = normalize_address(business_address)
    return {
        "name": name_feats,
        "address": addr_feats,
        "country": (country or "").strip(),
    }


# --- Batch normalization over full source files, cached to Parquet ------

def _row_normalize(row: pd.Series) -> pd.Series:
    rec = normalize_record(row["business_name"], row["business_address"], row["country"])
    return pd.Series({
        "name_clean": rec["name"]["clean"],
        "name_core": rec["name"]["sorted_core"],
        "name_is_latin": rec["name"]["is_latin"],
        "name_legal_suffix": rec["name"]["legal_suffix"],
        "addr_clean": rec["address"]["sorted_core"],
        "addr_pin": rec["address"]["pin"],
        "addr_has_landmark": rec["address"]["has_landmark"],
        "country": rec["country"],
    })


def normalize_source_file(input_tsv: str, cache_parquet: str, chunksize: int = 200_000,
                           force: bool = False) -> str:
    """Normalize a raw source TSV in chunks (bounded memory) and cache the result
    as Parquet alongside entity_id and the normalized fields. Skips recomputation
    if the cache already exists, unless force=True. Returns the cache path.

    Streams output chunk-by-chunk via a ParquetWriter (rather than holding every
    chunk in memory and writing once at the end) — this keeps peak memory flat
    regardless of file size AND gives real per-chunk progress, since this step
    is the slowest part of the pipeline for the ~470-480MB Source-2/3 files
    (a few regex passes per row, run row-by-row in Python). Writes to a .tmp
    path and renames on success only, so a killed/failed run never leaves a
    corrupt file that a later run would mistake for a valid cache and skip.

    Used by both blocking.py (candidate generation) and features.py (similarity
    features), so both stages always operate on identically normalized text.
    """
    if os.path.exists(cache_parquet) and not force:
        return cache_parquet

    os.makedirs(os.path.dirname(cache_parquet) or ".", exist_ok=True)
    tmp_path = cache_parquet + ".tmp"

    print(f"  normalizing {input_tsv} ...")
    start = time.time()
    rows_done = 0
    writer = None
    try:
        for i, chunk in enumerate(pd.read_csv(input_tsv, sep="\t", chunksize=chunksize,
                                               dtype=str, keep_default_na=False)):
            norm_cols = chunk.apply(_row_normalize, axis=1)
            out_chunk = pd.concat([chunk[["entity_id"]], norm_cols], axis=1)

            table = pa.Table.from_pandas(out_chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table)

            rows_done += len(out_chunk)
            elapsed = time.time() - start
            rate = rows_done / elapsed if elapsed > 0 else 0
            print(f"    chunk {i + 1}: {rows_done:,} rows normalized "
                  f"({rate:,.0f} rows/sec, {elapsed:,.0f}s elapsed)")
    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_path, cache_parquet)
    print(f"  done: {rows_done:,} rows -> {cache_parquet} ({time.time() - start:,.0f}s total)")
    return cache_parquet


if __name__ == "__main__":
    # Quick smoke test against a few real examples seen in the data.
    samples = [
        ("Consulting Nyasa Nursing Private Limited",
         "2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, "
         "Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra", "India"),
        ("-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC", "US"),
        ("Pvt. EFS Print Ventures Ltd.", "Door No 183, 41St Cross, Bengaluru", "India"),
    ]
    for name, addr, country in samples:
        rec = normalize_record(name, addr, country)
        print(rec["name"]["sorted_core"], "|", rec["address"]["pin"], "|", rec["address"]["sorted_core"])
