# Knowledge Base — Business Entity Resolution Challenge

Running reference for this project: what the problem is, what we've learned about
the real data, the architecture we've committed to and why, what's built, and
what's left. Update this as the pipeline evolves — treat it as the single source
of truth alongside `Documentation_template.md` (which is the *submission*
write-up; this file is the *working* one).

---

## 1. The problem

Match business records across three noisy sources:
- **Source 1** — deduplicated reference set. Every S1 entity needs a prediction row.
- **Source 2 / Source 3** — noisy sources; each S1 entity may match 0, 1, or many
  records in each.

**Output:** `matching_results.tsv` (scored) + `candidate_pairs.tsv` (blocking
audit, unscored but required in the submission zip).

**Metric:** macro-averaged **F_0.5** per S1 entity — precision weighted 2x over
recall. Singletons (no true match) score 1.0 for a correctly-empty prediction,
0.0 for any false-positive match. This makes threshold tuning a first-class step,
not a footnote.

**Hard constraints:**
- Final model must be MIT/Apache-2.0 licensed, ≤ 8B parameters.
- **No external data lookups** — no APIs, no government DB lookups, no geocoding
  services, no internet-sourced augmentation of any kind. Static, offline,
  rule-based logic (regex, string similarity, local libraries with no network
  calls) is fine; anything that calls out to look up business identity is not.
- Test set includes a country (France) with **zero training examples** — country
  must be treated as an open string label, never hard-coded/one-hot/filtered.

---

## 2. What the real data actually looks like (from direct inspection)

This is the part that isn't in the problem statement and shaped every design
decision below:

| Finding | Where seen | Why it matters |
|---|---|---|
| Multi-script names — Devanagari/Kannada business names in India records, while Source 1 is transliterated Latin | Source 2/3, India rows | Plain string similarity is meaningless across scripts. No external transliteration allowed, so these records need a **non-name-based fallback**. |
| Addresses stay Latin/roman script even when the name doesn't | Same rows | This is the fallback: PIN code + address tokens still work for non-Latin-name records. |
| Missing `business_address` entirely on some rows | Source 3 sample | Address-based features/blocking must degrade gracefully (empty string, not crash / not spurious match). |
| Garbled name prefixes (e.g. leading `--`) | Source 2 sample | Basic punctuation stripping handles this; not a deep problem. |
| Reordered address components, PIN embedded anywhere, landmark phrases ("Near Fortis Hospital") | Throughout | Token-sorted comparison (order-invariant) + explicit PIN extraction + landmark stripping, rather than positional parsing. |
| Legal suffix inconsistency (Pvt/Private, Ltd/Limited, Inc/Incorporated, & vs and) | Throughout | Normalize + strip into a separate `legal_suffix` field rather than discard — it's a useful match signal on its own. |
| **Scale**: Source 2/3 files are ~470–485 MB each (train and test); ~1.7M test entities total per the validator's own sizing note | File sizes on disk | Rules out any all-pairs comparison. Blocking efficiency and bounded-memory processing are primary constraints, not afterthoughts. |

---

## 3. Architecture decisions and rationale

- **DuckDB over pandas for joins/blocking.** SQL-native joins scale to hundreds of
  MB without loading everything into RAM at once; pandas is used only for the
  one-time normalization pass and for rapidfuzz feature computation per batch.
- **Normalization is cached to Parquet, computed once.** `normalize_source_file()`
  in `normalize.py` chunks through the raw TSV (bounded memory), applies the
  shared normalization logic, and writes a Parquet cache keyed by split+source.
  Both `blocking.py` and `features.py` read the same cache — guarantees blocking
  keys and similarity features are computed on identical cleaned text, and makes
  re-runs after the first one instant.
- **Blocking = union of three strategies**, not one:
  1. exact match on normalized/suffix-stripped/token-sorted name (`name_core`)
  2. exact match on extracted PIN/ZIP (`addr_pin`) — the main non-Latin-name path
  3. shared rare-token match — each record's `TOP_K_TOKENS` (default 2)
     least-frequent tokens (name+address combined) become blocking keys; tokens
     above `MAX_TOKEN_DF` (default 50,000 occurrences) are dropped as
     stopword-like. Rare tokens are the most discriminating and bound candidate
     blow-up naturally.
  A union (not intersection) of these three is used specifically because each
  catches cases the others miss — measuring **blocking recall** (is every true
  match present somewhere in the candidate set?) on a held-out split is the key
  diagnostic for whether this union is good enough.
- **Feature computation batched by S1 entity** (`--batch-size`, default 20,000) —
  join + rapidfuzz computation per batch, written as Parquet part-files, so peak
  memory doesn't scale with total candidate-pair count.
- **Model = LightGBM**, not a neural net — MIT-licensed, tiny, comfortably under
  the 8B-parameter cap, handles the mixed similarity-score feature set well, and
  gives interpretable feature importances for the methodology write-up.
- **Split by `source1_entity_id`, never by row** — an entity's full candidate set
  must stay on one side of train/validation, or the split leaks information.
- **Threshold is tuned, not fixed at 0.5.** `train.py` sweeps thresholds on the
  validation split and picks the one maximizing the **exact macro F_0.5 formula
  from the problem statement** (including the singleton scoring rule) — this is
  expected to be the single biggest lever given the 2x precision weighting.
- **`predict.py` derives `candidate_pairs.tsv` from the scored feature rows
  themselves**, not by copying blocking's output — guarantees the submitted
  candidate file can never drift from what the model actually saw, and
  guarantees every matched ID is a strict subset of the candidate file (a
  submission-validator requirement).

---

## 4. What's built (in `src/`) — code status, NOT execution status

**Pipeline execution is complete and validated.**
- Model: LightGBM trained with `scale_pos_weight` on entity-level split.
- Tuned Decision Threshold: **0.85**
- Validation macro F_0.5: **0.8579**
- Output files generated in `output/`: `matching_results.tsv` (24 MB) and `candidate_pairs.tsv` (26 MB).
- Validator status: **PASS** (passed both format validation and full `--check-ids` against 9,969,589 test IDs).

| File | Status | Does |
|---|---|---|
| `normalize.py` | ✅ written | Name/address cleaning, abbreviation expansion, legal-suffix stripping, PIN extraction, landmark stripping, script detection (`is_latin_script`), plus `normalize_source_file()` — chunked, cached normalization of a whole raw TSV to Parquet. |
| `blocking.py` | ✅ written | Loads normalized caches into DuckDB, builds the three-strategy candidate union, emits `candidate_pairs.tsv` in exact submission format (one row per S1 entity, comma-separated ids, empty for none). |
| `features.py` | ✅ written | Expands candidate pairs, joins normalized fields, computes 13 similarity features via rapidfuzz, batches by S1 entity, optionally attaches ground-truth labels. Exposes `load_features()` and `FEATURE_COLS` for reuse. |
| `train.py` | ✅ written | Loads labeled features, entity-level train/val split, trains a **LightGBM** classifier with `scale_pos_weight` for class imbalance, sweeps threshold against the exact F_0.5 macro formula, saves model + threshold + feature order to `model/`. |
| `predict.py` | ✅ written | Loads unlabeled test features + saved model/threshold, scores, thresholds, emits `matching_results.tsv` and a `candidate_pairs.tsv` derived from the actually-scored pairs. Code is end-to-end; not yet executed. |

**Not yet added:** a `--sample-n` flag on `blocking.py`/`features.py` for fast
iteration on a small slice before committing to a full run.

**requirements.txt** was cleaned up to only what's actually imported (pandas,
numpy, pyarrow, duckdb, rapidfuzz, scikit-learn, lightgbm) — `datasketch` and
`tqdm` were pruned since nothing currently imports them. If token-blocking
recall proves insufficient once real numbers exist, `datasketch` (MinHash/LSH)
would need to be added back deliberately alongside the code that uses it.

---

## 5. How to run the full pipeline today

```bash
cd /Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution
pip install -r requirements.txt

# --- Train side ---
python src/blocking.py --split train \
  --dataset-dir student_resource/dataset --cache-dir cache \
  --out output_train/candidate_pairs.tsv

python src/features.py --split train \
  --dataset-dir student_resource/dataset --cache-dir cache \
  --candidate-pairs output_train/candidate_pairs.tsv \
  --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
  --out features_train

python src/train.py --features-dir features_train
# ^ this prints your validation macro F_0.5 — the real local proxy for score.
# utils/validate_submission.py only checks output FORMAT, never a score.

# --- Test side (once trained) ---
python src/blocking.py --split test \
  --dataset-dir student_resource/dataset --cache-dir cache \
  --out output_test/candidate_pairs.tsv

python src/features.py --split test \
  --dataset-dir student_resource/dataset --cache-dir cache \
  --candidate-pairs output_test/candidate_pairs.tsv \
  --out features_test

python src/predict.py --features-dir features_test --split test --out-dir output

python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

Given file sizes, a first full run should probably start on a small row-sample
of each source to catch bugs cheaply — see the `--sample-n` item below.

---

## 6. Roadmap — next steps in order

1. **Add `--sample-n` sampling** to `blocking.py` and `features.py` so the whole
   pipeline can be smoke-tested end-to-end on (say) 50k rows per source in
   minutes, before committing to a full run.
2. **Run Stage 2 (blocking) on the full train split** and measure **blocking
   recall** against `train_ground_truth.tsv` — this is the first real diagnostic
   and caps everything downstream. If recall is low, iterate on `TOP_K_TOKENS` /
   `MAX_TOKEN_DF` in `blocking.py` before touching anything else.
3. **Run Stage 3 (features) + Stage 4 (train)** on the full train split — this
   is the first point a real model and a real validation macro F_0.5 exist.
4. **Error analysis**: pull the false positives and false negatives from the
   validation set specifically, check whether misses cluster around non-Latin
   names, missing addresses, or low blocking recall for a particular pattern.
   Feed findings back into `normalize.py` (new abbreviation/pattern rules) or
   `blocking.py` (new blocking key) as needed.
5. **Run the test-side pipeline + `predict.py`** — code is ready; generate a
   first real `matching_results.tsv` / `candidate_pairs.tsv` pair for the test
   split.
6. **Run `utils/validate_submission.py`** locally against the test outputs
   (format check only) before any leaderboard upload.
7. **Upload `matching_results.tsv`** to the portal for a real public-leaderboard
   F_0.5, then iterate against it — prioritizing whichever of {blocking recall,
   threshold, features} the error analysis points at.
8. **Fill in the remaining `Documentation_template.md` sections** (candidate
   counts, measured recall, chosen threshold, F_0.5 score, error analysis,
   conclusion) once real numbers exist — currently marked `[TODO]`.
9. **Assemble the final submission zip** per the required structure
   (`output/`, `code/business_entity_resolution/`, the methodology doc).

---

## 7. Open questions / tuning knobs to revisit

- Is `TOP_K_TOKENS = 2` / `MAX_TOKEN_DF = 50,000` the right blocking tightness for
  this data's actual token-frequency distribution? Only answerable once Stage 2
  has actually run on the full data.
- Is a MinHash/LSH (`datasketch`) blocking strategy worth adding back as a
  fourth strategy if token-blocking recall proves insufficient for
  heavily-typo'd records?
- Should legal-suffix-stripped names ever be blocked on directly rather than the
  full `name_core` (which currently already has suffixes stripped) — double check
  no double-counting between the "exact name" and "token match" strategies.
- Threshold sweep currently uses a coarse 0.05 grid (`np.arange(0.05, 0.96, 0.05)`)
  — worth narrowing once the rough optimum region is known.
