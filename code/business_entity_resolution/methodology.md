# Methodology — Business Entity Resolution

A technical write-up of the approach: what we're solving, how the pipeline is
structured, and why each design choice was made. This is the narrative
counterpart to `KNOWLEDGE_BASE.md` (which tracks live status/roadmap) and
`student_resource/Documentation_template.md` (the challenge's required
submission format) — this file is the standalone technical reference.

---

## 1. Problem framing

Given three noisy business-record sources (S1 = deduplicated reference, S2/S3 =
noisy), find every S2/S3 record that refers to the same real-world business as
each S1 record. An S1 entity may have zero, one, or many true matches.

We frame this as **blocking + pairwise binary classification**:
1. **Blocking** narrows the O(|S1| × |S2∪S3|) comparison space down to a
   manageable candidate set per S1 entity, without discarding true matches.
2. **Classification** scores each (S1, candidate) pair and a **tuned decision
   threshold** converts scores into final match/no-match calls.

This decomposition is standard for entity resolution at scale, and lets each
stage be measured and improved independently: blocking is judged by *recall*
(does the true match survive blocking?), and classification is judged by the
challenge's own **macro F_0.5** metric.

---

## 2. Data analysis

Direct inspection of the training files (not just the problem statement)
surfaced noise patterns that shaped the pipeline:

- **Multi-script names.** India records in S2/S3 sometimes appear in
  Devanagari or Kannada script, while S1 is transliterated to Latin script.
  String similarity is meaningless across scripts, and external transliteration
  services are disallowed by the challenge rules, so these records need a
  fallback that doesn't depend on the name field.
- **Addresses stay Latin/roman-script** even when the business name doesn't —
  this is the fallback: PIN/ZIP code and address tokens remain comparable even
  for non-Latin-name records.
- **Missing addresses** on some S2/S3 rows — features and blocking keys must
  degrade to empty/neutral rather than error or spuriously match.
- **Unstructured, reordered addresses** — component order varies freely, PIN
  codes are embedded anywhere in the string, and landmark phrases ("Near
  Fortis Hospital") are common. This rules out positional address parsing in
  favor of token-set (order-invariant) comparison plus explicit PIN
  extraction and landmark stripping.
- **Legal-suffix inconsistency** (Pvt/Private, Ltd/Limited, Inc/Incorporated,
  Corp/Corporation, & vs "and") — normalized and stripped into a separate
  field rather than discarded, since suffix agreement is itself a useful
  (weak) signal.
- **Open-set country label** — the test set introduces France, absent from
  training. Country is therefore used only as a soft equality feature, never
  a hard filter or a fixed category.
- **Scale** — S2/S3 files run several hundred MB each, with roughly 1.7M test
  entities overall. This makes O(n×m) all-pairs comparison infeasible and
  makes blocking efficiency and bounded-memory processing primary engineering
  constraints, not implementation details.

---

## 3. Preprocessing / normalization

A single shared normalization module (`normalize.py`) is used identically by
both the blocking and feature stages, so blocking keys and similarity features
are always computed on the same cleaned text — this consistency matters more
than any individual cleaning rule.

**Name normalization:**
- Lowercase, strip punctuation (keeping `&`, `/`, `-`).
- Expand common abbreviations (`&` → `and`, `Corp` → `corporation`, `Pvt` →
  `private`, `Ltd` → `limited`, etc.).
- Strip a recognized legal suffix into a separate `legal_suffix` field.
- Tokenize and sort tokens (`sorted_core`) for order-invariant comparison
  (handles word-order transpositions).
- Flag whether the string is Latin-script (`is_latin`), via a
  Unicode-category check on alphabetic characters — used to route non-Latin
  records toward address-based signals instead of name-based ones.

**Address normalization:**
- Strip landmark phrases (`near ...`) into a separate `has_landmark` flag.
- Extract a PIN/ZIP code via regex (5–6 digit sequences, optionally with a
  ZIP+4 suffix), regardless of position in the string.
- Lowercase, expand abbreviations (`Rd` → `road`, `St` → `street`, etc.).
- Tokenize and sort tokens for order-invariant comparison.

**Batch processing:** raw TSVs are streamed in chunks (bounded memory),
normalized, and cached to Parquet once per (split, source) — both later stages
read this cache, so normalization is never recomputed and stays consistent
across the pipeline.

---

## 4. Candidate generation (blocking)

Blocking uses a **union of three independent strategies**, run via DuckDB SQL
directly against the normalized Parquet caches (no full in-memory pandas
join):

1. **Exact normalized-name match** — join on the suffix-stripped, token-sorted
   name key. Cheap, high-precision, catches straightforward duplicates and
   reordered-word variants.
2. **Exact PIN/ZIP match** — join on the extracted postal code. This is the
   primary recall path for non-Latin-script names, since the address usually
   remains comparable even when the name doesn't.
3. **Shared rare-token match** — for every record, its name+address tokens are
   ranked by global document frequency within the source being blocked
   against; the `TOP_K_TOKENS` (default 2) rarest tokens become that record's
   blocking keys, after dropping any token that appears in more than
   `MAX_TOKEN_DF` (default 50,000) records as stopword-like noise. Two records
   sharing a rare token are joined as candidates. This catches typos,
   abbreviation differences, and partial-name/address matches that the first
   two strategies miss, while bounding candidate-set size — rare tokens are by
   construction the most discriminating, so this doesn't blow up combinatorially
   even for records built from generic vocabulary.

The three strategies are **unioned, not intersected**, because each recovers
matches the others miss; the combined strategy's recall is measured directly
on a held-out validation split (fraction of ground-truth matches present
anywhere in the candidate set), which is the primary blocking-quality
diagnostic and caps everything downstream.

---

## 5. Feature engineering

For every (S1, candidate) pair that survives blocking, 13 features are
computed from the normalized fields, using `rapidfuzz` for the string metrics:

| Feature | Description |
|---|---|
| `name_jw` | Jaro-Winkler similarity, normalized name |
| `name_lev` | Normalized Levenshtein similarity, normalized name |
| `name_jaccard` | Token-set Jaccard overlap, name |
| `name_exact` | 1.0 if normalized names are identical (and non-empty) |
| `addr_jw` | Jaro-Winkler similarity, normalized address |
| `addr_lev` | Normalized Levenshtein similarity, normalized address |
| `addr_jaccard` | Token-set Jaccard overlap, address |
| `pin_match` | 1.0 if extracted PIN/ZIP codes match (and non-empty) |
| `country_match` | 1.0 if country strings match exactly (soft signal, never a filter) |
| `suffix_match` | 1.0 if stripped legal suffixes match (and non-empty) |
| `script_mismatch` | 1.0 if one name is Latin-script and the other isn't |
| `len_diff_name` | Absolute character-length difference, normalized name |
| `len_diff_addr` | Absolute character-length difference, normalized address |

Feature computation is batched by S1 entity (default 20,000 entities per
batch) so peak memory is bounded regardless of total candidate-pair count;
each batch is written as a separate Parquet part-file.

---

## 6. Matching model

**Model:** LightGBM (gradient-boosted decision trees) — MIT-licensed, a few
hundred KB to low MB in size (comfortably under the challenge's 8B-parameter
ceiling), handles a small set of mixed similarity-score features well, trains
in minutes even at scale, and produces interpretable feature importances.

**Training data:** every blocked (S1, candidate) pair is labeled 1 (true
match, from `train_ground_truth.tsv`) or 0 (not a true match). Because
candidates already passed blocking, negative examples are naturally
**hard negatives** — pairs plausible enough to survive blocking but not truly
matching — which is exactly the distinction the classifier needs to learn.

**Class imbalance:** handled via `scale_pos_weight` (ratio of negative to
positive training examples), since true matches are a small minority of all
blocked candidate pairs.

**Validation split:** performed by `source1_entity_id`, never by row — an
entity's full candidate set stays entirely on one side of the split, since
splitting by row would leak information (other candidates for the same entity
appearing in both train and validation).

**Threshold selection:** rather than the default 0.5 cutoff, the decision
threshold is swept on the validation split to directly maximize the
challenge's own **macro-averaged F_0.5**, computed per S1 entity with the
exact formula and singleton-scoring rule from the problem statement (a
singleton scores 1.0 for a correctly-empty prediction, 0.0 for any
false-positive). Because F_0.5 weights precision 2x over recall, the optimal
threshold is expected to sit above 0.5 — this tuning step is treated as a
first-class part of the methodology, not a post-hoc adjustment, since it can
move the final score more than most feature or model changes.

---

## 7. Inference and output generation

At test time, the same normalization → blocking → feature pipeline runs on
the test sources (no ground truth available). The saved model and threshold
are applied to produce:
- `matching_results.tsv` — pairs predicted as a match, one row per S1 entity,
  empty for predicted singletons.
- `candidate_pairs.tsv` — derived directly from the scored feature rows
  (not copied from the blocking stage's raw output), guaranteeing this file
  always reflects the exact candidate set the model saw, and that every
  matched ID is a subset of it.

Both are validated locally against every submission rule
(`utils/validate_submission.py`) before any leaderboard upload.

---

## 8. Compliance with challenge constraints

- **No external data lookups:** the entire pipeline — normalization,
  blocking, features, model — operates only on the provided training/test
  data plus static, offline, rule-based logic (regex, string-similarity
  libraries with no network calls). No business-identity APIs, government
  database lookups, or geocoding services are used anywhere.
- **Model size/license:** LightGBM is Apache-2.0/MIT-family licensed and
  several orders of magnitude below the 8B-parameter ceiling.
- **Open-set country handling:** country is used only as a soft equality
  feature; nothing in normalization, blocking, or features hard-codes,
  filters, or one-hot-encodes a fixed country set.

---

## 9. Validation strategy and iteration loop

Two numbers are tracked on every run, not just the final F_0.5:

1. **Blocking recall** (validation split) — the ceiling on everything
   downstream. If low, iterate on blocking keys (`TOP_K_TOKENS`,
   `MAX_TOKEN_DF`) before touching features or the model.
2. **Threshold-swept validation macro F_0.5** — if blocking recall is already
   high but this lags, iterate on feature engineering or model
   hyperparameters instead.

Error analysis on the validation set (false positives = over-merges, false
negatives = missed matches) is used to check whether errors cluster around a
specific pattern identified in Section 2 (non-Latin names, missing addresses,
particular countries), which then feeds back into normalization or blocking
rather than blind model tuning.

---

## 10. Known limitations / future work

- Token-blocking is a simpler substitute for the originally-considered
  MinHash/LSH fuzzy-blocking approach; if measured recall proves insufficient
  for heavily-typo'd records, LSH remains a candidate fourth blocking
  strategy (the `datasketch` dependency is already in `requirements.txt`).
- Cross-script name matching currently has no dedicated signal beyond the
  `script_mismatch` flag and the address-based fallback; a rule-based (not
  API-based) transliteration pass is a possible future addition if error
  analysis shows this is a meaningful source of missed matches.
- Blocking constants (`TOP_K_TOKENS`, `MAX_TOKEN_DF`) and the threshold-sweep
  grid are currently defaults, not yet tuned against measured recall/F_0.5
  numbers from a full run.
