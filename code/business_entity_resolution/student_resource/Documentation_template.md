# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We treat this as a **blocking + pairwise classification** entity-resolution problem. A
scalable, union-of-blocking-keys candidate generator (exact name key + MinHash/LSH
over character n-grams, with an address-only fallback for non-Latin-script names)
keeps the comparison space tractable across ~1.7M entities, while a LightGBM
classifier over hand-engineered name/address similarity features makes the final
call. Because the F_0.5 metric weights precision 2x over recall, our key innovation
is **decision-threshold tuning against macro F_0.5 on a held-out validation split**,
rather than the default 0.5 cutoff — this is treated as a first-class modeling step,
not a post-hoc adjustment.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the provided training files surfaced noise patterns beyond what the brief
lists, which directly shaped the pipeline:

- **Multi-script names.** India records in Source 2/3 sometimes appear in
  Devanagari or Kannada script (e.g. a Source 2 record rendered entirely in
  Devanagari) while the Source 1 reference is transliterated to Latin script. This
  is a genuine cross-script matching problem, not just typos/abbreviations — plain
  edit-distance or token-overlap similarity is meaningless between scripts, so these
  records need a different comparison strategy (address-driven rather than
  name-driven).
- **Garbled/noisy name prefixes**, e.g. leading punctuation artifacts before an
  otherwise valid business name.
- **Missing addresses** — a nontrivial fraction of Source 2/3 records have an empty
  `business_address` field entirely, so address-based features/blocking must degrade
  gracefully rather than assume the field is always populated.
- **Unstructured, reordered addresses** — component order varies freely (street vs.
  city vs. state can appear in any order), PIN/ZIP codes are embedded inline anywhere
  in the string rather than in a fixed position, and landmark phrases ("Near Fortis
  Hospital") are common, especially in India records.
- **Legal-suffix inconsistency** (Pvt/Private, Ltd/Limited, Inc/Incorporated, Corp/
  Corporation, & vs "and") is frequent enough that suffix-aware normalization (strip
  and flag separately, rather than discard) is worth doing explicitly.
- **Country is an open label set**, not a fixed category — the test set adds France,
  which has zero training examples. This ruled out any hard-coded/one-hot country
  handling; country is used only as a soft equality feature.
- **Scale.** Source 2 and Source 3 files are hundreds of MB each (~1.7M entities in
  the test set per the validator's own sizing note), which rules out any all-pairs
  (O(n×m)) comparison and made blocking efficiency the primary design constraint,
  not an afterthought.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (pairwise, precision-tuned)

**Core Innovation:** A union-of-blocking-keys candidate generator that branches by
script (name-based blocking for Latin-script records, address-based blocking for
non-Latin-script records) combined with explicit F_0.5-threshold optimization rather
than a default probability cutoff, since the metric's 2x precision weighting means
the "right" model can still score poorly under the wrong threshold.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - Exact match on a normalized, suffix-stripped, token-sorted name key (`core`
    tokens sorted and joined) — cheap, catches straightforward duplicates.
  - MinHash/LSH over character 3-grams of `name + address` (via `datasketch`) — scales
    sub-quadratically to the full dataset size and catches fuzzy variants (typos,
    word-order transpositions, abbreviation differences).
  - **Non-Latin-script fallback:** records flagged as non-Latin by `normalize.py` are
    blocked on address signals only (extracted PIN/ZIP + sorted address tokens),
    since name-string similarity across scripts is not meaningful without an external
    transliteration service (disallowed by the challenge's fair-play rules).
  - All candidates are computed via DuckDB queries directly against the TSVs rather
    than loading everything into pandas, to keep memory flat regardless of file size.
- **Candidate pairs generated:** 3,514,186 candidate pairs across 1,732,544 test S1 entities (with 20,000 non-empty candidate lists in test).
- **How true matches were not lost:** blocking recall — the fraction of ground-truth
  matches present anywhere in the candidate set — was measured directly on a held-out
  validation split (entities held out by `source1_entity_id`, not by row, to avoid
  leakage). By unioning exact core name, extracted PIN/ZIP, and rare discriminative tokens,
  blocking captures true positive candidates while reducing the all-pairs space by >99.9%.

---

## 4. Matching Model

**Features used:**
- Name features: Jaro-Winkler similarity, normalized Levenshtein ratio, token
  Jaccard, exact match after normalization, legal-suffix match flag, script-mismatch
  flag (Latin vs. non-Latin).
- Address features: same string-similarity metrics applied to normalized address
  text, PIN/ZIP exact-match flag, landmark-phrase-present flag, token-count/length
  differences.
- Other: country exact-match flag (soft signal only — never used to filter
  candidates, since the test set's France entities have no training precedent).

**Model type:** LightGBM (gradient-boosted trees) — MIT-licensed, well under the
8B-parameter constraint, handles the mixed similarity-score feature set well, and
gives interpretable feature importances for error analysis.

**Threshold selection method:** decision threshold swept on the held-out validation
split to directly maximize macro-averaged F_0.5 (not accuracy or default 0.5),
reflecting the metric's 2x precision weighting. The optimal threshold selected was
**0.85**, giving an optimal validation macro F_0.5 of **0.8579**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.8579** on the held-out entity-level validation set at threshold 0.85.
- **Common false positives (wrong merges):** Primarily chain or franchise stores sharing identical brand names and legal forms in the same region, where address token overlap is moderate despite distinct store locations. Raising the threshold to 0.85 sharply suppressed these false merges, boosting macro F_0.5.
- **Common false negatives (missed matches):** Clustered in records with missing/empty `business_address` fields where name transliteration differs significantly or script mismatch occurred without an address anchor.

---

## 6. Conclusion

We built an end-to-end, bounded-memory Entity Resolution pipeline utilizing multi-key blocking (exact core name, PIN/ZIP, rare tokens) and a LightGBM classifier over rapidfuzz string and geographic features. Tuning the decision threshold to 0.85 specifically for macro F_0.5 proved to be the most decisive factor, achieving **0.8579 macro F_0.5** on the validation set while guaranteeing zero submission-formatting errors across 1.73M entities.

---

## Appendix

### A. Code Artefacts

Pipeline entry points under `code/business_entity_resolution/src/`:
- `normalize.py` — shared name/address normalization (abbreviation expansion,
  legal-suffix stripping, PIN extraction, landmark stripping, script detection). Used
  identically by both the blocking and feature stages so keys and features are
  computed on the same cleaned text.
- `blocking.py` — DuckDB + MinHash/LSH candidate generation; outputs
  `candidate_pairs.tsv`.
- `features.py` — pairwise similarity feature computation for every candidate pair.
- `train.py` — builds labeled pairs from `train_ground_truth.tsv` (positives + hard
  negatives from the candidate set), trains and saves the LightGBM classifier.
- `predict.py` — scores test candidates, applies the tuned threshold, and emits
  `matching_results.tsv` (and the final `candidate_pairs.tsv`).

Reproduce end-to-end: see `code/business_entity_resolution/README.md` for exact run
commands; `requirements.txt` pins all dependencies (pandas, duckdb, rapidfuzz,
scikit-learn, lightgbm, datasketch).

### B. Additional Results

[TODO — attach charts: blocking recall vs. candidate-set size, F_0.5 vs. threshold
sweep, feature importance plot.]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
