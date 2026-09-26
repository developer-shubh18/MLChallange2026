# Business Entity Resolution Pipeline

An end-to-end, high-performance machine learning pipeline designed to resolve and match noisy business entity records across multiple disparate data sources under the **Macro $F_{0.5}$** evaluation metric (prioritizing precision $2\times$ over recall).

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Directory Structure](#directory-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [End-to-End Execution Guide](#end-to-end-execution-guide)
  - [1. Candidate Generation / Blocking](#stage-1-candidate-generation--blocking-blockingpy)
  - [2. Feature Engineering](#stage-2-pairwise-feature-engineering-featurespy)
  - [3. Model Training & Threshold Tuning](#stage-3-model-training--threshold-tuning-trainpy)
  - [4. Test Inference & Prediction](#stage-4-test-inference--submission-generation-predictpy)
  - [5. Submission Validation](#stage-5-submission-validation)
- [Quick Reference: Complete Workflow](#quick-reference-complete-workflow)
- [Key Hyperparameters & Tuning Knobs](#key-hyperparameters--tuning-knobs)
- [Memory & Performance Optimizations](#memory--performance-optimizations)
- [Troubleshooting](#troubleshooting)

---

## Pipeline Architecture

The pipeline processes high-volume, noisy tabular datasets (hundreds of MBs per source, ~1.7M records) in five modular stages:

```
[Raw Sources: S1, S2, S3]
         │
         ▼
 1. normalize.py  ──► Token standardization, legal-suffix stripping, PIN extraction (cached to Parquet)
         │
         ▼
 2. blocking.py   ──► Multi-strategy candidate pairing (Name core, PIN match, Rare-token index via DuckDB)
         │
         ▼
 3. features.py   ──► 13 similarity features (Jaro-Winkler, Levenshtein, Jaccard, script/pin/country flags)
         │
         ▼
 4. train.py      ──► LightGBM classifier with class reweighting + exact Macro F_0.5 threshold sweep
         │
         ▼
 5. predict.py    ──► Batched test inference ──► matching_results.tsv & candidate_pairs.tsv
         │
         ▼
 validate_submission.py ──► Official schema and submission verification
```

---

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   ├── normalize.py           # Shared text/address normalization & Parquet streaming cache
│   ├── blocking.py            # High-recall DuckDB candidate generation (Stage 2)
│   ├── features.py            # Batched pairwise similarity feature extractor (Stage 3)
│   ├── train.py               # LightGBM training & Macro F_0.5 threshold optimizer (Stage 4)
│   └── predict.py             # Model scoring, threshold application & submission output (Stage 5)
├── cache/                     # Generated Parquet caches for normalized inputs & intermediate tables
├── output_train/              # Training candidate pairs and feature part files
├── output/                    # Final test predictions and candidate pairs for submission
├── model/                     # Saved LightGBM booster model and threshold metadata
├── student_resource/
│   ├── dataset/
│   │   ├── train/             # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   │   └── test/              # test_source1.tsv, test_source2.tsv, test_source3.tsv
│   └── utils/
│       └── validate_submission.py  # Official verification script
├── requirements.txt           # Pinned production dependencies
├── methodology.md             # In-depth architectural & design rationale
└── README.md                  # This run guide
```

---

## Prerequisites & Environment Setup

Python 3.9+ is recommended. Activate your virtual environment and install the required dependencies:

```bash
# Navigate to the project root
cd /Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution

# On macOS: LightGBM requires libomp (OpenMP runtime)
brew install libomp

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- **pandas** (>= 2.2): Chunked TSV streaming and tabular operations.
- **duckdb** (>= 1.0): High-speed, bounded-memory SQL engine for joins and candidate generation.
- **pyarrow** (>= 15): Fast Parquet serialization and streaming storage.
- **rapidfuzz** (>= 3.9): C++ accelerated Levenshtein and Jaro-Winkler string similarity.
- **scikit-learn** (>= 1.4): Entity-grouped cross-validation (`GroupShuffleSplit`).
- **lightgbm** (>= 4.3): Gradient boosting classifier optimized for tabular matching.

---

## End-to-End Execution Guide

Follow these sequential steps to train the model, tune the threshold, and generate the final submission files.

### Stage 1: Candidate Generation / Blocking (`blocking.py`)

Generates high-recall candidate pairs between Source-1 and Source-2/3.
- On the first run, it automatically streams and normalizes raw TSVs via `normalize.py` into `cache/*.parquet`.
- Evaluates 3 unioned blocking strategies:
  1. Exact normalized name match (`name_core`)
  2. Exact PIN/ZIP code match (`addr_pin`)
  3. Discriminative rare-token match (`TOP_K_TOKENS=2`)

```bash
python src/blocking.py \
  --split train \
  --dataset-dir student_resource/dataset \
  --cache-dir cache \
  --out output_train/candidate_pairs.tsv
```

- **Input**: `student_resource/dataset/train/train_source{1,2,3}.tsv`
- **Output**: `output_train/candidate_pairs.tsv` and cached parquet tables in `cache/`.

---

### Stage 2: Pairwise Feature Engineering (`features.py`)

Computes 13 similarity features across candidate pairs in batches (`--batch-size 20000`) using DuckDB disk-backed views and RapidFuzz C++ routines:
- String metrics: Jaro-Winkler, Levenshtein, token-set Jaccard for names and addresses.
- Structured checks: PIN equality, country match, legal suffix match, script mismatch, length disparities.
- When `--ground-truth` is passed, attaches binary match labels (`label = 1/0`).

```bash
python src/features.py \
  --split train \
  --dataset-dir student_resource/dataset \
  --cache-dir cache \
  --candidate-pairs output_train/candidate_pairs.tsv \
  --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
  --out output_train/features \
  --batch-size 20000
```

- **Input**: `output_train/candidate_pairs.tsv`, normalized caches in `cache/`, `train_ground_truth.tsv`
- **Output**: Multi-part Parquet files in `output_train/features/part_*.parquet`

---

### Stage 3: Model Training & Threshold Tuning (`train.py`)

Trains the LightGBM classifier and tunes the decision threshold:
- **Entity-Level Splitting**: Uses `GroupShuffleSplit` on `source1_entity_id` so an entity's candidate set never crosses the train/validation split (preventing data leakage).
- **Imbalance Handling**: Automatically calculates `scale_pos_weight` based on negative-to-positive ratio.
- **Metric Optimization**: Sweeps decision thresholds over $[0.05, 0.95]$ against the challenge's exact **Macro $F_{0.5}$** formula (rewarding precision $2\times$ over recall and accounting for singletons).

```bash
python src/train.py \
  --features-dir output_train/features \
  --model-out model/lgbm_model.txt \
  --threshold-out model/threshold.json \
  --test-size 0.15 \
  --seed 42
```

- **Outputs**:
  - `model/lgbm_model.txt`: Trained LightGBM booster.
  - `model/threshold.json`: Optimal decision threshold and validation metrics.

---

### Stage 4: Test Inference & Submission Generation (`predict.py`)

Once the model is trained, execute test-set candidate blocking, feature extraction, and scoring.

#### 4a. Generate Test Candidates
```bash
python src/blocking.py \
  --split test \
  --dataset-dir student_resource/dataset \
  --cache-dir cache \
  --out output/test_candidate_pairs_blocking.tsv
```

#### 4b. Generate Test Features (Unlabeled)
```bash
python src/features.py \
  --split test \
  --dataset-dir student_resource/dataset \
  --cache-dir cache \
  --candidate-pairs output/test_candidate_pairs_blocking.tsv \
  --out output/features_test \
  --batch-size 20000
```

#### 4c. Score Candidates & Output TSVs
Applies the trained LightGBM model and calibrated threshold to produce the two required submission TSV files:
- `matching_results.tsv`: Only predicted matches exceeding the threshold (one row per S1 entity).
- `candidate_pairs.tsv`: All candidates scored by the model.

```bash
python src/predict.py \
  --features-dir output/features_test \
  --model model/lgbm_model.txt \
  --threshold-file model/threshold.json \
  --dataset-dir student_resource/dataset \
  --split test \
  --out-dir output
```

---

### Stage 5: Submission Validation

Run the provided validator before submitting to ensure compliant headers, TSV formatting, entity coverage, and subset rules:

```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

> **Tip**: Pass `--check-ids` if you want a complete diagnostic verifying that every predicted ID exists in the raw test sources.

---

## Quick Reference: Complete Workflow

Run all stages consecutively with this automated shell script:

```bash
#!/usr/bin/env bash
set -e

DATA_DIR="student_resource/dataset"
CACHE_DIR="cache"
MODEL_DIR="model"
OUTPUT_DIR="output"
TRAIN_OUT="output_train"

mkdir -p "$CACHE_DIR" "$MODEL_DIR" "$OUTPUT_DIR" "$TRAIN_OUT"

echo "=== Stage 1: Train Blocking ==="
python src/blocking.py --split train --dataset-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" --out "$TRAIN_OUT/candidate_pairs.tsv"

echo "=== Stage 2: Train Feature Extraction ==="
python src/features.py --split train --dataset-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --candidate-pairs "$TRAIN_OUT/candidate_pairs.tsv" \
  --ground-truth "$DATA_DIR/train/train_ground_truth.tsv" \
  --out "$TRAIN_OUT/features"

echo "=== Stage 3: Training & Metric Sweep ==="
python src/train.py --features-dir "$TRAIN_OUT/features" --model-out "$MODEL_DIR/lgbm_model.txt" --threshold-out "$MODEL_DIR/threshold.json"

echo "=== Stage 4: Test Blocking ==="
python src/blocking.py --split test --dataset-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" --out "$OUTPUT_DIR/test_candidate_pairs_blocking.tsv"

echo "=== Stage 5: Test Feature Extraction ==="
python src/features.py --split test --dataset-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
  --candidate-pairs "$OUTPUT_DIR/test_candidate_pairs_blocking.tsv" \
  --out "$OUTPUT_DIR/features_test"

echo "=== Stage 6: Prediction & Emission ==="
python src/predict.py --features-dir "$OUTPUT_DIR/features_test" \
  --model "$MODEL_DIR/lgbm_model.txt" \
  --threshold-file "$MODEL_DIR/threshold.json" \
  --dataset-dir "$DATA_DIR" \
  --split test \
  --out-dir "$OUTPUT_DIR"

echo "=== Stage 7: Submission Validation ==="
python student_resource/utils/validate_submission.py \
  --matching "$OUTPUT_DIR/matching_results.tsv" \
  --candidate "$OUTPUT_DIR/candidate_pairs.tsv" \
  --test-dir "$DATA_DIR/test"

echo "Pipeline execution finished successfully!"
```

---

## Key Hyperparameters & Tuning Knobs

| Component | Parameter | Location | Default | Description |
|---|---|---|---|---|
| **Blocking** | `TOP_K_TOKENS` | `src/blocking.py:39` | `2` | Number of rarest discriminative tokens chosen as blocking keys per record. |
| **Blocking** | `MAX_TOKEN_DF` | `src/blocking.py:40` | `2000` | Token document frequency cutoff above which tokens are treated as noise. |
| **Blocking** | `MAX_BLOCK_FREQ` | `src/blocking.py:41` | `20` | Max records per blocking key to prevent Cartesian explosion. |
| **Features** | `batch_size` | `src/features.py:188` | `20000` | S1 entity batch size during DuckDB joins to bound RAM. |
| **Model** | `n_estimators` | `src/train.py:92` | `500` | Max boosting iterations (with early stopping = 30). |
| **Model** | `num_leaves` | `src/train.py:94` | `31` | Tree complexity parameter for LightGBM. |
| **Model** | `scale_pos_weight` | `src/train.py:95` | `neg/pos` | Class reweighting ratio to account for candidate imbalance. |
| **Threshold**| `thresholds` | `src/train.py:53` | `0.05..0.95` | Grid evaluated on validation set to maximize macro $F_{0.5}$. |

---

## Memory & Performance Optimizations

1. **DuckDB Disk Spilling**:
   - Both `blocking.py` and `features.py` configure DuckDB with `SET memory_limit='4GB'` and specify a persistent `temp_directory` inside the cache folder. Large joins spill gracefully to disk rather than throwing OOM errors.
2. **Chunked Parquet Caching**:
   - `normalize.py` processes raw TSVs in 200,000-row chunks via PyArrow `ParquetWriter`. Intermediate `.tmp` files guarantee atomicity and guard against corrupt partial caches.
3. **Partitioned Feature Storage**:
   - Features are emitted in indexed Parquet part files (`part_*.parquet`), preventing monolithic in-memory DataFrame bottlenecks.

---

## Troubleshooting

- **`No feature part-files found under ...`**:
  Make sure `features.py` finished writing all batches to the specified directory.
- **Cache Invalidation**:
  If you modify normalization rules in `normalize.py`, clear the cached Parquet files in `cache/` (`rm -rf cache/*.parquet`) or run with `force=True`.
- **Validation Warnings on Missing Candidate IDs**:
  `predict.py` guarantees that all matched IDs in `matching_results.tsv` are a strict subset of `candidate_pairs.tsv`. If the validator emits a warning, confirm you passed the generated `output/candidate_pairs.tsv`.
- **High Memory on Validation**:
  The default `validate_submission.py` command is lightweight. If passing `--check-ids` encounters memory pressure, omit `--candidate` during ID checks as suggested by the validator docstring.
