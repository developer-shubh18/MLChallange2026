# Business Entity Resolution

A machine learning pipeline to identify and match duplicate business entities across datasets using name/address normalization, blocking, and a pairwise classifier.

## Project Structure

```
business_entity_resolution/
├── src/
│   ├── normalize.py     # Name and address cleaning/standardization
│   ├── blocking.py      # Candidate pair generation to reduce comparison space
│   ├── features.py      # Pairwise feature engineering for each candidate pair
│   ├── train.py         # Train and save the match classifier
│   └── predict.py       # Run inference, output results
├── README.md
└── requirements.txt
```

## Pipeline Overview

```
Raw Data → normalize.py → blocking.py → features.py → train.py → predict.py → Outputs
```

1. **Normalize** — Lowercases, strips punctuation, expands abbreviations (St → Street, LLC → etc.), and standardizes addresses.
2. **Block** — Groups records into candidate pairs using keys (e.g. zip code + first token of name) to avoid O(n²) comparisons.
3. **Features** — Computes similarity scores per pair: name (TF-IDF cosine, Jaro-Winkler), address (token overlap), and structured fields.
4. **Train** — Trains a gradient boosted classifier (XGBoost/LightGBM) on labeled pairs, saves model to `model/`.
5. **Predict** — Loads model, scores all candidate pairs, outputs results.

## Outputs

| File | Description |
|---|---|
| `candidate_pairs.tsv` | All blocked candidate pairs with feature vectors |
| `matching_results.tsv` | Pairs predicted as matches with confidence scores |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train
python src/train.py --input data/entities.csv --labels data/labels.csv

# Predict
python src/predict.py --input data/entities.csv --model model/classifier.pkl
```

## Requirements

See `requirements.txt` for dependencies.
