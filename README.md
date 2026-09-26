# ML Challenge 2026 — Business Entity Resolution

This repository contains the machine learning solution for **ML Challenge 2026: Business Entity Resolution**.

## Overview

The goal is to accurately match duplicate business entities across three heterogeneous, noisy data sources:
- **Source 1**: Deduplicated reference entity set (every S1 entity must receive a prediction).
- **Source 2 & Source 3**: Noisy, multi-lingual, and multi-country operational sources.

Performance is evaluated under the competition's **Macro $F_{0.5}$** metric, which weights precision twice as heavily as recall and scores singletons (records with no valid match).

## Getting Started

All pipeline source code, environment specifications, and execution instructions are located in the [`code/business_entity_resolution`](file:///Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution) directory.

### Quick Links

- **Execution Guide & Run Commands**: [`code/business_entity_resolution/README.md`](file:///Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution/README.md)
- **Technical Methodology & Architecture**: [`code/business_entity_resolution/methodology.md`](file:///Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution/methodology.md)
- **Project Knowledge Base & Roadmap**: [`code/business_entity_resolution/KNOWLEDGE_BASE.md`](file:///Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution/KNOWLEDGE_BASE.md)
- **Dependencies**: [`code/business_entity_resolution/requirements.txt`](file:///Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution/requirements.txt)

---

### Quick Run

```bash
cd code/business_entity_resolution
pip install -r requirements.txt

# Run candidate generation (blocking)
python src/blocking.py --split train --dataset-dir student_resource/dataset --cache-dir cache --out output_train/candidate_pairs.tsv

# Extract features
python src/features.py --split train --dataset-dir student_resource/dataset --cache-dir cache \
  --candidate-pairs output_train/candidate_pairs.tsv \
  --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
  --out output_train/features

# Train classifier and sweep F_0.5 threshold
python src/train.py --features-dir output_train/features --model-out model/lgbm_model.txt --threshold-out model/threshold.json
```

For full details on testing, inference, and submission validation, refer to [the pipeline README](file:///Users/shubh/Desktop/MLChallange2026/code/business_entity_resolution/README.md).
