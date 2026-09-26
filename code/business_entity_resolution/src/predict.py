"""
predict.py — Stage 5: score test candidates with the trained model, apply the
tuned threshold, and emit the two required submission files.

Inputs:
  - unlabeled test feature Parquet parts (from features.py --split test, run
    WITHOUT --ground-truth)
  - the model + threshold saved by train.py

Outputs (written to --out-dir):
  - matching_results.tsv — thresholded predictions; the file scored on the
    leaderboard. One row per test S1 entity, empty string for singletons.
  - candidate_pairs.tsv  — the exact candidate set the model actually scored.
    This is derived from the feature rows themselves (every pair that survived
    blocking AND got a feature row), not just copied from blocking.py's output —
    so it can never silently drift from what was really fed to the model, even
    if a join in features.py happened to drop a pair (e.g. a normalization
    cache miss). This is also why every ID in matching_results.tsv is
    guaranteed to appear here: matched pairs are a strict subset of scored pairs.
"""

import argparse
import json
import os

import lightgbm as lgb
import pandas as pd

from features import load_features, FEATURE_COLS


def _all_s1_ids(dataset_dir: str, split: str) -> pd.Series:
    """Every S1 entity that must appear in the output, straight from the raw
    source1 file — this is the required row set regardless of blocking."""
    path = os.path.join(dataset_dir, split, f"{split}_source1.tsv")
    df = pd.read_csv(path, sep="\t", usecols=["entity_id"], dtype=str)
    return df["entity_id"].rename("source1_entity_id")


def _write_id_list_tsv(s1_ids: pd.Series, id_col: str, pairs_df: pd.DataFrame, out_path: str):
    """Shared writer for both output files — same shape: one row per S1 entity,
    comma-separated candidate/matched ids, empty when none, no duplicates."""
    if pairs_df.empty:
        grouped = pd.Series(dtype=str, name=id_col)
    else:
        grouped = (
            pairs_df.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .rename(id_col)
        )
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out = out.merge(grouped, on="source1_entity_id", how="left")
    out[id_col] = out[id_col].fillna("")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)


def predict(features_dir: str, model_path: str, threshold_path: str,
            dataset_dir: str, split: str, out_dir: str):
    df = load_features(features_dir)
    if "label" in df.columns:
        print("Note: feature files include a `label` column — make sure these "
              "were generated for the TEST split (label is simply ignored here "
              "if this was accidentally a labeled train-features run).")

    with open(threshold_path) as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    feature_cols = meta.get("feature_cols", FEATURE_COLS)
    print(f"Using threshold={threshold:.2f} "
          f"(validation macro F_0.5 at train time: {meta.get('val_f_half', float('nan')):.4f})")

    booster = lgb.Booster(model_file=model_path)
    df["score"] = booster.predict(df[feature_cols])
    df["pred"] = (df["score"] >= threshold).astype(int)

    s1_ids = _all_s1_ids(dataset_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    # candidate_pairs.tsv: every pair actually scored, regardless of predicted label.
    _write_id_list_tsv(s1_ids, "candidate_entity_ids", df,
                        os.path.join(out_dir, "candidate_pairs.tsv"))

    # matching_results.tsv: only pairs predicted as a match.
    # Apply conflict resolution: each candidate record (S2/S3) is assigned
    # exclusively to the highest-scoring S1 entity (deduplicated S1 reference).
    matched = df[df["pred"] == 1].copy()
    if not matched.empty:
        matched = matched.sort_values("score", ascending=False).drop_duplicates(
            subset=["candidate_entity_id"], keep="first"
        )
    _write_id_list_tsv(s1_ids, "matched_entity_ids", matched,
                        os.path.join(out_dir, "matching_results.tsv"))

    n_entities = len(s1_ids)
    n_matched_entities = matched["source1_entity_id"].nunique() if not matched.empty else 0
    print(f"\n{n_entities:,} S1 entities total")
    print(f"  {n_matched_entities:,} predicted to have at least one match")
    print(f"  {n_entities - n_matched_entities:,} predicted as singletons (no match)")
    print(f"  {len(matched):,} total matched (entity, candidate) pairs")
    print(f"\nWrote {out_dir}/matching_results.tsv and {out_dir}/candidate_pairs.tsv")
    print("Next: run utils/validate_submission.py against these before submitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 5: score test candidates, apply threshold, emit submission files."
    )
    parser.add_argument("--features-dir", required=True,
                         help="Unlabeled feature Parquet parts from features.py (no --ground-truth).")
    parser.add_argument("--model", default="model/lgbm_model.txt")
    parser.add_argument("--threshold-file", default="model/threshold.json")
    parser.add_argument("--dataset-dir", default="student_resource/dataset")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                         help="Which split's source1 file defines the required S1 entity rows.")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    predict(args.features_dir, args.model, args.threshold_file,
            args.dataset_dir, args.split, args.out_dir)
