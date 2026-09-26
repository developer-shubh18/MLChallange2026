"""
verify_features_v2.py — correctness check: v2 features.py vs the existing v1
output already on disk (output_train/features/part_00000000/20000/40000.parquet).

v2 sorts candidate pairs by source1_entity_id before batching, so its part-file
names don't correspond to v1's batches — this script sidesteps that by joining
old vs. new on (source1_entity_id, candidate_entity_id) rather than assuming
matching filenames, so it's safe to run regardless of how either version
batches internally.

Usage:
    python src/verify_features_v2.py \
        --old-dir output_train/features \
        --candidate-pairs output_train/candidate_pairs.tsv \
        --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
        --dataset-dir student_resource/dataset --cache-dir cache
"""

import argparse
import glob
import os
import tempfile

import numpy as np
import pandas as pd

from features import FEATURE_COLS, build_features


def main(old_dir, candidate_pairs_path, ground_truth_path, dataset_dir, cache_dir, split="train"):
    old_paths = sorted(glob.glob(os.path.join(old_dir, "*.parquet")))
    if not old_paths:
        raise FileNotFoundError(f"No existing v1 feature parts found under {old_dir}")
    old_df = pd.concat([pd.read_parquet(p) for p in old_paths], ignore_index=True)
    print(f"Loaded {len(old_df):,} v1 feature rows from {len(old_paths)} part file(s)")

    s1_ids = set(old_df["source1_entity_id"].unique())
    print(f"Verifying against {len(s1_ids):,} S1 entities covered by the existing v1 parts")

    # Build a candidate_pairs.tsv containing only those same S1 entities, so
    # v2 recomputes features for exactly the rows v1 already covered.
    full_pairs = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
    subset_pairs = full_pairs[full_pairs["source1_entity_id"].isin(s1_ids)]

    with tempfile.TemporaryDirectory() as tmp:
        subset_candidates_path = os.path.join(tmp, "candidate_pairs_subset.tsv")
        subset_pairs.to_csv(subset_candidates_path, sep="\t", index=False)

        new_out_dir = os.path.join(tmp, "features_v2_subset")
        build_features(dataset_dir, split, cache_dir, subset_candidates_path,
                        new_out_dir, ground_truth_path, batch_size=20_000)

        new_paths = sorted(glob.glob(os.path.join(new_out_dir, "*.parquet")))
        new_df = pd.concat([pd.read_parquet(p) for p in new_paths], ignore_index=True)

    print(f"v2 produced {len(new_df):,} feature rows for the same entity subset")

    merged = old_df.merge(
        new_df, on=["source1_entity_id", "candidate_entity_id"],
        suffixes=("_old", "_new"), how="outer", indicator=True
    )

    only_old = (merged["_merge"] == "left_only").sum()
    only_new = (merged["_merge"] == "right_only").sum()
    both = (merged["_merge"] == "both").sum()
    print(f"\nRow overlap: {both:,} in both, {only_old:,} only in v1, {only_new:,} only in v2")
    if only_old or only_new:
        print("MISMATCH: the two versions produced a different candidate-pair set for "
              "the same entities — investigate before trusting the feature diffs below.")

    print("\nMax absolute difference per feature (should be ~0.0, allowing float rounding):")
    both_rows = merged[merged["_merge"] == "both"]
    all_ok = True
    for col in FEATURE_COLS:
        old_col, new_col = f"{col}_old", f"{col}_new"
        if old_col not in both_rows.columns or new_col not in both_rows.columns:
            print(f"  {col}: SKIPPED (column missing on one side)")
            continue
        diff = (both_rows[old_col].astype(float) - both_rows[new_col].astype(float)).abs()
        max_diff = float(diff.max()) if len(diff) else float("nan")
        status = "OK" if max_diff < 1e-9 else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  {col}: max diff = {max_diff:.2e}  [{status}]")

    if "label" in both_rows.columns.union(pd.Index([])) and "label_old" in both_rows.columns:
        label_diff = (both_rows["label_old"].astype(int) != both_rows["label_new"].astype(int)).sum()
        print(f"  label: {label_diff:,} mismatched rows [{'OK' if label_diff == 0 else 'MISMATCH'}]")

    print("\n" + ("ALL FEATURES MATCH — safe to run v2 on the full dataset."
                   if all_ok and not only_old and not only_new
                   else "DO NOT run the full dataset yet — see mismatches above."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify features.py v2 against existing v1 output.")
    parser.add_argument("--old-dir", required=True, help="Directory of existing v1 Parquet parts.")
    parser.add_argument("--candidate-pairs", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--dataset-dir", default="student_resource/dataset")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    main(args.old_dir, args.candidate_pairs, args.ground_truth,
         args.dataset_dir, args.cache_dir, args.split)
