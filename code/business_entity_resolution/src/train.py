"""
train.py — Stage 4: train and evaluate the match classifier.

Loads labeled feature rows (produced by features.py with --ground-truth),
splits by source1_entity_id — never by row, so an entity's candidate set never
straddles both sides of the split and leaks information — trains a LightGBM
classifier, then sweeps the decision threshold on the held-out validation split
to directly maximize the challenge's own macro-averaged F_0.5 metric (per the
exact formula in the problem statement), rather than optimizing accuracy or a
default 0.5 cutoff. Since F_0.5 weights precision 2x over recall, the "right"
model can still score poorly at the wrong threshold — this sweep is the main
lever, not an afterthought.

Saves the trained model + chosen threshold so predict.py can reuse them exactly.
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from features import load_features, FEATURE_COLS


def f_half_macro(df, id_col="source1_entity_id", label_col="label", pred_col="pred") -> float:
    """Exact challenge metric: per-source1-entity F_0.5, macro-averaged across
    all S1 entities in df, fully vectorized for speed. A singleton (no true matches)
    scores 1.0 for a correctly-empty prediction and 0.0 for any false-positive prediction."""
    tp_mask = ((df[pred_col] == 1) & (df[label_col] == 1)).astype(int)
    grouped = df.groupby(id_col).agg(
        n_true=(label_col, "sum"),
        n_pred=(pred_col, "sum"),
    )
    grouped["tp"] = df.assign(tp=tp_mask).groupby(id_col)["tp"].sum()

    n_true = grouped["n_true"].values
    n_pred = grouped["n_pred"].values
    tp = grouped["tp"].values

    is_singleton = (n_true == 0)
    singleton_scores = np.where(n_pred == 0, 1.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_pred > 0, tp / n_pred, 0.0)
        recall = np.where(n_true > 0, tp / n_true, 0.0)
        denom = 0.25 * precision + recall
        f_half = np.where(denom > 0, (1.25 * precision * recall) / denom, 0.0)

    scores = np.where(is_singleton, singleton_scores, f_half)
    return float(np.mean(scores)) if len(scores) else 0.0


def sweep_threshold(val_df, thresholds=np.arange(0.50, 0.96, 0.02)):
    """Return (best_threshold, best_f_half) over the given grid."""
    best_t, best_score = 0.7, -1.0
    for t in thresholds:
        val_df["pred"] = (val_df["score"] >= t).astype(int)
        score = f_half_macro(val_df)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def train(features_dir: str, model_out: str, threshold_out: str,
          test_size: float = 0.15, seed: int = 42):
    df = load_features(features_dir)
    if "label" not in df.columns:
        raise ValueError(
            "Feature files have no `label` column — regenerate them with "
            "features.py --ground-truth train_ground_truth.tsv"
        )

    print(f"Loaded {len(df):,} labeled pairs "
          f"({df['label'].sum():,} positive, {(df['label'] == 0).sum():,} negative)")

    groups = df["source1_entity_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    print(f"Split: {train_df['source1_entity_id'].nunique():,} train entities, "
          f"{val_df['source1_entity_id'].nunique():,} validation entities "
          "(split by entity, not by row)")

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["label"]

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    val_df["score"] = model.predict_proba(X_val)[:, 1]
    best_t, best_f_half = sweep_threshold(val_df)

    print(f"\nBest threshold: {best_t:.2f} -> validation macro F_0.5: {best_f_half:.4f}")
    print("\nFeature importances (gain):")
    for name, imp in sorted(zip(FEATURE_COLS, model.booster_.feature_importance(importance_type="gain")),
                             key=lambda x: -x[1]):
        print(f"  {name}: {imp:.1f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    model.booster_.save_model(model_out)
    with open(threshold_out, "w") as f:
        json.dump({"threshold": best_t, "val_f_half": best_f_half,
                    "feature_cols": FEATURE_COLS}, f, indent=2)
    print(f"\nSaved model to {model_out}")
    print(f"Saved threshold + feature order to {threshold_out}")

    return model, best_t, best_f_half


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: train the match classifier.")
    parser.add_argument("--features-dir", required=True,
                         help="Directory of labeled feature Parquet parts from features.py.")
    parser.add_argument("--model-out", default="model/lgbm_model.txt")
    parser.add_argument("--threshold-out", default="model/threshold.json")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.features_dir, args.model_out, args.threshold_out, args.test_size, args.seed)
