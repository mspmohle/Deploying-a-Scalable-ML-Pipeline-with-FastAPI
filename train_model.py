#!/usr/bin/env python3
"""
train_model.py — trains the model on census data, saves artifacts, and logs metrics + slice metrics.
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from data import process_data, _guess_target_name, DEFAULT_CATEGORICALS
from model import train_model, inference, compute_model_metrics
import joblib

def compute_slice_metrics(df: pd.DataFrame,
                          target: str,
                          categorical_columns: list[str],
                          encoder,
                          lb,
                          model) -> list[str]:
    """Compute precision/recall/fbeta for each category in each categorical column."""
    lines = []
    for col in categorical_columns:
        if col not in df.columns:
            continue
        for val in sorted(df[col].dropna().unique()):
            slice_df = df[df[col] == val]
            if len(slice_df) == 0:
                continue
            X_slice = slice_df.drop(columns=[target])
            y_slice = slice_df[target]
            Xp, yp, _, _ = process_data(X_slice, y=y_slice, training=False, encoder=encoder, lb=lb)
            preds = inference(model, Xp)
            p, r, f = compute_model_metrics(yp, preds)
            lines.append(f"{col} = {val:>15} | n={len(slice_df):4d} | precision={p:.3f} recall={r:.3f} fbeta={f:.3f}")
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/census.csv")
    ap.add_argument("--target", default=None, help="Target column name; auto-detect if omitted")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--screens-dir", default="screenshot")
    args = ap.parse_args()

    df = pd.read_csv(args.data_path)
    target = args.target or _guess_target_name(df)

    # split
    X = df.drop(columns=[target])
    y = df[target]
    Xtr_df, Xte_df, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    # process
    Xtr, ytr, encoder, lb = process_data(Xtr_df, y=ytr, training=True)
    Xte, yte,  _,   _     = process_data(Xte_df, y=yte, training=False, encoder=encoder, lb=lb)

    # train
    clf = train_model(Xtr, ytr)

    # eval
    preds = inference(clf, Xte)
    precision, recall, fbeta = compute_model_metrics(yte, preds)

    # save artifacts
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(encoder, os.path.join(args.model_dir, "encoder.joblib"))
    joblib.dump(lb, os.path.join(args.model_dir, "lb.joblib"))

    # save metrics + slice metrics
    Path(args.screens_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = os.path.join(args.screens_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"precision": float(precision), "recall": float(recall), "fbeta": float(fbeta)},
            f, indent=2
        )

    slice_lines = compute_slice_metrics(
        df, target, DEFAULT_CATEGORICALS, encoder, lb, clf
    )
    slice_path = os.path.join(args.screens_dir, "slice_output.txt")
    with open(slice_path, "w") as f:
        f.write("\n".join(slice_lines) + ("\n" if slice_lines else ""))

    print(f"Saved model to: {args.model_dir}/(model|encoder|lb).joblib")
    print(f"Metrics: {metrics_path}")
    print(f"Slice metrics: {slice_path}")
    print(f"Overall -> precision: {precision:.3f}, recall: {recall:.3f}, fbeta: {fbeta:.3f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
