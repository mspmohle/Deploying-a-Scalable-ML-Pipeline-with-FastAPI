# train_model.py
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from app.model import (
    LABEL,
    Artifacts,
    EXPLICIT_CATEGORICAL_FEATURES,
    compute_model_metrics,
    infer_categorical_features,
    inference,
    process_data,
    save_artifacts,
    train_model,
)

# ======== CONFIG ========
DATA_PATH = Path("data/census.csv")     # <- update if your CSV lives elsewhere
SLICE_REPORT = Path("slice_output.txt")
MIN_SLICE = 10                          # skip tiny groups in slice metrics
# ========================


def evaluate_slices(
    df: pd.DataFrame,
    categorical_features: List[str],
    model,
    encoder,
    label: str,
) -> str:
    """Return a text report with precision/recall/F1 for each value of each categorical feature."""
    lines: list[str] = []
    for feat in categorical_features:
        if feat not in df.columns:
            continue
        values = sorted({str(v) for v in df[feat].dropna().astype(str).unique()})
        if not values:
            continue

        lines.append(f"=== Feature: {feat} ===")
        for val in values:
            sub = df[df[feat].astype(str) == val]
            if len(sub) < MIN_SLICE:
                continue
            X_proc, y, _ = process_data(
                sub,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
            )
            preds = inference(model, X_proc)
            p, r, f1 = compute_model_metrics(y, preds)
            lines.append(f"{feat}={val} | n={len(y)} | precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
        lines.append("")  # spacer between features
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    # --- Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)

    # --- Select categorical features
    if EXPLICIT_CATEGORICAL_FEATURES is not None:
        categorical_features = [c for c in EXPLICIT_CATEGORICAL_FEATURES if c in df.columns]
    else:
        categorical_features = infer_categorical_features(df)
        if LABEL in categorical_features:
            categorical_features.remove(LABEL)

    # --- Basic checks
    if LABEL not in df.columns:
        raise ValueError(
            f"LABEL '{LABEL}' not in dataset columns. "
            f"Found columns (first 20): {list(df.columns)[:20]}"
        )

    # --- Train/test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # --- Process/train
    X_train, y_train, enc = process_data(
        train_df,
        categorical_features=categorical_features,
        label=LABEL,
        training=True,
        encoder=None,
    )
    model = train_model(X_train, y_train)

    # --- Save artifacts
    save_artifacts(model, enc, art=Artifacts())

    # --- Evaluate on test set
    X_test, y_test, _ = process_data(
        test_df,
        categorical_features=categorical_features,
        label=LABEL,
        training=False,
        encoder=enc,
    )
    preds = inference(model, X_test)
    p, r, f1 = compute_model_metrics(y_test, preds)
    print(f"[Test] n={len(y_test)}  precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}")

    # --- Slice metrics report
    report = evaluate_slices(test_df, categorical_features, model, enc, LABEL)
    SLICE_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote slice report -> {SLICE_REPORT.resolve()}")

    # --- Sanity: artifacts exist
    art = Artifacts()
    assert art.model_path.exists(), "model.pkl not found after training"
    assert art.encoder_path.exists(), "encoder.pkl not found after training"


if __name__ == "__main__":
    Path("model").mkdir(parents=True, exist_ok=True)
    main()
