# train_model.py
from __future__ import annotations

from pathlib import Path

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

DATA_PATH = Path("data/census.csv")     # Update if your CSV is elsewhere
SLICE_REPORT = Path("slice_output.txt")
MIN_SLICE = 10                          # skip tiny groups in slice report


def evaluate_slices(
    df: pd.DataFrame,
    categorical_features: list[str],
    model,
    encoder,
    label: str,
) -> str:
    lines: list[str] = []
    for feat in categorical_features:
        # Robust against missing or all-NaN columns
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
        lines.append("")  # spacer
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    # Choose categorical features
    if EXPLICIT_CATEGORICAL_FEATURES is not None:
        categorical_features = [c for c in EXPLICIT_CATEGORICAL_FEATURES if c in df.columns]
    else:
        categorical_features = infer_categorical_features(df)
        if LABEL in categorical_features:
            categorical_features.remove(LABEL)

    # Train/test split (stratified if label is binary)
    if LABEL not in df.columns:
        raise ValueError(f"LABEL '{LABEL}' not found in dataset columns: {list(df.columns)[:15]} ...")

    # Simple manual split to avoid introducing extra dependencies
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    X_train, y_train, enc = process_data(
        train_df,
        categorical_features=categorical_features,
        label=LABEL,
        training=True,
        encoder=None,
    )
    model = train_model(X_train, y_train)

    # Save artifacts
    save_artifacts(model, enc, art=Artifacts())

    # Quick overall metrics on test set (printed to console)
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

    # Slice report
    report = evaluate_slices(test_df, categorical_features, model, enc, LABEL)
    SLICE_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote slice report -> {SLICE_REPORT.resolve()}")

    # Assert artifacts exist for sanity
    assert Artifacts().model_path.exists(), "model.pkl not found after training"
    assert Artifacts().encoder_path.exists(), "encoder.pkl not found after training"


if __name__ == "__main__":
    Path("model").mkdir(parents=True, exist_ok=True)
    main()
