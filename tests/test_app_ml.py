from pathlib import Path
import numpy as np
import pandas as pd

from app.model import (
    process_data,
    compute_model_metrics,
    inference,
    load_artifacts,
    LABEL,
)

def test_artifacts_exist():
    assert Path("model/model.pkl").exists()
    assert Path("model/encoder.pkl").exists()

def test_inference_runs_one_row():
    # Load trained artifacts
    model, enc = load_artifacts()

    # Load one real row from the CSV so columns/categories match training
    df = pd.read_csv("data/census.csv")
    assert LABEL in df.columns
    row = df.iloc[[0]].copy()
    if LABEL in row.columns:
        row = row.drop(columns=[LABEL])

    # Use object-dtype columns as categoricals (same rule as training)
    categorical_features = [c for c in df.columns if df[c].dtype == "object"]
    if LABEL in categorical_features:
        categorical_features.remove(LABEL)

    # Transform and predict
    X_proc, _, _ = process_data(
        row,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=enc,
    )
    preds = inference(model, X_proc)
    assert preds.shape == (1,)

def test_metrics_return_floats():
    y = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    p, r, f1 = compute_model_metrics(y, preds)
    for v in (p, r, f1):
        assert isinstance(v, float)
