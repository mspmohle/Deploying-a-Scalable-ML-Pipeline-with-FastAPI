from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

# ======== CONFIG ========
# Set this to your label column name if different
LABEL = "salary"

# Optionally list categorical columns explicitly; if None we auto-detect object dtype
EXPLICIT_CATEGORICAL_FEATURES: Optional[List[str]] = None
# ========================

@dataclass
class Artifacts:
    model_dir: Path = Path("model")
    model_path: Path = model_dir / "model.pkl"
    encoder_path: Path = model_dir / "encoder.pkl"

def _ensure_binary_labels(y: Iterable) -> np.ndarray:
    """Coerce common string labels to 0/1; leave ints as-is."""
    y = np.asarray(y)
    if y.dtype.kind in {"i", "u", "b"}:
        return y.astype(int)
    mapping = {
        "<=50K": 0, "<=50K.": 0, "0": 0, "no": 0, "No": 0, "NO": 0,
        ">50K": 1,  ">50K.": 1, "1": 1, "yes": 1, "Yes": 1, "YES": 1,
    }
    y_bin = np.array([mapping.get(str(v).strip(), v) for v in y], dtype=object)
    try:
        return y_bin.astype(int)
    except Exception as exc:
        raise ValueError("Could not coerce labels to 0/1. Update LABEL/mapping in app/model.py.") from exc

def infer_categorical_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if df[c].dtype == "object"]

def split_features(df: pd.DataFrame, label: Optional[str], categorical_features: List[str]) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    X = df.copy()
    y = None
    if label is not None and label in X.columns:
        y = _ensure_binary_labels(X[label].values)
        X = X.drop(columns=[label])
    X_cat = X[categorical_features].astype(str) if categorical_features else pd.DataFrame(index=X.index)
    X_num = X.drop(columns=categorical_features) if categorical_features else X
    return (pd.concat([X_num, X_cat], axis=1), y)

def process_data(
    X: pd.DataFrame,
    categorical_features: List[str],
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], OneHotEncoder]:
    X_all, y = split_features(X, label=label, categorical_features=categorical_features)
    X_num = X_all.drop(columns=categorical_features, errors="ignore")
    X_cat = X_all[categorical_features] if categorical_features else pd.DataFrame(index=X_all.index)

    if training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_enc = encoder.fit_transform(X_cat) if not X_cat.empty else np.empty((len(X_all), 0))
    else:
        if encoder is None:
            raise ValueError("Encoder must be provided during inference (training=False).")
        X_cat_enc = encoder.transform(X_cat) if not X_cat.empty else np.empty((len(X_all), 0))

    X_num_vals = X_num.to_numpy(dtype=float, copy=False) if not X_num.empty else np.empty((len(X_all), 0))
    X_proc = np.concatenate([X_num_vals, X_cat_enc], axis=1)
    return X_proc, y, encoder  # type: ignore[return-value]

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict(X)

def save_artifacts(model: RandomForestClassifier, encoder: OneHotEncoder, art: Artifacts = Artifacts()) -> None:
    art.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, art.model_path)
    joblib.dump(encoder, art.encoder_path)

def load_artifacts(art: Artifacts = Artifacts()):
    model = joblib.load(art.model_path)
    encoder = joblib.load(art.encoder_path)
    return model, encoder
