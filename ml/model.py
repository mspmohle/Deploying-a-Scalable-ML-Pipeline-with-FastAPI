import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train a simple baseline classifier and return it.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 (beta=1).
    zero_division=1 avoids warnings/crashes on rare edge cases.
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Run model inference and return predictions.
    """
    return model.predict(X)


def save_model(obj, path: str) -> None:
    """
    Serialize a model/encoder/label-binarizer to disk.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path: str):
    """
    Load a serialized object from disk.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value,
    categorical_features: List[str],
    label: str,
    encoder,
    lb,
    model: RandomForestClassifier,
) -> Tuple[float, float, float]:
    """
    Compute precision/recall/F1 on the subset of `data` where `column_name == slice_value`.
    Uses provided encoder/lb with training=False (no refit).
    """
    sl = data[data[column_name] == slice_value]
    if sl.empty:
        return 0.0, 0.0, 0.0

    X_sl, y_sl, _, _ = process_data(
        sl,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_sl)
    return compute_model_metrics(y_sl, preds)
