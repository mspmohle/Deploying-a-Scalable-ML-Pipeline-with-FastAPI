#!/usr/bin/env python3
"""
model.py — model utilities for training, metrics, inference, and persistence.

Functions (rubric-standard):
    train_model(X_train, y_train) -> fitted model
    compute_model_metrics(y, preds) -> (precision, recall, fbeta)
    inference(model, X) -> predictions
    save_model(model, path) / load_model(path)
"""

from __future__ import annotations
from typing import Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Fits a simple baseline classifier. You can tune hyperparameters later.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns precision, recall, and fbeta.
    """
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    fbeta = fbeta_score(y, preds, beta=1.0, zero_division=0)
    return precision, recall, fbeta

def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return the predictions.
    """
    return model.predict(X)

def save_model(model, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)
