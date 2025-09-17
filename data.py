#!/usr/bin/env python3
"""
data.py — preprocessing utilities for the Census dataset.

Key entry point (rubric-standard):
    process_data(X, y=None, categorical_features=None, training=True, encoder=None, lb=None)
returns:
    X_processed (ndarray), y_processed (ndarray or None), fitted encoder, fitted label binarizer

This matches the Udacity D501 P2 expectations so train/test code and tests can import it directly.
"""

from __future__ import annotations
import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

# Canonical categorical feature list for the Census dataset
DEFAULT_CATEGORICALS: List[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def process_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    categorical_features: Optional[List[str]] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], OneHotEncoder, Optional[LabelBinarizer]]:
    """
    Process the data used in the machine learning pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Features only (no target column).
    y : pd.Series, optional
        Label column (binary), e.g. "salary" with values like <=50K/>50K.
    categorical_features : list[str], optional
        Which columns in X are categorical. Defaults to DEFAULT_CATEGORICALS ∩ X.columns.
    training : bool
        If True, fit encoder/lb; else use provided ones to transform.
    encoder : OneHotEncoder, optional
        The fitted OneHotEncoder, required when training=False.
    lb : LabelBinarizer, optional
        The fitted LabelBinarizer for y, required when training=False if y is provided.

    Returns
    -------
    X_processed : np.ndarray
    y_processed : np.ndarray | None
    encoder : OneHotEncoder
    lb : LabelBinarizer | None
    """
    if categorical_features is None:
        categorical_features = [c for c in DEFAULT_CATEGORICALS if c in X.columns]

    # Split categorical vs. continuous
    X_cat = X[categorical_features] if categorical_features else pd.DataFrame(index=X.index)
    X_cont = X.drop(columns=categorical_features, errors="ignore")

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat_enc = encoder.fit_transform(X_cat) if len(X_cat.columns) else np.empty((len(X), 0))
    else:
        if encoder is None:
            raise ValueError("encoder must be provided when training=False")
        X_cat_enc = encoder.transform(X_cat) if len(X_cat.columns) else np.empty((len(X), 0))

    X_proc = np.concatenate([X_cont.to_numpy(), X_cat_enc], axis=1)

    y_proc = None
    if y is not None:
        if training:
            lb = LabelBinarizer()
            y_proc = lb.fit_transform(y.values).ravel()
        else:
            if lb is None:
                raise ValueError("lb (LabelBinarizer) must be provided when training=False and y is not None")
            y_proc = lb.transform(y.values).ravel()

    return X_proc, y_proc, encoder, lb


def _guess_target_name(df: pd.DataFrame) -> str:
    # Common label names in this project
    for cand in ("salary", "income", "target"):
        if cand in df.columns:
            return cand
    raise KeyError("Target column not found. Expected one of: 'salary', 'income', 'target'.")


def _cli():
    parser = argparse.ArgumentParser(description="Smoke-test the data preprocessing")
    parser.add_argument("--data-path", default="data/census.csv", help="Path to census.csv")
    parser.add_argument("--target", default=None, help="Target column (default: auto-detect)")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    target = args.target or _guess_target_name(df)

    X = df.drop(columns=[target])
    y = df[target]

    Xtr, ytr, enc, lb = process_data(X, y=y, training=True)
    print("Train shapes:", Xtr.shape, ytr.shape)
    # simulate test-time transform
    Xte, yte, _, _ = process_data(X, y=y, training=False, encoder=enc, lb=lb)
    print("Test transform shapes:", Xte.shape, yte.shape)


if __name__ == "__main__":
    try:
        _cli()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
