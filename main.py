#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from joblib import load as joblib_load

app = FastAPI(title="Income classifier API", version="1.0.7")

# Columns used during training
NUMERIC = ["age", "fnlgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICALS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class IncomeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

_MODEL = None
_ENCODER = None
_LB = None

def _p(name: str) -> Path:
    return Path("model") / name

def _load_file(path: Path, what: str):
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Missing artifact: {path}")
    try:
        obj = joblib_load(path)
        if obj is None:
            raise RuntimeError(f"{what} loaded as None")
        return obj
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {what} from {path.name}: {e}")

def _load_artifacts():
    global _MODEL, _ENCODER, _LB
    if _MODEL is None:
        _MODEL = _load_file(_p("model.joblib"), "model")
    if _ENCODER is None:
        _ENCODER = _load_file(_p("encoder.joblib"), "encoder")
    if _LB is None:
        _LB = _load_file(_p("lb.joblib"), "label binarizer")

@app.get("/")
def root():
    return {"message": "Income classifier API"}

@app.get("/health")
def health():
    ok = _p("model.joblib").exists() and _p("encoder.joblib").exists() and _p("lb.joblib").exists()
    return {"status": "ok" if ok else "missing_artifacts"}

@app.post("/predict")
def predict(req: IncomeRequest):
    _load_artifacts()

    # Build a single-row DataFrame with original names
    row = {
        "age": req.age,
        "workclass": req.workclass,
        "fnlgt": req.fnlgt,
        "education": req.education,
        "education-num": req.education_num,
        "marital-status": req.marital_status,
        "occupation": req.occupation,
        "relationship": req.relationship,
        "race": req.race,
        "sex": req.sex,
        "capital-gain": req.capital_gain,
        "capital-loss": req.capital_loss,
        "hours-per-week": req.hours_per_week,
        "native-country": req.native_country,
    }
    X_df = pd.DataFrame([row])

    # Numeric cast + exact order
    try:
        for c in NUMERIC:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0).astype(int)
        X_cont = X_df[NUMERIC]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"numeric cast/order failed: {e}")

    # One-hot encode categoricals with saved encoder (sparse -> dense)
    try:
        cats = [c for c in CATEGORICALS if c in X_df.columns]
        X_cat = X_df[cats] if cats else pd.DataFrame(index=X_df.index)
        X_cat_enc = _ENCODER.transform(X_cat) if len(cats) else np.empty((len(X_df), 0))
        if hasattr(X_cat_enc, "toarray"):
            X_cat_enc = X_cat_enc.toarray()
        X_proc = np.concatenate([X_cont.to_numpy(), X_cat_enc], axis=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"categorical transform failed: {e}")

    # Predict (same as your successful smoke test)
    try:
        preds = _MODEL.predict(X_proc)
        pred_int = int(preds[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    # Map to string label
    try:
        if hasattr(_LB, "classes_"):
            classes = list(_LB.classes_)
            pred_label = str(classes[pred_int]) if 0 <= pred_int < len(classes) else (">50K" if pred_int == 1 else "<=50K")
        else:
            pred_label = ">50K" if pred_int == 1 else "<=50K"
    except Exception:
        pred_label = ">50K" if pred_int == 1 else "<=50K"

    prob_gt_50k: Optional[float] = None
    if hasattr(_MODEL, "predict_proba"):
        try:
            prob_gt_50k = float(_MODEL.predict_proba(X_proc)[:, 1][0])
        except Exception:
            prob_gt_50k = None

    return {"prediction": pred_label, "prob_gt_50k": prob_gt_50k}
