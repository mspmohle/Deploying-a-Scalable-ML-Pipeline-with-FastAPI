#!/usr/bin/env python3
from __future__ import annotations
import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from data import process_data
from model import inference

app = FastAPI(title="D501 P2 Income Classifier", version="1.0.0")

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
ENC_PATH   = os.path.join(MODEL_DIR, "encoder.joblib")
LB_PATH    = os.path.join(MODEL_DIR, "lb.joblib")

class CensusRecord(BaseModel):
    age: int
    fnlgt: int
    education_num: int = Field(..., alias="education-num")
    capital_gain: int  = Field(..., alias="capital-gain")
    capital_loss: int  = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    workclass: str
    education: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(..., alias="native-country")
    model_config = ConfigDict(populate_by_name=True)

@lru_cache(maxsize=1)
def _artifacts():
    missing = [p for p in (MODEL_PATH, ENC_PATH, LB_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts: " + ", ".join(missing)
            + ". Run: python train_model.py --data-path data/census.csv --target salary"
        )
    return (joblib.load(MODEL_PATH), joblib.load(ENC_PATH), joblib.load(LB_PATH))

@app.get("/")
def root():
    return {"message": "Income classifier API", "version": app.version}

@app.get("/health")
def health():
    try:
        _ = _artifacts()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Artifacts not ready: {e}")

@app.post("/predict")
def predict(payload: CensusRecord):
    row = pd.DataFrame([payload.model_dump(by_alias=True)])
    model, enc, lb = _artifacts()
    X_proc, _, _, _ = process_data(row, training=False, encoder=enc, lb=lb)
    pred = int(inference(model, X_proc)[0])
    prob = float(model.predict_proba(X_proc)[0, 1]) if hasattr(model, "predict_proba") else None
    classes = list(getattr(lb, "classes_", ["<=50K", ">50K"]))
    label = classes[pred] if 0 <= pred < len(classes) else str(pred)
    return {"prediction": label, "prob_gt_50k": prob}
