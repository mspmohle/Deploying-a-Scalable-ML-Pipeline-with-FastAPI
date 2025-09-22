from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from app.model import (
    process_data,
    inference,
    load_artifacts,
    LABEL,
)

app = FastAPI(title="ML Inference Service")

class Payload(BaseModel):
    age: int
    workclass: str
    fnlwgt: Optional[int] = None
    fnlgt: Optional[int] = None
    education: str
    education_num: Optional[int] = None
    education_num_: Optional[int] = None
    marital_status: Optional[str] = None
    marital_status_dash: Optional[str] = None
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Optional[int] = None
    capital_loss: Optional[int] = None
    hours_per_week: Optional[int] = None
    native_country: Optional[str] = None
    native_country_dash: Optional[str] = None
    hours_per_week_dash: Optional[int] = None
    capital_gain_dash: Optional[int] = None
    capital_loss_dash: Optional[int] = None

def _normalize_input(d: dict) -> pd.DataFrame:
    m = d.copy()
    if "fnlgt" in m and "fnlwgt" not in m: m["fnlwgt"] = m.pop("fnlgt")
    if "education_num" in m and "education-num" not in m: m["education-num"] = m.pop("education_num")
    if "education_num_" in m and "education-num" not in m: m["education-num"] = m.pop("education_num_")
    if "marital_status" in m and "marital-status" not in m: m["marital-status"] = m.pop("marital_status")
    if "marital_status_dash" in m and "marital-status" not in m: m["marital-status"] = m.pop("marital_status_dash")
    if "native_country" in m and "native-country" not in m: m["native-country"] = m.pop("native_country")
    if "native_country_dash" in m and "native-country" not in m: m["native-country"] = m.pop("native_country_dash")
    if "hours_per_week" in m and "hours-per-week" not in m: m["hours-per-week"] = m.pop("hours_per_week")
    if "hours_per_week_dash" in m and "hours-per-week" not in m: m["hours-per-week"] = m.pop("hours_per_week_dash")
    if "capital_gain" in m and "capital-gain" not in m: m["capital-gain"] = m.pop("capital_gain")
    if "capital_gain_dash" in m and "capital-gain" not in m: m["capital-gain"] = m.pop("capital_gain_dash")
    if "capital_loss" in m and "capital-loss" not in m: m["capital-loss"] = m.pop("capital_loss")
    if "capital_loss_dash" in m and "capital-loss" not in m: m["capital-loss"] = m.pop("capital_loss_dash")
    m = {k: v for k, v in m.items() if v is not None}
    return pd.DataFrame([m])

def _expected_categorical_from_encoder(encoder) -> List[str]:
    # OneHotEncoder stores the original categorical feature names here
    try:
        return [str(c) for c in encoder.feature_names_in_]
    except Exception:
        # Fallback: empty list (shouldn’t happen if you trained successfully)
        return []

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Payload):
    # Load artifacts
    try:
        model, enc = load_artifacts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifacts missing: {e}")

    # Normalize input names (underscore ↔ hyphen)
    X = _normalize_input(payload.dict())

    # Use the SAME categorical column names & order as training
    expected_cats = _expected_categorical_from_encoder(enc)

    # Ensure every expected categorical column exists, as string, in the right order
    for c in expected_cats:
        if c not in X.columns:
            X[c] = ""  # empty category -> ignored by OneHotEncoder(handle_unknown="ignore")
        X[c] = X[c].astype(str)
    # Put categorical columns at the end or anywhere; process_data uses the list we pass in
    # Make sure no label column sneaks in
    if LABEL in X.columns:
        X = X.drop(columns=[LABEL])

    try:
        X_proc, _, _ = process_data(
            X,
            categorical_features=expected_cats,   # <-- exact list/order from training
            label=None,
            training=False,
            encoder=enc,
        )
        pred = inference(model, X_proc)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Input processing failed: {e}")

    return {"prediction": int(pred)}
