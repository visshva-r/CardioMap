from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ml.config import MODELS_DIR, TARGETS
from ml.data import feature_columns, load_raw_dataframe, transform_features


def _load_preprocessor():
    return joblib.load(MODELS_DIR / "preprocess.joblib")


def _load_model(target_key: str) -> Pipeline:
    return joblib.load(MODELS_DIR / f"{target_key.lower()}.joblib")


def _feature_names() -> list[str]:
    schema = json.loads((MODELS_DIR / "schema.json").read_text(encoding="utf-8"))
    return schema["transformed_feature_names"]


def build_global_importance(max_background: int = 120) -> dict[str, list[dict[str, Any]]]:
    df = load_raw_dataframe()
    preprocessor = _load_preprocessor()
    feat_cols = feature_columns(df)
    X = transform_features(preprocessor, feat_cols, df)
    names = _feature_names()
    background = shap.sample(X, min(max_background, len(X)), random_state=42)

    global_out: dict[str, list[dict[str, Any]]] = {}
    for target_key in TARGETS:
        model = _load_model(target_key)
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
            pairs = sorted(zip(names, vals), key=lambda x: x[1], reverse=True)[:15]
            global_out[target_key] = [
                {"feature": f, "importance": float(v)} for f, v in pairs
            ]
            continue

        explainer = shap.LinearExplainer(clf, background)
        shap_vals = explainer.shap_values(background)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        pairs = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)[:15]
        global_out[target_key] = [
            {"feature": f, "importance": float(v)} for f, v in pairs
        ]

    (MODELS_DIR / "global_shap.json").write_text(
        json.dumps(global_out, indent=2), encoding="utf-8"
    )
    return global_out


def _shap_row(sv: Any, n_features: int) -> np.ndarray:
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    arr = np.asarray(sv)
    if arr.ndim == 2:
        row = arr[0]
    else:
        row = arr.reshape(-1)
    if row.size != n_features:
        row = row[:n_features]
    return row.astype(float)


def explain_patient(patient_features: dict[str, Any]) -> dict[str, Any]:
    df = load_raw_dataframe()
    preprocessor = _load_preprocessor()
    feat_cols = feature_columns(df)
    row = pd.DataFrame([patient_features])
    for col in feat_cols:
        if col not in row.columns:
            row[col] = df[col].mode().iloc[0]
    X = transform_features(preprocessor, feat_cols, row)
    names = _feature_names()

    predictions: dict[str, Any] = {}
    for target_key, meta in TARGETS.items():
        model = _load_model(target_key)
        prob = float(model.predict_proba(X)[0, 1])
        clf = model.named_steps["clf"]
        contributions: list[dict[str, Any]] = []

        if hasattr(clf, "feature_importances_"):
            bg = shap.sample(
                transform_features(preprocessor, feat_cols, df),
                min(80, len(df)),
                random_state=42,
            )
            explainer = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional")
            sv = explainer.shap_values(X)
            values = _shap_row(sv, len(names))
        else:
            bg = shap.sample(
                transform_features(preprocessor, feat_cols, df),
                min(80, len(df)),
                random_state=42,
            )
            explainer = shap.LinearExplainer(clf, bg)
            sv = explainer.shap_values(X)
            values = _shap_row(sv, len(names))

        pairs = sorted(
            zip(names, values.tolist()), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        for fname, val in pairs:
            contributions.append(
                {
                    "feature": fname,
                    "shap": float(val),
                    "direction": "increases_risk" if val > 0 else "decreases_risk",
                }
            )

        predictions[target_key] = {
            "label": meta["display"],
            "probability": prob,
            "risk_level": _risk_band(prob),
            "contributions": contributions,
        }

    return {"predictions": predictions}


def _risk_band(prob: float) -> str:
    if prob >= 0.66:
        return "high"
    if prob >= 0.33:
        return "moderate"
    return "low"


if __name__ == "__main__":
    build_global_importance()
    print("Wrote global SHAP importances")
