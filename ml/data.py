from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import DATA_PATH, LEAKAGE_COLUMNS, MODELS_DIR, SAMPLE_PATIENT_INDICES, TARGETS


def load_raw_dataframe() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    bad = LEAKAGE_COLUMNS.intersection(cols)
    if bad:
        raise ValueError(f"Leakage columns in feature list: {bad}")
    return cols


def assert_no_leakage(feature_names: list[str]) -> None:
    bad = LEAKAGE_COLUMNS.intersection(feature_names)
    if bad:
        raise ValueError(f"Leakage columns in feature matrix: {bad}")


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    feat_cols = feature_columns(df)
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feat_cols if c not in numeric_cols]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor, feat_cols


def target_series(df: pd.DataFrame, target_key: str) -> pd.Series:
    meta = TARGETS[target_key]
    col = meta["column"]
    positive = meta["positive_label"]
    return (df[col] == positive).astype(int)


def fit_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    preprocessor, feat_cols = build_preprocessor(df)
    X = df[feat_cols]
    preprocessor.fit(X)
    transformed_names = preprocessor.get_feature_names_out().tolist()
    assert_no_leakage(feat_cols)
    return preprocessor, feat_cols, transformed_names


def transform_features(
    preprocessor: ColumnTransformer, feat_cols: list[str], row: pd.DataFrame
) -> np.ndarray:
    return preprocessor.transform(row[feat_cols])


def build_schema(df: pd.DataFrame, feat_cols: list[str]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for col in feat_cols:
        series = df[col]
        entry: dict[str, Any] = {
            "name": col,
            "type": "number" if pd.api.types.is_numeric_dtype(series) else "category",
        }
        if entry["type"] == "number":
            entry["min"] = float(series.min())
            entry["max"] = float(series.max())
            entry["default"] = float(series.median())
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            entry["options"] = values
            entry["default"] = series.mode().iloc[0]
        schema.append(entry)
    return schema


def row_to_patient_dict(row: pd.Series, feat_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in feat_cols:
        val = row[col]
        if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float, np.integer, np.floating)):
            out[col] = float(val) if not pd.isna(val) else None
        else:
            out[col] = str(val)
    return out


def save_artifacts(
    preprocessor: ColumnTransformer,
    feat_cols: list[str],
    transformed_names: list[str],
    schema: list[dict[str, Any]],
    df: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocess.joblib")
    payload = {
        "feature_columns": feat_cols,
        "transformed_feature_names": transformed_names,
        "schema": schema,
        "targets": TARGETS,
        "leakage_columns": sorted(LEAKAGE_COLUMNS),
    }
    (MODELS_DIR / "schema.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    samples = []
    for idx in SAMPLE_PATIENT_INDICES:
        if idx < len(df):
            samples.append({"index": idx, "features": row_to_patient_dict(df.iloc[idx], feat_cols)})
    (MODELS_DIR / "sample_patients.json").write_text(
        json.dumps(samples, indent=2), encoding="utf-8"
    )


def load_schema() -> dict[str, Any]:
    return json.loads((MODELS_DIR / "schema.json").read_text(encoding="utf-8"))
