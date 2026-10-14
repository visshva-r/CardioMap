from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import CV_REPEATS, CV_SPLITS, MODELS_DIR, RANDOM_STATE, TARGETS
from ml.data import (
    build_schema,
    feature_columns,
    fit_preprocessor,
    load_raw_dataframe,
    save_artifacts,
    target_series,
    transform_features,
)


def candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "lightgbm": Pipeline(
            [
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        verbosity=-1,
                    ),
                )
            ]
        ),
    }


def evaluate_model(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    out: dict[str, float] = {}
    for key in scoring:
        test_key = f"test_{key}"
        vals = scores[test_key]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def train_all() -> None:
    df = load_raw_dataframe()
    preprocessor, feat_cols, transformed_names = fit_preprocessor(df)
    schema = build_schema(df, feat_cols)
    save_artifacts(preprocessor, feat_cols, transformed_names, schema, df)

    X_raw = df[feat_cols]
    X = preprocessor.transform(X_raw)
    metrics: dict[str, dict] = {"cv": {"splits": CV_SPLITS, "repeats": CV_REPEATS}}

    for target_key, meta in TARGETS.items():
        y = target_series(df, target_key).values
        best_name = None
        best_auc = -1.0
        best_metrics: dict[str, float] = {}
        best_model = None

        for name, model in candidate_models().items():
            fold_metrics = evaluate_model(model, X, y)
            auc = fold_metrics["roc_auc_mean"]
            print(f"{target_key} {name} roc_auc={auc:.3f} ± {fold_metrics['roc_auc_std']:.3f}")
            if auc > best_auc:
                best_auc = auc
                best_name = name
                best_metrics = fold_metrics
                best_model = model

        assert best_model is not None and best_name is not None
        best_model.fit(X, y)
        joblib.dump(best_model, MODELS_DIR / f"{target_key.lower()}.joblib")

        probs = best_model.predict_proba(X)[:, 1]
        holdout_style = {
            "accuracy": float(accuracy_score(y, (probs >= 0.5).astype(int))),
            "precision": float(precision_score(y, (probs >= 0.5).astype(int), zero_division=0)),
            "recall": float(recall_score(y, (probs >= 0.5).astype(int), zero_division=0)),
            "f1": float(f1_score(y, (probs >= 0.5).astype(int), zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)),
            "brier": float(brier_score_loss(y, probs)),
        }

        metrics[target_key] = {
            "display": meta["display"],
            "selected_model": best_name,
            "cross_validation": best_metrics,
            "train_reference": holdout_style,
            "positive_rate": float(y.mean()),
        }

    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved models to", MODELS_DIR)


if __name__ == "__main__":
    train_all()
