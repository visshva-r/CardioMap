from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import MODELS_DIR, RANDOM_STATE, TARGETS
from ml.data import feature_columns, load_raw_dataframe, target_series, transform_features
from ml.train import candidate_models

DOCS_FIG = ROOT / "docs" / "figures"
WEB_FIG = ROOT / "web" / "public" / "figures"


def _selected_model_name(target_key: str) -> str:
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    return metrics[target_key]["selected_model"]


def _build_model(model_name: str):
    models = candidate_models()
    if model_name not in models:
        raise KeyError(f"Unknown model {model_name}")
    from sklearn.base import clone

    return clone(models[model_name])


def generate_figures() -> dict[str, dict[str, float]]:
    df = load_raw_dataframe()
    import joblib

    preprocessor = joblib.load(MODELS_DIR / "preprocess.joblib")
    feat_cols = feature_columns(df)
    X = transform_features(preprocessor, feat_cols, df)

    DOCS_FIG.mkdir(parents=True, exist_ok=True)
    WEB_FIG.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for target_key, meta in TARGETS.items():
        y = target_series(df, target_key).values
        model_name = _selected_model_name(target_key)
        model = _build_model(model_name)

        probas = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probas >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y, preds)

        slug = target_key.lower()
        title_base = meta["display"]

        fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=120)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name).plot(
            ax=ax, color="#2563eb"
        )
        ax.set_title(f"{title_base} — OOF ROC (5-fold CV)")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=0.8)
        fig.tight_layout()
        for folder in (DOCS_FIG, WEB_FIG):
            fig.savefig(folder / f"{slug}_roc.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=120)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Negative", "Positive"]
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{title_base} — OOF confusion (threshold 0.5)")
        fig.tight_layout()
        for folder in (DOCS_FIG, WEB_FIG):
            fig.savefig(folder / f"{slug}_cm.png")
        plt.close(fig)

        summary[target_key] = {
            "selected_model": model_name,
            "oof_roc_auc_5fold": float(roc_auc),
            "oof_accuracy_5fold": float((preds == y).mean()),
        }
        print(f"{target_key}: OOF ROC-AUC={roc_auc:.3f} ({model_name})")

    (MODELS_DIR / "eval_plots_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    generate_figures()
    print("Wrote figures to", DOCS_FIG, "and", WEB_FIG)
