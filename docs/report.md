# CardioMap — Technical Report (Track A)

**Multimodal AI Hackathon 2026** · Solo submission · Extension of Z-Alizadeh Sani dataset (n=303)

## 1. Problem

Clinicians and patients struggle to interpret isolated risk percentages. CardioMap maps model outputs onto **LAD**, **LCX**, and **RCA** structures while explaining drivers with SHAP values on clinical features.

## 2. Data

- Source: UCI ML Repository ID 411 (`data/raw/SOURCE.md`).
- Targets: `Cath` (CAD vs Normal), vessel labels `LAD`/`LCX`/`RCA` (Stenotic vs Normal).
- **Leakage control:** all four label columns are removed from inputs for every model (`ml/config.py`).

## 3. Modeling

- Preprocessing: median imputation + scaling (numeric), one-hot encoding (categorical).
- Candidates: logistic regression, random forest, LightGBM with `class_weight=balanced`.
- Selection: repeated stratified 5-fold CV × 3 repeats; pick by mean ROC-AUC; refit on full data.
- Metrics exported to `models/metrics.json` (accuracy, precision, recall, F1, ROC-AUC, Brier on train reference).

## 4. Explainability

- Per-prediction SHAP (TreeExplainer / LinearExplainer) on transformed features.
- UI shows top 10 contributors with signed impact for the selected target (overall CAD or vessel).

## 5. Visualization & integration

- React Three Fiber heart with three clickable coronary tubes colored by stenosis probability.
- FastAPI serves schema-driven forms, samples, metrics, and `/api/predict`.
- Visible disclaimer: educational decision support only.

## 6. Limitations

- Small cohort (303) → wide CV variance; not validated on external sites.
- Anatomic mesh is schematic, not patient-specific imaging.
- LCX/RCA tasks are harder (lower AUC) due to class balance and feature signal.
- Not cleared for clinical use or emergency decisions.

## 7. Reproducibility

```powershell
pip install -r requirements.txt
python -m ml.train
python -m ml.explain
python -m uvicorn api.main:app --port 8000
cd web && npm install && npm run dev
```

Fixed random seed: 42.
