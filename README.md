# CardioMap — Track A (Multimodal AI Hackathon 2026)

Interactive **coronary artery disease (CAD)** and **LAD / LCX / RCA stenosis** prediction with a 3D heart view, SHAP-style explanations, and cross-validated metrics on the [Extension of Z-Alizadeh Sani dataset](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset) (303 patients).

> **Disclaimer:** Educational / decision-support prototype only — not a medical device or diagnostic substitute.

## Prerequisites

- Python 3.10+
- Node.js 18+

## Setup

```powershell
cd Project
pip install -r requirements.txt
python -m ml.train
python -m ml.explain
python -m ml.eval_plots
cd web
npm install
```

## Run

Terminal 1 — API:

```powershell
cd Project
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — frontend:

```powershell
cd Project/web
npm run dev
```

Open http://127.0.0.1:5173 — load a sample patient, click **Predict risk**, rotate the heart, click vessels for SHAP drivers.

## Leakage rule

Model inputs **never** include `LAD`, `LCX`, `RCA`, or `Cath` columns (see `ml/config.py`). Training aborts if those appear in the feature matrix.

## Repository layout

| Path | Purpose |
|------|---------|
| `ml/` | Preprocessing, training, SHAP explanations |
| `models/` | Joblib models, `metrics.json`, `schema.json` |
| `api/` | FastAPI backend |
| `web/` | React + React Three Fiber dashboard |
| `docs/report.md` | Technical report (export to PDF for submission) |
| `docs/figures/` | OOF ROC & confusion matrices (`python -m ml.eval_plots`) |

## Demo video checklist

1. Dataset citation and leakage exclusion  
2. Sample patient → predict → 3D vessel colors update  
3. Click LAD/LCX/RCA and SHAP panel  
4. Validation tab with CV metrics  
5. Disclaimer on screen  

## License

Hackathon submission — dataset remains under UCI terms; see `data/raw/SOURCE.md`.
