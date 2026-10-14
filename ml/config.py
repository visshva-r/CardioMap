from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42

# Must never appear as model inputs (Track A leakage rule).
LEAKAGE_COLUMNS = frozenset({"LAD", "LCX", "RCA", "Cath"})

TARGETS = {
    "Cath": {"column": "Cath", "positive_label": "CAD", "display": "Overall CAD"},
    "LAD": {"column": "LAD", "positive_label": "Stenotic", "display": "LAD stenosis"},
    "LCX": {"column": "LCX", "positive_label": "Stenotic", "display": "LCX stenosis"},
    "RCA": {"column": "RCA", "positive_label": "Stenotic", "display": "RCA stenosis"},
}

CV_SPLITS = 5
CV_REPEATS = 3

SAMPLE_PATIENT_INDICES = [0, 50, 120, 200, 280]
