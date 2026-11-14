"""
Central configuration for the Construction Cost Overrun Early-Warning project.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
SQL_DIR = ROOT / "sql"

RAW_PROJECTS = RAW_DIR / "projects.csv"
RAW_BOQ = RAW_DIR / "boq_items.csv"
RAW_CLAIMS = RAW_DIR / "claims.csv"
RAW_MONTHLY = RAW_DIR / "monthly_progress.csv"

CLEAN_PROJECTS = PROCESSED_DIR / "projects_clean.csv"
CLEAN_BOQ = PROCESSED_DIR / "boq_items_clean.csv"
CLEAN_CLAIMS = PROCESSED_DIR / "claims_clean.csv"
CLEAN_MONTHLY = PROCESSED_DIR / "monthly_progress_clean.csv"
FEATURE_FILE = PROCESSED_DIR / "features_checkpoint.csv"
RISK_FILE = PROCESSED_DIR / "project_risk_scores.csv"
DB_FILE = PROCESSED_DIR / "construction.db"

RANDOM_STATE = 42
N_PROJECTS = 650
N_CONTRACTORS = 45

# Business definition of a "significant" overrun (final cost vs contract)
OVERRUN_THRESHOLD = 0.10

# Early-warning checkpoint: share of PLANNED duration elapsed when we score
CHECKPOINT = 0.30
CHECKPOINTS_TO_COMPARE = [0.2, 0.3, 0.4, 0.5, 0.6]

PROJECT_TYPES = {
    # type                      risk_offset  weight
    "Residential new build":      (0.0,       0.35),
    "Residential refurbishment":  (0.8,       0.25),
    "Commercial":                 (0.3,       0.15),
    "Infrastructure":             (0.5,       0.25),
}

TRADES = {
    # trade              share_of_value
    "Earthworks":          0.08,
    "Concrete & rebar":    0.25,
    "Masonry":             0.10,
    "Steel works":         0.07,
    "MEP":                 0.22,
    "Finishes":            0.15,
    "Roofing & facade":    0.08,
    "External works":      0.05,
}
CLAIM_TYPES = ["Design change", "Unforeseen ground", "Quantity variation",
               "Price escalation", "Delay / disruption", "Client instruction"]
