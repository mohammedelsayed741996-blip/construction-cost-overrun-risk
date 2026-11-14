"""
Run the full pipeline end-to-end:
generate -> clean -> features -> EDA -> train -> SQL.

Usage:  python run_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import clean_validate  # noqa: E402
import eda  # noqa: E402
import features  # noqa: E402
import generate_data  # noqa: E402
import load_sql  # noqa: E402
import train_model  # noqa: E402

if __name__ == "__main__":
    for step in (generate_data, clean_validate, features, eda,
                 train_model, load_sql):
        step.main()
