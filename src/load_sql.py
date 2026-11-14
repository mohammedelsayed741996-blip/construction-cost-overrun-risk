"""
Step 5 - Load the cost-control data mart into SQL and run KPI queries.
SQLite for zero setup; the DDL and queries also run on PostgreSQL.

Usage:  python src/load_sql.py
"""
import re
import sqlite3

import pandas as pd

from config import (CLEAN_BOQ, CLEAN_CLAIMS, CLEAN_MONTHLY, CLEAN_PROJECTS,
                    DB_FILE, RISK_FILE, SQL_DIR)

pd.set_option("display.width", 160)


def main() -> None:
    p = pd.read_csv(CLEAN_PROJECTS)
    con = sqlite3.connect(DB_FILE)
    con.executescript((SQL_DIR / "01_schema.sql").read_text())
    tables = {
        "dim_contractor": p[["contractor_id", "contractor_past_overrun_rate"]
                            ].drop_duplicates("contractor_id"),
        "dim_project": p.drop(columns=["contractor_past_overrun_rate"]),
        "fact_boq_item": pd.read_csv(CLEAN_BOQ),
        "fact_claim": pd.read_csv(CLEAN_CLAIMS),
        "fact_monthly_progress": pd.read_csv(CLEAN_MONTHLY),
        "fact_risk_score": pd.read_csv(RISK_FILE)[
            ["project_id", "overrun_probability", "predicted_overrun_pct", "risk_band"]],
    }
    for name, t in tables.items():
        t.to_sql(name, con, if_exists="append", index=False)
    print("[sql] loaded " + ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))

    sql = (SQL_DIR / "02_cost_control_kpis.sql").read_text()
    for name, q in re.findall(r"-- name: (\w+)\n(.*?;)", sql, re.S):
        print(f"\n== {name} ==")
        print(pd.read_sql_query(q, con).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
