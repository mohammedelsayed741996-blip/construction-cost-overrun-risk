"""
Step 1 - Cleaning & validation across the four tables.

Every rule is counted and written to reports/data_quality_report.json,
together with referential-integrity and reconciliation checks
(e.g. BoQ total must equal the contract value - the same check a QS
does before certifying a payment).

Usage:  python src/clean_validate.py
"""
import json

import numpy as np
import pandas as pd

from config import (CLEAN_BOQ, CLEAN_CLAIMS, CLEAN_MONTHLY, CLEAN_PROJECTS,
                    PROCESSED_DIR, RAW_BOQ, RAW_CLAIMS, RAW_MONTHLY,
                    RAW_PROJECTS, REPORT_DIR)

dq: dict[str, int] = {}


def log(rule: str, n) -> None:
    dq[rule] = int(n)
    print(f"  - {rule:<52} {int(n):>6,}")


def main() -> None:
    print("[clean] running data-quality rules")
    p = pd.read_csv(RAW_PROJECTS)
    boq = pd.read_csv(RAW_BOQ)
    cl = pd.read_csv(RAW_CLAIMS)
    m = pd.read_csv(RAW_MONTHLY)

    # --- projects -----------------------------------------------------------
    bad = ~p.region.isin(["North", "East", "South", "West"])
    p["region"] = p.region.str.strip().str.title()
    log("projects: region spelling standardised", bad.sum())

    miss = p.design_completeness_pct.isna()
    p["design_completeness_pct"] = p.design_completeness_pct.fillna(
        p.groupby("project_type").design_completeness_pct.transform("median"))
    log("projects: design completeness imputed (type median)", miss.sum())

    # --- claims -------------------------------------------------------------
    neg = cl.amount_approved_eur < 0
    cl.loc[neg, "amount_approved_eur"] = np.nan          # unknown, not zero
    log("claims: negative approved amounts set to NULL", neg.sum())
    over = cl.amount_approved_eur > cl.amount_claimed_eur
    log("claims: approved > claimed (should be 0)", over.sum())

    # --- monthly progress ---------------------------------------------------
    before = len(m)
    m = m.drop_duplicates(["project_id", "month"])
    log("monthly: duplicate project-month rows removed", before - len(m))

    # --- integrity & reconciliation checks ----------------------------------
    orphan = ~cl.project_id.isin(p.project_id)
    log("claims: orphan project_id", orphan.sum())
    boq_total = (boq.bid_qty * boq.bid_rate_eur).groupby(boq.project_id).sum()
    diff = (p.set_index("project_id").contract_value_eur - boq_total).abs()
    log("projects: BoQ total != contract value (> EUR 1)", (diff > 1).sum())
    assert orphan.sum() == 0 and (diff > 1).sum() == 0

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    p.to_csv(CLEAN_PROJECTS, index=False)
    boq.to_csv(CLEAN_BOQ, index=False)
    cl.to_csv(CLEAN_CLAIMS, index=False)
    m.to_csv(CLEAN_MONTHLY, index=False)
    (REPORT_DIR / "data_quality_report.json").write_text(json.dumps(dq, indent=2))


if __name__ == "__main__":
    main()
