"""
Step 2 - Feature engineering at an early-warning checkpoint.

The core idea: score each project when only X % of its PLANNED duration
has elapsed (default 30 %). Only information available at that month is
used - no peeking at final cost - so the model is usable in real life.

Features
  static   : project type, procurement route, design completeness, ground
             risk, contractor track record, award year, size, duration
  EVM      : CPI = EV / AC, SPI = EV / PV at the checkpoint month
  claims   : number and value (% of contract) of claims submitted so far,
             approved-to-claimed ratio on claims settled so far
  trend    : actual-cost growth over the last 3 months vs earned value

Target
  overrun_pct  = final_cost / contract_value - 1
  overrun_flag = overrun_pct > 10 %

Usage:  python src/features.py
"""
import numpy as np
import pandas as pd

from config import (CHECKPOINT, CLEAN_CLAIMS, CLEAN_MONTHLY, CLEAN_PROJECTS,
                    FEATURE_FILE, OVERRUN_THRESHOLD)

STATIC_NUM = ["design_completeness_pct", "contractor_past_overrun_rate",
              "planned_duration_months", "log_contract_value", "award_year"]
EVM_NUM = ["cpi", "spi", "claims_count", "claims_value_pct",
           "claims_approval_ratio", "cost_trend_3m"]
CATEGORICAL = ["project_type", "procurement_route", "ground_risk"]
NUMERIC = STATIC_NUM + EVM_NUM


def build_features(checkpoint: float = CHECKPOINT) -> pd.DataFrame:
    p = pd.read_csv(CLEAN_PROJECTS)
    m = pd.read_csv(CLEAN_MONTHLY)
    cl = pd.read_csv(CLEAN_CLAIMS)

    p["checkpoint_month"] = np.ceil(p.planned_duration_months * checkpoint).astype(int)
    p["log_contract_value"] = np.log(p.contract_value_eur)

    # EVM snapshot at the checkpoint month
    snap = m.merge(p[["project_id", "checkpoint_month"]], on="project_id")
    now = snap[snap.month == snap.checkpoint_month].set_index("project_id")
    prev = snap[snap.month == snap.checkpoint_month - 3].set_index("project_id")
    p = p.set_index("project_id")
    p["cpi"] = now.earned_value_eur / now.actual_cost_eur
    p["spi"] = now.earned_value_eur / now.planned_value_eur
    d_ac = now.actual_cost_eur - prev.actual_cost_eur
    d_ev = now.earned_value_eur - prev.earned_value_eur
    p["cost_trend_3m"] = (d_ac / d_ev).reindex(p.index).fillna(1.0)

    # Claims known at the checkpoint month
    cl = cl.merge(p[["checkpoint_month"]], left_on="project_id", right_index=True)
    sub = cl[cl.submitted_month <= cl.checkpoint_month]
    settled = cl[cl.settled_month <= cl.checkpoint_month].dropna(
        subset=["amount_approved_eur"])
    g = sub.groupby("project_id")
    p["claims_count"] = g.size().reindex(p.index).fillna(0)
    p["claims_value_pct"] = (g.amount_claimed_eur.sum()
                             / p.contract_value_eur).reindex(p.index).fillna(0)
    s = settled.groupby("project_id")
    p["claims_approval_ratio"] = (s.amount_approved_eur.sum()
                                  / s.amount_claimed_eur.sum()
                                  ).reindex(p.index).fillna(0.5)  # prior

    # Targets
    p["overrun_pct"] = p.final_cost_eur / p.contract_value_eur - 1
    p["overrun_flag"] = (p.overrun_pct > OVERRUN_THRESHOLD).astype(int)
    p["delay_months"] = p.actual_duration_months - p.planned_duration_months
    # Classic EVM forecast for comparison: EAC = BAC / CPI
    p["evm_forecast_overrun_pct"] = 1 / p.cpi - 1
    return p.reset_index()


def main() -> None:
    df = build_features()
    df.to_csv(FEATURE_FILE, index=False)
    print(f"[features] {len(df)} projects scored at {CHECKPOINT:.0%} of planned "
          f"duration | overrun>10%: {df.overrun_flag.mean():.1%} | "
          f"median overrun {df.overrun_pct.median():.1%}")


if __name__ == "__main__":
    main()
