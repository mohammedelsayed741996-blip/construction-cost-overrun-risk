"""
Step 0 - Synthetic construction portfolio generator.

Produces four related tables that mirror what a contractor's or
developer's technical office actually holds:

  projects.csv          one row per project (contract, contractor, risk drivers)
  boq_items.csv         Bill of Quantities: bid vs final quantity & rate per item
  claims.csv            variation orders / contractor claims (claimed vs approved)
  monthly_progress.csv  monthly Planned Value, Earned Value, Actual Cost (EVM)

Domain logic built in (from quantity-surveying practice):
  * low design completeness at tender -> more quantity variations & claims
  * refurbishment & infrastructure carry more hidden-condition risk
  * re-measured BoQ contracts pass quantity risk to the client
  * 2021-2023 awards suffer material price escalation
  * claims are SETTLED LATE, so early cost data understates the final overrun
    (the reason classic EVM forecasts are too optimistic early on)

Usage:  python src/generate_data.py
"""
import numpy as np
import pandas as pd

from config import (CLAIM_TYPES, N_CONTRACTORS, N_PROJECTS, PROJECT_TYPES,
                    RANDOM_STATE, RAW_BOQ, RAW_CLAIMS, RAW_DIR, RAW_MONTHLY,
                    RAW_PROJECTS, TRADES)

rng = np.random.default_rng(RANDOM_STATE)


def s_curve(x: np.ndarray) -> np.ndarray:
    """Cumulative progress S-curve (smoothstep) on [0, 1]."""
    x = np.clip(x, 0, 1)
    return 3 * x ** 2 - 2 * x ** 3


def make_projects() -> pd.DataFrame:
    # Contractors: hidden quality + observable track record
    quality = rng.normal(0, 1, N_CONTRACTORS)
    contractors = pd.DataFrame({
        "contractor_id": [f"C{i:03d}" for i in range(1, N_CONTRACTORS + 1)],
        "contractor_quality": quality,
        "contractor_past_overrun_rate": np.clip(
            0.25 - 0.08 * quality + rng.normal(0, 0.05, N_CONTRACTORS), 0.02, 0.6
        ).round(2),
    })

    types = list(PROJECT_TYPES)
    w = np.array([PROJECT_TYPES[t][1] for t in types])
    n = N_PROJECTS
    df = pd.DataFrame({
        "project_id": [f"P{i:04d}" for i in range(1, n + 1)],
        "project_type": rng.choice(types, n, p=w / w.sum()),
        "region": rng.choice(["North", "East", "South", "West"], n),
        "contractor_id": rng.choice(contractors.contractor_id, n),
        "procurement_route": rng.choice(
            ["Lump sum", "Re-measured BoQ", "Design & build"], n,
            p=[0.4, 0.4, 0.2]),
        "design_completeness_pct": rng.integers(40, 101, n),
        "ground_risk": rng.choice(["Low", "Medium", "High"], n,
                                  p=[0.5, 0.35, 0.15]),
        "award_year": rng.integers(2015, 2025, n),
        "planned_duration_months": rng.integers(8, 37, n),
    }).merge(contractors, on="contractor_id")

    # Latent risk score (hidden; drives overrun, delay and claims)
    df["risk"] = (
        df.project_type.map({t: v[0] for t, v in PROJECT_TYPES.items()})
        + 1.6 * (100 - df.design_completeness_pct) / 60
        + df.ground_risk.map({"Low": 0, "Medium": 0.3, "High": 0.8})
        + df.procurement_route.map({"Lump sum": 0, "Re-measured BoQ": 0.4,
                                    "Design & build": 0.2})
        - 0.45 * df.contractor_quality
        + 0.4 * df.award_year.between(2021, 2023)
        + rng.normal(0, 0.45, n)
    )
    df["actual_duration_months"] = np.ceil(df.planned_duration_months * (
        1 + np.clip(0.03 + 0.07 * df.risk + rng.normal(0, 0.06, n), 0, 0.8)
    )).astype(int)
    return df


def make_boq(projects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in projects.itertuples():
        contract_value = rng.lognormal(np.log(6e6), 0.8)       # EUR
        lump = p.procurement_route == "Lump sum"
        infl = 0.07 if 2021 <= p.award_year <= 2023 else 0.015
        for trade, share in TRADES.items():
            for k in range(int(rng.integers(3, 8))):
                value = contract_value * share * rng.uniform(0.5, 1.5) / 5
                rate = rng.lognormal(np.log(120), 1.0)
                qty = value / rate
                q_var = rng.normal(-0.01 + 0.035 * p.risk, 0.07)
                q_var *= 0.35 if lump else 1.0                  # contractor carries qty risk
                r_var = infl * (1 if trade in ("Concrete & rebar", "Steel works",
                                               "MEP") else 0.4) + rng.normal(0, 0.02)
                rows.append({
                    "project_id": p.project_id,
                    "item_code": f"{trade[:3].upper()}-{k + 1:02d}",
                    "trade": trade,
                    "unit": rng.choice(["m3", "m2", "t", "m", "pcs"]),
                    "bid_qty": round(qty, 2),
                    "bid_rate_eur": round(rate, 2),
                    "final_qty": round(qty * (1 + q_var), 2),
                    "final_rate_eur": round(rate * (1 + r_var), 2),
                })
    return pd.DataFrame(rows)


def make_claims(projects: pd.DataFrame, bac: pd.Series) -> pd.DataFrame:
    rows, cid = [], 1
    for p in projects.itertuples():
        lam = max(0.15, 0.3 + 1.0 * p.risk) * p.actual_duration_months / 12
        for _ in range(rng.poisson(lam)):
            month = int(rng.integers(1, p.actual_duration_months + 1))
            claimed = bac[p.project_id] * rng.lognormal(np.log(0.010), 0.7)
            approve = np.clip(rng.beta(3, 3) + 0.06 * p.risk, 0.05, 1)
            rows.append({
                "claim_id": f"CL{cid:05d}",
                "project_id": p.project_id,
                "claim_type": rng.choice(CLAIM_TYPES),
                "submitted_month": month,
                # settlement takes 2-9 months -> cost impact shows up late
                "settled_month": min(month + int(rng.integers(2, 10)),
                                     p.actual_duration_months + 3),
                "amount_claimed_eur": round(claimed, 2),
                "amount_approved_eur": round(claimed * approve, 2),
            })
            cid += 1
    return pd.DataFrame(rows)


def make_monthly(projects, bac, boq_final, claims) -> pd.DataFrame:
    rows = []
    for p in projects.itertuples():
        B, F = bac[p.project_id], boq_final[p.project_id]
        cl = claims[claims.project_id == p.project_id]
        for t in range(1, p.actual_duration_months + 4):     # +3 close-out months
            ev = B * s_curve(t / p.actual_duration_months)
            pv = B * s_curve(t / p.planned_duration_months)
            ac = F * s_curve(t / p.actual_duration_months) * rng.normal(1, 0.015)
            ac += cl.loc[cl.settled_month <= t, "amount_approved_eur"].sum()
            rows.append({"project_id": p.project_id, "month": t,
                         "planned_value_eur": round(pv, 2),
                         "earned_value_eur": round(ev, 2),
                         "actual_cost_eur": round(ac, 2)})
    out = pd.DataFrame(rows)
    # cumulative cost can never fall: booking noise is smoothed out
    out["actual_cost_eur"] = out.groupby("project_id").actual_cost_eur.cummax()
    return out


def main() -> None:
    projects = make_projects()
    boq = make_boq(projects)
    bac = (boq.bid_qty * boq.bid_rate_eur).groupby(boq.project_id).sum()
    boq_final = (boq.final_qty * boq.final_rate_eur).groupby(boq.project_id).sum()
    claims = make_claims(projects, bac)
    monthly = make_monthly(projects, bac, boq_final, claims)

    # Ground truth: final cost = re-measured BoQ + approved claims
    approved = claims.groupby("project_id").amount_approved_eur.sum()
    projects["contract_value_eur"] = projects.project_id.map(bac).round(2)
    projects["final_cost_eur"] = (projects.project_id.map(boq_final)
                                  + projects.project_id.map(approved).fillna(0)).round(2)

    # ---- inject data-quality defects --------------------------------------
    n = len(projects)
    projects.loc[rng.choice(n, 12, replace=False), "design_completeness_pct"] = np.nan
    projects.loc[rng.choice(n, 8, replace=False), "region"] = "  east "
    claims.loc[rng.choice(len(claims), 15, replace=False), "amount_approved_eur"] = -1
    monthly = pd.concat([monthly, monthly.sample(60, random_state=1)])  # dup rows

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    projects.drop(columns=["risk", "contractor_quality"]).to_csv(RAW_PROJECTS, index=False)
    boq.to_csv(RAW_BOQ, index=False)
    claims.to_csv(RAW_CLAIMS, index=False)
    monthly.to_csv(RAW_MONTHLY, index=False)
    print(f"[generate] projects={len(projects)}, boq_items={len(boq):,}, "
          f"claims={len(claims):,}, monthly_rows={len(monthly):,}, "
          f"total contract value EUR {bac.sum()/1e9:.2f}bn")


if __name__ == "__main__":
    main()
