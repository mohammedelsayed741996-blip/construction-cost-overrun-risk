"""
Step 3 - Exploratory analysis of overruns, claims and BoQ variances.

Usage:  python src/eda.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config import CLEAN_BOQ, CLEAN_CLAIMS, CLEAN_MONTHLY, FEATURE_FILE, FIG_DIR  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(FEATURE_FILE)
    boq = pd.read_csv(CLEAN_BOQ)
    cl = pd.read_csv(CLEAN_CLAIMS)
    m = pd.read_csv(CLEAN_MONTHLY)

    # 1. Overrun distribution by project type
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=f, x="project_type", y=f.overrun_pct * 100, ax=ax,
                color="#4C72B0")
    ax.axhline(10, ls="--", c="r", lw=1)
    ax.set(title="Final cost overrun by project type (red = 10 % threshold)",
           xlabel="", ylabel="Overrun (%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "01_overrun_by_type.png", dpi=120)

    # 2. Design completeness vs overrun
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = pd.cut(f.design_completeness_pct, [39, 55, 70, 85, 100])
    (f.groupby(bins, observed=True).overrun_flag.mean() * 100).plot.bar(
        ax=ax, color="#C44E52")
    ax.set(title="Share of projects overrunning >10 % by design completeness",
           xlabel="Design completeness at tender (%)", ylabel="% of projects")
    fig.tight_layout(); fig.savefig(FIG_DIR / "02_design_completeness.png", dpi=120)

    # 3. BoQ cost variance by trade (quantity vs rate effect)
    boq["qty_effect"] = (boq.final_qty - boq.bid_qty) * boq.bid_rate_eur
    boq["rate_effect"] = (boq.final_rate_eur - boq.bid_rate_eur) * boq.final_qty
    t = boq.groupby("trade")[["qty_effect", "rate_effect"]].sum() / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t.sort_values("rate_effect").plot.barh(ax=ax, stacked=True,
                                           color=["#4C72B0", "#DD8452"])
    ax.set(title="BoQ variance by trade: quantity vs rate effect",
           xlabel="EUR million", ylabel="")
    fig.tight_layout(); fig.savefig(FIG_DIR / "03_boq_variance_by_trade.png", dpi=120)

    # 4. Claims: claimed vs approved by type
    c = cl.groupby("claim_type")[["amount_claimed_eur", "amount_approved_eur"]].sum() / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c.sort_values("amount_claimed_eur").plot.barh(ax=ax, color=["#8C8C8C", "#55A868"])
    ax.set(title="Claims: claimed vs approved by type", xlabel="EUR million", ylabel="")
    fig.tight_layout(); fig.savefig(FIG_DIR / "04_claims_by_type.png", dpi=120)

    # 5. Example EVM curves for one overrunning project
    pid = f.sort_values("overrun_pct").project_id.iloc[-20]
    e = m[m.project_id == pid].set_index("month")[
        ["planned_value_eur", "earned_value_eur", "actual_cost_eur"]] / 1e6
    fig, ax = plt.subplots(figsize=(8, 4.5))
    e.plot(ax=ax)
    ax.set(title=f"Earned value curves - project {pid}",
           xlabel="Month", ylabel="EUR million (cumulative)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "05_evm_curves_example.png", dpi=120)
    plt.close("all")

    print(f"[eda] BoQ net variance EUR {(boq.qty_effect.sum() + boq.rate_effect.sum())/1e6:.1f}M "
          f"(qty {boq.qty_effect.sum()/1e6:.1f}M, rate {boq.rate_effect.sum()/1e6:.1f}M) | "
          f"claims approved {cl.amount_approved_eur.sum()/cl.amount_claimed_eur.sum():.0%} of claimed")


if __name__ == "__main__":
    main()
