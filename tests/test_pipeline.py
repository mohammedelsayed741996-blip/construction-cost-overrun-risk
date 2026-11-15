"""
Tests for the Construction Cost Overrun Early-Warning pipeline.

They run on the committed data and model outputs (no re-training), so the
suite is fast and gives the same answer on every scikit-learn version:

  * generator      same seed -> byte-identical raw tables
  * cleaning       every data-quality rule holds on the processed tables
  * features       no leakage: data after the checkpoint cannot change a feature
  * model outputs  metrics are internally consistent and match the README

Run:  pytest -q
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import features  # noqa: E402

RAW_TABLES = ["projects", "boq_items", "claims", "monthly_progress"]


@pytest.fixture(scope="module")
def clean():
    return {
        "projects": pd.read_csv(config.CLEAN_PROJECTS),
        "boq": pd.read_csv(config.CLEAN_BOQ),
        "claims": pd.read_csv(config.CLEAN_CLAIMS),
        "monthly": pd.read_csv(config.CLEAN_MONTHLY),
    }


@pytest.fixture(scope="module")
def metrics():
    return json.loads((config.REPORT_DIR / "model_metrics.json").read_text())


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def test_generator_is_deterministic(tmp_path, monkeypatch):
    """Re-running the generator with seed 42 reproduces the committed raw tables exactly."""
    sys.modules.pop("generate_data", None)
    gen = importlib.import_module("generate_data")      # fresh module = fresh RNG
    monkeypatch.setattr(gen, "RAW_DIR", tmp_path)
    for name, attr in zip(RAW_TABLES, ["RAW_PROJECTS", "RAW_BOQ", "RAW_CLAIMS", "RAW_MONTHLY"]):
        monkeypatch.setattr(gen, attr, tmp_path / f"{name}.csv")
    gen.main()
    for name in RAW_TABLES:
        new = pd.read_csv(tmp_path / f"{name}.csv")
        old = pd.read_csv(config.RAW_DIR / f"{name}.csv")
        # Money is rounded to cents; a half-cent can round either way on another
        # platform's floating point, so allow 1 cent.
        pd.testing.assert_frame_equal(new, old, check_exact=False, rtol=0, atol=0.011)


def test_portfolio_size():
    p = pd.read_csv(config.RAW_PROJECTS)
    assert len(p) == config.N_PROJECTS == 650
    assert p.contractor_id.nunique() <= config.N_CONTRACTORS


# --------------------------------------------------------------------------- #
# Cleaning & validation
# --------------------------------------------------------------------------- #
def test_boq_total_reconciles_to_contract_value(clean):
    """The QS check: bid quantities x bid rates must equal the contract value (BAC)."""
    boq = clean["boq"]
    total = (boq.bid_qty * boq.bid_rate_eur).groupby(boq.project_id).sum()
    bac = clean["projects"].set_index("project_id").contract_value_eur
    assert set(total.index) == set(bac.index)
    assert (bac - total.reindex(bac.index)).abs().max() <= 1.0


def test_regions_standardised(clean):
    assert set(clean["projects"].region) <= {"North", "East", "South", "West"}


def test_no_missing_design_completeness(clean):
    assert clean["projects"].design_completeness_pct.notna().all()


def test_claims_amounts_valid(clean):
    cl = clean["claims"]
    approved = cl.amount_approved_eur.dropna()
    assert (approved >= 0).all(), "negative approved amounts must be set to NULL"
    known = cl.dropna(subset=["amount_approved_eur"])
    assert (known.amount_approved_eur <= known.amount_claimed_eur + 1e-6).all()


def test_referential_integrity(clean):
    ids = set(clean["projects"].project_id)
    for table in ("boq", "claims", "monthly"):
        assert set(clean[table].project_id) <= ids, table


def test_no_duplicate_project_months(clean):
    assert not clean["monthly"].duplicated(["project_id", "month"]).any()


def test_data_quality_log_matches_cleaning():
    dq = json.loads((config.REPORT_DIR / "data_quality_report.json").read_text())
    raw_m = pd.read_csv(config.RAW_MONTHLY)
    clean_m = pd.read_csv(config.CLEAN_MONTHLY)
    assert dq["monthly: duplicate project-month rows removed"] == len(raw_m) - len(clean_m) == 60
    assert dq["claims: orphan project_id"] == 0
    assert dq["projects: BoQ total != contract value (> EUR 1)"] == 0


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
FEATURE_COLS = features.NUMERIC + features.CATEGORICAL


def _build_from(tmp_path, monkeypatch, projects, monthly, claims) -> pd.DataFrame:
    for name, df, attr in (("p", projects, "CLEAN_PROJECTS"),
                           ("m", monthly, "CLEAN_MONTHLY"),
                           ("c", claims, "CLEAN_CLAIMS")):
        path = tmp_path / f"{name}.csv"
        df.to_csv(path, index=False)
        monkeypatch.setattr(features, attr, path)
    return features.build_features().set_index("project_id")


def test_committed_features_are_reproducible(clean, tmp_path, monkeypatch):
    rebuilt = _build_from(tmp_path, monkeypatch, clean["projects"], clean["monthly"], clean["claims"])
    committed = pd.read_csv(config.FEATURE_FILE).set_index("project_id")
    pd.testing.assert_frame_equal(rebuilt[FEATURE_COLS], committed[FEATURE_COLS],
                                  check_exact=False, rtol=1e-9, check_dtype=False)


def test_no_leakage_after_checkpoint(clean, tmp_path, monkeypatch):
    """Scrambling everything that happens after the 30 % checkpoint must not change any feature."""
    p, m, cl = clean["projects"], clean["monthly"].copy(), clean["claims"].copy()
    base = _build_from(tmp_path, monkeypatch, p, m, cl)

    cp = np.ceil(p.set_index("project_id").planned_duration_months * config.CHECKPOINT).astype(int)
    late_m = m.month > m.project_id.map(cp)
    m.loc[late_m, ["actual_cost_eur", "earned_value_eur"]] *= 7.0
    late_sub = cl.submitted_month > cl.project_id.map(cp)
    cl.loc[late_sub, "amount_claimed_eur"] *= 9.0
    late_settle = cl.settled_month > cl.project_id.map(cp)
    cl.loc[late_settle, "amount_approved_eur"] = 0.0
    assert late_m.any() and late_sub.any() and late_settle.any()

    scrambled = _build_from(tmp_path, monkeypatch, p, m, cl)
    pd.testing.assert_frame_equal(scrambled[FEATURE_COLS], base[FEATURE_COLS])


def test_target_definition():
    f = pd.read_csv(config.FEATURE_FILE)
    expected = (f.final_cost_eur / f.contract_value_eur - 1 > config.OVERRUN_THRESHOLD).astype(int)
    assert (f.overrun_flag == expected).all()


# --------------------------------------------------------------------------- #
# Model outputs
# --------------------------------------------------------------------------- #
def test_threshold_policy_catches_80_percent():
    """pick_threshold returns the highest threshold whose recall is still >= 80 %."""
    from sklearn.metrics import recall_score

    train_model = importlib.import_module("train_model")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2_000)
    proba = np.clip(0.35 * y + rng.uniform(0, 0.65, y.size), 0, 1)
    thr = train_model.pick_threshold(y, proba)
    assert recall_score(y, proba >= thr) >= 0.80
    assert recall_score(y, proba >= thr + 0.01) < 0.80


def test_metrics_internally_consistent(metrics):
    c, r = metrics["classification"], metrics["regression"]
    assert c["n_train"] + c["n_test"] == config.N_PROJECTS
    assert c["best_model"] == max(c["cv_auc"], key=c["cv_auc"].get)
    assert r["best_model"] == min(r["all_mae_pp"], key=r["all_mae_pp"].get)
    assert c["test_recall"] >= 0.80                      # the business rule
    assert c["test_recall"] > c["evm_recall"]            # ML catches more overruns than EVM
    assert r["test_mae_pp"] < r["evm_mae_pp"]            # ML forecasts the overrun better
    assert r["evm_mean_bias_pp"] < 0                     # EVM under-forecasts (claims settle late)


def test_risk_register_complete():
    risk = pd.read_csv(config.RISK_FILE)
    assert len(risk) == config.N_PROJECTS and risk.project_id.is_unique
    assert risk.overrun_probability.between(0, 1).all()


def test_readme_headline_matches_metrics(metrics):
    """The numbers recruiters read in the README are the numbers in model_metrics.json."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    c, r = metrics["classification"], metrics["regression"]
    expected = {
        "recall": f"{c['test_recall']:.0%}".replace("%", " %"),
        "precision": f"{c['test_precision']:.0%}".replace("%", " %"),
        "auc": f"{c['test_auc']:.2f}",
        "mae": f"{r['test_mae_pp']:.2f} pp",
        "evm_mae": f"{r['evm_mae_pp']:.2f} pp",
    }
    glance = readme[readme.index("## Results at a glance"):]
    glance = glance[:glance.index("\n## ", 5)]
    for name, value in expected.items():
        assert value in glance or value.replace(" %", "%") in glance, (name, value)
