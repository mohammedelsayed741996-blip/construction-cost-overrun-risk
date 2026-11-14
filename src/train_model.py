"""
Step 4 - Early-warning models.

A) Classifier : will the project overrun its contract value by > 10 %?
B) Regressor  : how large will the final overrun be (%)?

Both are benchmarked against the classic Earned Value forecast
(EAC = BAC / CPI), which is what most project controls teams use today.
C) Sensitivity: how early can we warn? Models re-trained at checkpoints
   of 20 % ... 60 % of planned duration.

Usage:  python src/train_model.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (mean_absolute_error, precision_score,  # noqa: E402
                             recall_score, roc_auc_score, f1_score)
from sklearn.model_selection import (StratifiedKFold,  # noqa: E402
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from config import (CHECKPOINTS_TO_COMPARE, FEATURE_FILE, FIG_DIR,  # noqa: E402
                    MODEL_DIR, OVERRUN_THRESHOLD, RANDOM_STATE, REPORT_DIR,
                    RISK_FILE)
from features import CATEGORICAL, NUMERIC, build_features  # noqa: E402

CV = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def prep() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])


CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE),
}


def pick_threshold(y, proba, min_recall=0.80) -> float:
    """Highest threshold that still catches >= 80 % of overrunning projects
    (missing an overrun is costlier than reviewing a false alarm)."""
    for t in np.arange(0.9, 0.05, -0.01):
        if recall_score(y, proba >= t) >= min_recall:
            return float(round(t, 2))
    return 0.5


def main() -> None:
    df = pd.read_csv(FEATURE_FILE)
    X, y, y_reg = df[NUMERIC + CATEGORICAL], df.overrun_flag, df.overrun_pct
    X_tr, X_te, y_tr, y_te, r_tr, r_te, i_tr, i_te = train_test_split(
        X, y, y_reg, df.index, test_size=0.25, stratify=y,
        random_state=RANDOM_STATE)

    # ---------------- A) classification --------------------------------------
    cv_auc = {}
    for name, clf in CLASSIFIERS.items():
        pipe = Pipeline([("prep", prep()), ("model", clf)])
        cv_auc[name] = cross_val_score(pipe, X_tr, y_tr, cv=CV,
                                       scoring="roc_auc").mean()
        print(f"  {name:<22} CV ROC-AUC = {cv_auc[name]:.3f}")
    best_name = max(cv_auc, key=cv_auc.get)
    clf = Pipeline([("prep", prep()), ("model", CLASSIFIERS[best_name])])

    oof = cross_val_predict(clf, X_tr, y_tr, cv=CV, method="predict_proba")[:, 1]
    thr = pick_threshold(y_tr, oof)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    flag = proba >= thr

    evm_score = df.loc[i_te, "evm_forecast_overrun_pct"]
    evm_flag = evm_score > OVERRUN_THRESHOLD
    over_eur = (df.final_cost_eur - df.contract_value_eur).clip(lower=0).loc[i_te]

    cls = {
        "best_model": best_name, "threshold": thr, "cv_auc": cv_auc,
        "test_auc": roc_auc_score(y_te, proba),
        "test_recall": recall_score(y_te, flag),
        "test_precision": precision_score(y_te, flag),
        "test_f1": f1_score(y_te, flag),
        "evm_auc": roc_auc_score(y_te, evm_score),
        "evm_recall": recall_score(y_te, evm_flag),
        "evm_precision": precision_score(y_te, evm_flag, zero_division=0),
        "share_overrun_eur_flagged_ml": float(over_eur[flag].sum() / over_eur.sum()),
        "share_overrun_eur_flagged_evm": float(over_eur[evm_flag.values].sum() / over_eur.sum()),
        "share_projects_flagged_ml": float(flag.mean()),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    print(f"[classify] {best_name} | test AUC={cls['test_auc']:.3f} "
          f"recall={cls['test_recall']:.0%} precision={cls['test_precision']:.0%} "
          f"@thr={thr} | EVM rule AUC={cls['evm_auc']:.3f} "
          f"recall={cls['evm_recall']:.0%} | overrun EUR caught: ML "
          f"{cls['share_overrun_eur_flagged_ml']:.0%} vs EVM "
          f"{cls['share_overrun_eur_flagged_evm']:.0%}")

    # ---------------- B) regression ------------------------------------------
    regs = {"Ridge": Ridge(1.0),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_depth=4,
                random_state=RANDOM_STATE)}
    reg_res = {}
    for name, r in regs.items():
        pipe = Pipeline([("prep", prep()), ("model", r)]).fit(X_tr, r_tr)
        reg_res[name] = mean_absolute_error(r_te, pipe.predict(X_te))
        regs[name] = pipe
    best_reg = min(reg_res, key=reg_res.get)
    reg = {"best_model": best_reg,
           "test_mae_pp": reg_res[best_reg] * 100,
           "all_mae_pp": {k: v * 100 for k, v in reg_res.items()},
           "evm_mae_pp": mean_absolute_error(r_te, evm_score) * 100,
           "evm_mean_bias_pp": float((evm_score - r_te).mean() * 100)}
    print(f"[regress] {best_reg} MAE={reg['test_mae_pp']:.2f} pp | "
          f"EVM (BAC/CPI) MAE={reg['evm_mae_pp']:.2f} pp, "
          f"bias={reg['evm_mean_bias_pp']:+.2f} pp")

    # ---------------- C) how early can we warn? ------------------------------
    curve = {}
    for c in CHECKPOINTS_TO_COMPARE:
        d = build_features(c)
        pipe = Pipeline([("prep", prep()), ("model", CLASSIFIERS[best_name])])
        ml = cross_val_score(pipe, d[NUMERIC + CATEGORICAL], d.overrun_flag,
                             cv=CV, scoring="roc_auc").mean()
        curve[f"{c:.0%}"] = {"ml_auc": ml, "evm_auc": roc_auc_score(
            d.overrun_flag, d.evm_forecast_overrun_pct)}
    print("[checkpoints] " + " | ".join(
        f"{k}: ML {v['ml_auc']:.2f} / EVM {v['evm_auc']:.2f}" for k, v in curve.items()))

    # ---------------- figures & artefacts ------------------------------------
    cdf = pd.DataFrame(curve).T
    fig, ax = plt.subplots(figsize=(7, 4))
    cdf.plot(ax=ax, marker="o")
    ax.set(title="Early-warning power over project life",
           xlabel="Share of planned duration elapsed", ylabel="ROC-AUC")
    ax.legend(["ML model", "EVM rule (BAC/CPI)"])
    fig.tight_layout(); fig.savefig(FIG_DIR / "06_auc_by_checkpoint.png", dpi=120)

    imp = permutation_importance(clf, X_te, y_te, scoring="roc_auc",
                                 n_repeats=10, random_state=RANDOM_STATE)
    imp_s = pd.Series(imp.importances_mean, X_te.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp_s.plot.barh(ax=ax, color="#4C72B0")
    ax.set(title=f"Permutation importance ({best_name})", xlabel="Drop in ROC-AUC")
    fig.tight_layout(); fig.savefig(FIG_DIR / "07_feature_importance.png", dpi=120)
    plt.close("all")

    # Risk register for the whole portfolio (out-of-fold, so no project is
    # scored by a model trained on itself) -> Power BI
    all_oof = cross_val_predict(clf, X, y, cv=CV, method="predict_proba")[:, 1]
    out = df[["project_id", "project_type", "contractor_id", "contract_value_eur",
              "cpi", "spi", "claims_count", "claims_value_pct",
              "evm_forecast_overrun_pct", "overrun_pct", "overrun_flag"]].copy()
    out["overrun_probability"] = all_oof.round(3)
    out["predicted_overrun_pct"] = regs[best_reg].predict(X).round(4)
    out["risk_band"] = pd.cut(out.overrun_probability, [-0.01, 0.3, thr, 1.0],
                              labels=["Low", "Watch", "High"])
    out.sort_values("overrun_probability", ascending=False).to_csv(RISK_FILE, index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "overrun_classifier.joblib")
    joblib.dump(regs[best_reg], MODEL_DIR / "overrun_regressor.joblib")
    imp_top = imp_s.sort_values(ascending=False).head(6).round(4).to_dict()
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(
        {"classification": cls, "regression": reg, "checkpoints": curve,
         "top_features": imp_top}, indent=2, default=float))


if __name__ == "__main__":
    main()
