# Construction Cost Overrun Early-Warning System

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![SQL](https://img.shields.io/badge/SQL-PostgreSQL%20%7C%20SQLite-lightgrey) ![Power BI](https://img.shields.io/badge/Power%20BI-DAX-yellow)

> Flags projects that will overrun their contract value by more than 10 % when only **30 % of the planned schedule has elapsed**, using Bills of Quantities, contractor claims and Earned Value data, and benchmarks the model against the standard EVM forecast (EAC = BAC / CPI).

**Data note:** all data is **synthetic** (`src/generate_data.py`), modelled on quantity-surveying practice; real project cost data is confidential. All figures below are from a reproducible run (seed 42).

---

## 1. Business objectives

- Identify high-risk projects **early enough to act** (re-scope, renegotiate, add contingency).
- Forecast the size of the final overrun more accurately than the traditional EVM formula.
- Explain *where* cost growth comes from: quantity changes, price escalation or claims.
- Provide a SQL data mart and Power BI measures for monthly cost reviews.

## 2. Domain bridge

Built by a former technical-office / quantity-surveying engineer who prepared 40+ BoQs and payment certificates and resolved 25+ contractor claims. Two field observations drive the design:

1. **Claims are settled months after they are submitted**, so actual cost early in a project understates the final cost. The EVM forecast is therefore systematically optimistic.
2. **Low design completeness at tender** and re-measured contracts produce quantity variations; 2021–2023 awards carry price escalation.

## 3. Data (4 related tables)

| Table | Rows | Grain / key columns |
|---|---|---|
| `projects` | 650 | type, region, contractor, procurement route, design completeness, ground risk, award year, planned/actual duration, contract value (BAC), final cost |
| `boq_items` | 26,131 | project × item: trade, unit, bid qty/rate, final qty/rate |
| `claims` | 2,903 | type, submitted & settled month, claimed & approved amount |
| `monthly_progress` | 18,755 | project × month: Planned Value, Earned Value, Actual Cost |

Portfolio: €5.54bn contract value, 45 contractors, 4 project types. Injected defects: inconsistent region spelling, missing design completeness, negative approved amounts, duplicate monthly rows.

## 4. Methodology

```
generate_data → clean_validate → features (checkpoint) → eda → train_model → load_sql → Power BI
```

| Step | Detail |
|---|---|
| **Cleaning & validation** | Standardised 8 region values, imputed 12 missing design-completeness values, set 15 invalid approved amounts to NULL, removed 60 duplicate monthly rows; referential-integrity check and **BoQ total = contract value reconciliation** for all 650 projects |
| **Features (no leakage)** | Only data known at 30 % of planned duration: CPI, SPI, 3-month cost trend, claims count & value to date, approval ratio of settled claims, plus project and contractor attributes |
| **Classification** | Logistic Regression, Random Forest, HistGradientBoosting – stratified 5-fold CV; threshold chosen to catch ≥ 80 % of overruns |
| **Regression** | Ridge vs HistGradientBoosting for final overrun %, vs EVM forecast |
| **Early-warning curve** | Re-trained at 20 %, 30 %, 40 %, 50 %, 60 % of planned duration |
| **SQL / BI** | Star schema, EVM/contractor/claims/BoQ-variance queries (CTEs, window functions), DAX measures |

## 5. Results (checkpoint = 30 % of planned duration)

**Classification – overrun > 10 %** (hold-out: 163 projects)

| Approach | ROC-AUC | Recall | Precision |
|---|---|---|---|
| **Random Forest (selected; CV AUC 0.930)** | **0.964** | **87 %** | **93 %** |
| EVM rule (EAC/BAC – 1 > 10 %) | 0.942 | 66 % | 96 % |

**Regression – final overrun %**

| Approach | MAE | Bias |
|---|---|---|
| **HistGradientBoosting** | **2.17 pp** | – |
| EVM forecast (BAC / CPI) | 3.29 pp | −2.23 pp (under-forecasts) |

→ **34 % lower forecast error** than the EVM formula, and **21 pp more overrunning projects caught** (87 % vs 66 % recall) at the cost of slightly lower precision (93 % vs 96 %).

## 6. Key insights

- **CPI is the strongest single signal**, but claim value to date and the 3-month cost trend add the information EVM misses.
- The EVM formula **under-forecasts the final overrun by 2.2 pp on average** because claims settle late.
- The ML model beats the EVM rule at every checkpoint from 20 % to 50 % of the schedule; by 60 % both converge (curve in `reports/figures/06_auc_by_checkpoint.png`).
- BoQ variance across the portfolio: **€321.5M**, of which €197.2M quantity effect and €124.3M rate effect; MEP and concrete & rebar show the highest variance (≈ 6.5–6.8 %).
- Only **62 %** of claimed amounts were approved – a negotiation lever per claim type.

## 7. Repository structure

```
├── data/raw/ , data/processed/   # generated tables, features, risk scores, construction.db
├── models/                        # trained classifier & regressor (joblib)
├── powerbi/dax_measures.md        # EVM, BoQ, claims and risk measures
├── reports/figures/               # EDA + model figures
├── reports/*.json                 # metrics and data-quality log
├── sql/01_schema.sql              # data-mart DDL
├── sql/02_cost_control_kpis.sql   # KPI queries
├── src/                           # pipeline steps
└── run_pipeline.py
```

## 8. Installation

```bash
git clone https://github.com/<your-username>/construction-cost-overrun-risk.git
cd construction-cost-overrun-risk
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py        # ~40 s
```

## 9. Limitations & next steps

- Synthetic data; on real data, validate on projects completed after the training period (time-based split).
- Hold-out set is small (163 projects); CV AUC (0.93) is the more conservative estimate.
- Next: survival analysis for delay, SHAP explanations per project, monthly re-scoring job feeding the Power BI risk register.

---
*Author: Mohamed Abdelrasoul – MSc Data Analytics (BSBI Berlin) · [LinkedIn](https://linkedin.com/in/mohamed-abdelrasoul-b07002173)*
