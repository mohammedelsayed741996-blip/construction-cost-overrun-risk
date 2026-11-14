-- =====================================================================
-- Construction cost-control data mart (PostgreSQL / SQLite compatible)
--
--   dim_contractor 1─* dim_project 1─* fact_boq_item
--                                 1─* fact_claim
--                                 1─* fact_monthly_progress
--                                 1─1 fact_risk_score
-- =====================================================================
DROP TABLE IF EXISTS fact_risk_score;
DROP TABLE IF EXISTS fact_monthly_progress;
DROP TABLE IF EXISTS fact_claim;
DROP TABLE IF EXISTS fact_boq_item;
DROP TABLE IF EXISTS dim_project;
DROP TABLE IF EXISTS dim_contractor;

CREATE TABLE dim_contractor (
    contractor_id                VARCHAR(8) PRIMARY KEY,
    contractor_past_overrun_rate NUMERIC(4,2)
);

CREATE TABLE dim_project (
    project_id               VARCHAR(8) PRIMARY KEY,
    contractor_id            VARCHAR(8) NOT NULL REFERENCES dim_contractor(contractor_id),
    project_type             VARCHAR(40) NOT NULL,
    region                   VARCHAR(10),
    procurement_route        VARCHAR(20),
    design_completeness_pct  NUMERIC(5,1) CHECK (design_completeness_pct BETWEEN 0 AND 100),
    ground_risk              VARCHAR(10),
    award_year               INTEGER,
    planned_duration_months  INTEGER NOT NULL,
    actual_duration_months   INTEGER,
    contract_value_eur       NUMERIC(14,2) NOT NULL,   -- BAC
    final_cost_eur           NUMERIC(14,2)
);

CREATE TABLE fact_boq_item (
    project_id      VARCHAR(8)  NOT NULL REFERENCES dim_project(project_id),
    item_code       VARCHAR(12) NOT NULL,
    trade           VARCHAR(30) NOT NULL,
    unit            VARCHAR(5),
    bid_qty         NUMERIC(14,2),
    bid_rate_eur    NUMERIC(12,2),
    final_qty       NUMERIC(14,2),
    final_rate_eur  NUMERIC(12,2),
    PRIMARY KEY (project_id, item_code)
);

CREATE TABLE fact_claim (
    claim_id             VARCHAR(10) PRIMARY KEY,
    project_id           VARCHAR(8) NOT NULL REFERENCES dim_project(project_id),
    claim_type           VARCHAR(30),
    submitted_month      INTEGER,
    settled_month        INTEGER,
    amount_claimed_eur   NUMERIC(14,2) NOT NULL,
    amount_approved_eur  NUMERIC(14,2),               -- NULL = not yet known
    CHECK (amount_approved_eur IS NULL OR amount_approved_eur <= amount_claimed_eur)
);

CREATE TABLE fact_monthly_progress (
    project_id         VARCHAR(8) NOT NULL REFERENCES dim_project(project_id),
    month              INTEGER    NOT NULL,
    planned_value_eur  NUMERIC(14,2),
    earned_value_eur   NUMERIC(14,2),
    actual_cost_eur    NUMERIC(14,2),
    PRIMARY KEY (project_id, month)
);

CREATE TABLE fact_risk_score (
    project_id             VARCHAR(8) PRIMARY KEY REFERENCES dim_project(project_id),
    overrun_probability    NUMERIC(5,3),
    predicted_overrun_pct  NUMERIC(7,4),
    risk_band              VARCHAR(10)
);
