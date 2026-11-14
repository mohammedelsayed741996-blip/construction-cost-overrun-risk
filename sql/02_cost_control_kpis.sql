-- =====================================================================
-- Cost-control KPI queries. "-- name: <id>" markers let
-- src/load_sql.py execute and print each query.
-- =====================================================================

-- name: evm_at_checkpoint
-- EVM status of every project at 30 % of planned duration, worst CPI first
WITH chk AS (
    SELECT  p.project_id, p.project_type, p.contract_value_eur,
            CAST(p.planned_duration_months * 0.3 + 0.999 AS INTEGER) AS chk_month
    FROM    dim_project p
)
SELECT  c.project_id, c.project_type,
        ROUND(c.contract_value_eur / 1e6, 2)                      AS bac_m_eur,
        ROUND(m.earned_value_eur / m.actual_cost_eur, 3)          AS cpi,
        ROUND(m.earned_value_eur / m.planned_value_eur, 3)        AS spi,
        ROUND(c.contract_value_eur * m.actual_cost_eur
              / m.earned_value_eur / 1e6, 2)                      AS eac_m_eur
FROM    chk c
JOIN    fact_monthly_progress m
        ON m.project_id = c.project_id AND m.month = c.chk_month
ORDER BY cpi
LIMIT 10;

-- name: contractor_league
-- Contractor performance: overrun, approved claims, delay (window rank)
WITH perf AS (
    SELECT  p.contractor_id,
            COUNT(*)                                                    AS projects,
            SUM(p.contract_value_eur)                                   AS bac,
            SUM(p.final_cost_eur)                                       AS final_cost,
            AVG(p.actual_duration_months - p.planned_duration_months)   AS avg_delay_m
    FROM    dim_project p
    GROUP BY p.contractor_id
)
SELECT  contractor_id, projects,
        ROUND(100.0 * (final_cost / bac - 1), 1)    AS overrun_pct,
        ROUND(avg_delay_m, 1)                       AS avg_delay_months,
        RANK() OVER (ORDER BY final_cost / bac DESC) AS worst_rank
FROM    perf
WHERE   projects >= 10
ORDER BY worst_rank
LIMIT 10;

-- name: claims_by_type
-- Claim approval ratio by type (only settled claims with a known amount)
SELECT  claim_type,
        COUNT(*)                                                    AS claims,
        ROUND(SUM(amount_claimed_eur)  / 1e6, 2)                    AS claimed_m_eur,
        ROUND(SUM(amount_approved_eur) / 1e6, 2)                    AS approved_m_eur,
        ROUND(100.0 * SUM(amount_approved_eur) / SUM(amount_claimed_eur), 1) AS approval_pct
FROM    fact_claim
WHERE   amount_approved_eur IS NOT NULL
GROUP BY claim_type
ORDER BY approved_m_eur DESC;

-- name: boq_variance_by_trade
-- Quantity vs rate variance per trade (classic QS variance analysis)
SELECT  trade,
        ROUND(SUM((final_qty - bid_qty) * bid_rate_eur) / 1e6, 2)         AS qty_effect_m_eur,
        ROUND(SUM((final_rate_eur - bid_rate_eur) * final_qty) / 1e6, 2)  AS rate_effect_m_eur,
        ROUND(100.0 * (SUM(final_qty * final_rate_eur)
                      / SUM(bid_qty * bid_rate_eur) - 1), 2)              AS variance_pct
FROM    fact_boq_item
GROUP BY trade
ORDER BY variance_pct DESC;

-- name: risk_register
-- High-risk projects for the monthly cost review
SELECT  p.project_id, p.project_type, p.contractor_id,
        ROUND(p.contract_value_eur / 1e6, 2)          AS bac_m_eur,
        r.overrun_probability,
        ROUND(100 * r.predicted_overrun_pct, 1)       AS predicted_overrun_pct,
        r.risk_band
FROM    fact_risk_score r
JOIN    dim_project p ON p.project_id = r.project_id
WHERE   r.risk_band = 'High'
ORDER BY r.overrun_probability DESC, bac_m_eur DESC
LIMIT 10;
