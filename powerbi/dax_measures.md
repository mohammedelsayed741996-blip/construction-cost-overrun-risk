# Power BI – Cost Control Dashboard

## Model
```
dim_contractor 1─* dim_project 1─* fact_boq_item
                              1─* fact_claim
                              1─* fact_monthly_progress
                              1─1 fact_risk_score
```
Load `data/processed/construction.db` (ODBC) or the CSVs in `data/processed/`.

## Earned Value measures (latest month in filter context)

```DAX
BAC = SUM ( dim_project[contract_value_eur] )

Current Month = MAX ( fact_monthly_progress[month] )

PV = CALCULATE ( SUM ( fact_monthly_progress[planned_value_eur] ),
                 fact_monthly_progress[month] = [Current Month] )
EV = CALCULATE ( SUM ( fact_monthly_progress[earned_value_eur] ),
                 fact_monthly_progress[month] = [Current Month] )
AC = CALCULATE ( SUM ( fact_monthly_progress[actual_cost_eur] ),
                 fact_monthly_progress[month] = [Current Month] )

CPI = DIVIDE ( [EV], [AC] )
SPI = DIVIDE ( [EV], [PV] )
Cost Variance = [EV] - [AC]
Schedule Variance = [EV] - [PV]
EAC (BAC/CPI) = DIVIDE ( [BAC], [CPI] )
VAC = [BAC] - [EAC (BAC/CPI)]
```
> When several projects are selected, compute EV/AC per project first with
> `SUMX ( VALUES ( dim_project[project_id] ), ... )` so each project uses its own latest month.

## BoQ & claims measures

```DAX
Bid Value   = SUMX ( fact_boq_item, fact_boq_item[bid_qty]   * fact_boq_item[bid_rate_eur] )
Final Value = SUMX ( fact_boq_item, fact_boq_item[final_qty] * fact_boq_item[final_rate_eur] )

Quantity Effect = SUMX ( fact_boq_item,
    ( fact_boq_item[final_qty] - fact_boq_item[bid_qty] ) * fact_boq_item[bid_rate_eur] )
Rate Effect = SUMX ( fact_boq_item,
    ( fact_boq_item[final_rate_eur] - fact_boq_item[bid_rate_eur] ) * fact_boq_item[final_qty] )

Claims Submitted = SUM ( fact_claim[amount_claimed_eur] )
Claims Approved  = SUM ( fact_claim[amount_approved_eur] )
Approval Rate %  = DIVIDE ( [Claims Approved],
    CALCULATE ( [Claims Submitted], NOT ISBLANK ( fact_claim[amount_approved_eur] ) ) )

Final Cost = SUM ( dim_project[final_cost_eur] )
Overrun %  = DIVIDE ( [Final Cost], [BAC] ) - 1
```

## Risk measures

```DAX
High Risk Projects = CALCULATE ( COUNTROWS ( fact_risk_score ), fact_risk_score[risk_band] = "High" )
Value at Risk =
SUMX ( fact_risk_score,
       RELATED ( dim_project[contract_value_eur] ) * fact_risk_score[predicted_overrun_pct] )
```

## Report pages

| Page | Content |
|---|---|
| Portfolio | KPI cards (BAC, EAC, CPI, SPI, high-risk count), overrun by type & region |
| Project drill-through | PV/EV/AC S-curves, claims timeline, BoQ variance by trade |
| Risk register | Projects by `overrun_probability`, conditional formatting by band |
| Contractors | Overrun %, delay, approval rate per contractor |
