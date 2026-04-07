# payment-overhang-analysis

Analysis of grant payment overhang at the Hewlett Foundation, covering 10 years of grantmaking data (2016-2025). Overhang is the portion of a grant commitment that has been approved but not yet paid out within the award year. The script joins Salesforce exports across requests, payments, and organizations, then produces console summaries, CSVs, and visualizations across several dimensions.

## what it covers

- **overhang by program and year** -- how much of each year's awards carry forward, broken down by program
- **payment pacing** -- cumulative payout curves showing how quickly each cohort's grants get fully disbursed
- **gos vs non-gos** -- whether general operating support grants behave differently than project grants
- **expenditure responsibility** -- whether ER grants carry more overhang than non-ER grants
- **payment timing accuracy** -- the gap between scheduled and actual payment dates, by program and year

## data sources

Three CSV exports from Salesforce:

| file | description |
|------|-------------|
| `organizations_*.csv` | organization records (13,494 rows) |
| `requests_*.csv` | grant requests (26,114 rows) |
| `payments_*.csv` | payment transactions (42,235 rows) |

Place all three CSVs in the same directory as the script. DCA (Direct Charitable Activity) requests are excluded automatically. 
The `_*` refers to GMS report codes omitted here for data security.

## setup

**Requirements:** Python 3.x, pandas, matplotlib, numpy

```
pip install pandas matplotlib numpy
```

## usage

```
python payment_analysis.py
```

All outputs go to `outputs/`, which the script creates if it doesn't exist.

## outputs

### csvs

- `payments_10yr_joined.csv` -- full joined dataset filtered to the analysis window
- `overhang_by_program.csv` -- overhang by program and award year
- `overhang_by_program_gos.csv` -- same, split by GOS vs non-GOS
- `overhang_by_program_er.csv` -- same, split by ER vs non-ER
- `payment_timing_gaps.csv` -- per-payment gap between scheduled and actual date

### visualizations

| figure | description |
|--------|-------------|
| 01 | stacked bar: annual awards split by same-year payments and overhang |
| 02 | stacked area: overhang by program over time |
| 03 | heatmap: overhang rate by program and year |
| 04 | horizontal bar: cumulative overhang by program |
| 05 | line chart: overhang rate trend by program |
| 06 | grouped bar: awarded, paid, and overhang side by side |
| 07 | stacked bar: payments by year, by award cohort age |
| 08 | side-by-side: award-year view vs payment-year view |
| 09 | small multiples: GOS vs non-GOS overhang rate by program |
| 10 | grouped bar: aggregate GOS vs non-GOS overhang rate |
| 11 | small multiples: ER vs non-ER overhang rate by program |
| 12 | grouped bar: aggregate ER vs non-ER overhang rate |
| 13 | box plot: payment timing gap distribution by program |
| 14 | small multiples: median timing gap trend by program |
| 15 | band chart: foundation-wide timing gap with IQR |
| 16 | bar chart: share of payments more than 30 days late |

## notes

- the join between payments and requests uses normalized reference numbers, stripping suffixes like `-GRA` and leading zeros to match formats across tables
- overhang is calculated per cohort: total awarded in year X minus payments made on those same grants within year X
- payment timing gap = actual payment date minus scheduled date (positive means late, negative means early)
- the 2025 timing data may reflect incomplete year depending on when the export was pulled