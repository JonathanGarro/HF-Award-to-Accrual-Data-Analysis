import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)

ORGS_FILE = 'organizations_00OUf00000EHY8VMAX.csv'
REQUESTS_FILE = 'requests_00OUf000005GbLiMAK.csv'
PAYMENTS_FILE = 'payments_00OUf00000HZP8GMAX.csv'

START_YEAR = 2016
END_YEAR = 2025



# load data
orgs = pd.read_csv(ORGS_FILE, encoding='latin-1')
requests = pd.read_csv(REQUESTS_FILE, encoding='latin-1')
payments = pd.read_csv(PAYMENTS_FILE, encoding='latin-1')

print(f"loaded organizations: {len(orgs):,} rows")
print(f"loaded requests: {len(requests):,} rows")
print(f"loaded payments: {len(payments):,} rows")

# clean payments

# parse dates
payments['Payment Date'] = pd.to_datetime(
    payments['Tranx/Paymt Date'], format='mixed', errors='coerce'
)
payments['Scheduled Date Parsed'] = pd.to_datetime(
    payments['Scheduled Date'], format='mixed', errors='coerce'
)

# prefer actual payment date, fall back to scheduled
payments['Effective Date'] = payments['Payment Date'].fillna(
    payments['Scheduled Date Parsed']
)
payments['Effective Year'] = payments['Effective Date'].dt.year

# ensure amount is numeric
payments['Payment Amount'] = pd.to_numeric(payments['Amount'], errors='coerce')

# normalize reference numbers for joining
# normalize by stripping the suffix and leading zeros from the numeric portion

def normalize_ref(ref):
    """normalize grant reference numbers to a common format"""
    if pd.isna(ref):
        return None
    parts = str(ref).split('-')
    if len(parts) >= 2:
        year = parts[0]
        num = parts[1].lstrip('0') or '0'
        return f'{year}-{num}'
    return str(ref)

requests['ref_norm'] = requests['Request: Reference Number'].apply(normalize_ref)

# strip the suffix (e.g., -GRA, -DCA) from payment request references
payments['ref_base'] = payments['Request'].str.extract(r'^(\d{4}-\d+)')[0]
payments['ref_norm'] = payments['ref_base'].apply(normalize_ref)

# join payments to requests
df = payments.merge(
    requests,
    on='ref_norm',
    how='left',
    suffixes=('_pay', '_req')
)

matched = df['Request: Reference Number'].notna().sum()
unmatched = df['Request: Reference Number'].isna().sum()
print(f"\npayments joined to requests: {len(df):,} rows")
print(f"  matched: {matched:,}")
print(f"  unmatched: {unmatched:,}")

# join orgs on organization name from the requests table
df = df.merge(
    orgs,
    left_on='Organization: Organization Name',
    right_on='Organization Name',
    how='left',
    suffixes=('', '_org')
)

print(f"full joined dataset: {len(df):,} rows")
print(f"  with org details: {df['Organization Name'].notna().sum():,}")

# exclude DCA requests
dca_count = (df['Request Type'] == 'DCA Request').sum()
df = df[df['Request Type'] != 'DCA Request']
print(f"\nexcluded {dca_count:,} DCA payment records")
print(f"  remaining: {len(df):,} rows")

# filter years (config at top)
df_10yr = df[
    (df['Effective Year'] >= START_YEAR) & (df['Effective Year'] <= END_YEAR)
].copy()

print(f"\nfiltered to {START_YEAR}-{END_YEAR}: {len(df_10yr):,} payment records")
print(f"  total payment amount: ${df_10yr['Payment Amount'].sum():,.0f}")
print(f"  unique grants: {df_10yr['ref_norm'].nunique():,}")
print(f"  unique organizations: {df_10yr['Organization: Organization Name'].nunique():,}")

# payments by year
print("\npayments by year:")
year_summary = (
    df_10yr.groupby('Effective Year')['Payment Amount']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'Total Amount', 'count': 'Payment Count'})
)
year_summary['Total Amount'] = year_summary['Total Amount'].apply(
    lambda x: f"${x:,.0f}"
)
print(year_summary.to_string())

# save joined dataset
df_10yr.to_csv('outputs/payments_10yr_joined.csv', index=False)
print(f"\nsaved joined dataset to outputs/payments_10yr_joined.csv")


# overhang analysis
# for each award year, overhang = total amount awarded that year minus the
# portion paid out within that same calendar year. this represents the
# future liability created by each year's grant commitments — the money
# that "hangs over" into subsequent years.

print("\n" + "=" * 70)
print("OVERHANG ANALYSIS BY PROGRAM (YEAR BY YEAR)")
print("=" * 70)

# parse award dates and amounts on the requests table
requests['Award Date'] = pd.to_datetime(
    requests['President Approval/Award Date'], format='mixed', errors='coerce'
)
requests['Award Year'] = requests['Award Date'].dt.year
requests['Award Amount'] = pd.to_numeric(requests['Amount'], errors='coerce')

grants_only = requests[requests['Request Type'] == 'Grant'].copy()

# join payments to grants to get award year and program on each payment
pay_grants = payments.merge(
    grants_only[['ref_norm', 'Top Level Primary Program', 'Award Amount', 'Award Year']],
    on='ref_norm',
    how='inner'
)

# build cohort overhang table
# for each award year, by program:
#   awarded = sum of grant amounts awarded that year
#   same-year paid = sum of payments made in the same year as the award
#   overhang = awarded - same-year paid

overhang_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    # grants awarded this year, by program
    cohort = grants_only[grants_only['Award Year'] == year]
    awards_by_prog = cohort.groupby('Top Level Primary Program')['Award Amount'].sum()

    # payments on this cohort's grants that occurred in the same year
    cohort_pays = pay_grants[
        (pay_grants['Award Year'] == year) &
        (pay_grants['Effective Year'] == year)
    ]
    same_year_paid_by_prog = cohort_pays.groupby('Top Level Primary Program')['Payment Amount'].sum()

    for prog in awards_by_prog.index:
        awarded = awards_by_prog.get(prog, 0)
        paid = same_year_paid_by_prog.get(prog, 0)
        overhang = awarded - paid
        overhang_rows.append({
            'Award Year': year,
            'Program': prog,
            'Awarded': awarded,
            'Paid Same Year': paid,
            'Overhang': overhang,
            'Overhang %': (overhang / awarded * 100) if awarded > 0 else 0,
            'Grant Count': len(cohort[cohort['Top Level Primary Program'] == prog])
        })

overhang_df = pd.DataFrame(overhang_rows)

# summary by program across all years
print(f"\noverhang summary by program ({START_YEAR}-{END_YEAR}):")
print("-" * 90)

prog_summary = (
    overhang_df.groupby('Program')
    .agg(
        total_awarded=('Awarded', 'sum'),
        total_same_year_paid=('Paid Same Year', 'sum'),
        total_overhang=('Overhang', 'sum'),
        total_grants=('Grant Count', 'sum')
    )
    .sort_values('total_overhang', ascending=False)
)
prog_summary['overhang_pct'] = (
    prog_summary['total_overhang'] / prog_summary['total_awarded'] * 100
)

for prog, row in prog_summary.iterrows():
    print(
        f"  {prog:35s}"
        f"  awarded: ${row['total_awarded']:>13,.0f}"
        f"  paid yr 1: ${row['total_same_year_paid']:>13,.0f}"
        f"  overhang: ${row['total_overhang']:>13,.0f}"
        f"  ({row['overhang_pct']:>4.1f}%)"
    )

totals = prog_summary.sum()
overall_pct = totals['total_overhang'] / totals['total_awarded'] * 100
print("-" * 90)
print(
    f"  {'TOTAL':35s}"
    f"  awarded: ${totals['total_awarded']:>13,.0f}"
    f"  paid yr 1: ${totals['total_same_year_paid']:>13,.0f}"
    f"  overhang: ${totals['total_overhang']:>13,.0f}"
    f"  ({overall_pct:>4.1f}%)"
)

# year-by-year overhang (all programs combined)
print(f"\nyear-by-year overhang (all programs):")
print("-" * 90)

year_totals = (
    overhang_df.groupby('Award Year')
    .agg(
        awarded=('Awarded', 'sum'),
        paid_same_year=('Paid Same Year', 'sum'),
        overhang=('Overhang', 'sum'),
        grants=('Grant Count', 'sum')
    )
)
year_totals['overhang_pct'] = (
    year_totals['overhang'] / year_totals['awarded'] * 100
)

for year, row in year_totals.iterrows():
    print(
        f"  {int(year)}:"
        f"  awarded: ${row['awarded']:>13,.0f}"
        f"  paid yr 1: ${row['paid_same_year']:>13,.0f}"
        f"  overhang: ${row['overhang']:>13,.0f}"
        f"  ({row['overhang_pct']:>4.1f}%)"
        f"  [{int(row['grants']):,} grants]"
    )

# overhang by program by year (pivot table)
top_programs = prog_summary.head(8).index.tolist()

print(f"\noverhang by year for top programs:")
print("-" * 90)

pivot_oh = (
    overhang_df[overhang_df['Program'].isin(top_programs)]
    .pivot_table(index='Award Year', columns='Program', values='Overhang', aggfunc='sum')
    [top_programs]
    .fillna(0)
)
pivot_oh['TOTAL'] = pivot_oh.sum(axis=1)

pivot_display = pivot_oh.copy()
for col in pivot_display.columns:
    pivot_display[col] = pivot_display[col].apply(lambda x: f"${x:>12,.0f}")
print(pivot_display.to_string())

# overhang rate by program by year
print(f"\noverhang rate (%) by year for top programs:")
print("-" * 90)

pivot_pct = (
    overhang_df[overhang_df['Program'].isin(top_programs)]
    .pivot_table(index='Award Year', columns='Program', values='Overhang %', aggfunc='mean')
    [top_programs]
    .fillna(0)
)

pivot_pct_display = pivot_pct.copy()
for col in pivot_pct_display.columns:
    pivot_pct_display[col] = pivot_pct_display[col].apply(lambda x: f"{x:>6.1f}%")
print(pivot_pct_display.to_string())

# payment pacing
print(f"\npayment pacing: how grants awarded each year get paid over time")
print("(% of award amount paid by year-end, cumulative)")
print("-" * 90)

for year in range(START_YEAR, END_YEAR + 1):
    cohort_awarded = grants_only[grants_only['Award Year'] == year]['Award Amount'].sum()
    if cohort_awarded == 0:
        continue

    cohort_payments = pay_grants[pay_grants['Award Year'] == year]
    pacing = []
    cum_paid = 0
    for pay_year in range(year, END_YEAR + 1):
        yr_paid = cohort_payments[
            cohort_payments['Effective Year'] == pay_year
        ]['Payment Amount'].sum()
        cum_paid += yr_paid
        pct = cum_paid / cohort_awarded * 100
        pacing.append(f"yr{pay_year - year}: {pct:>5.1f}%")

    print(f"  {year} cohort (${cohort_awarded / 1e6:>6.1f}M):  {'  '.join(pacing)}")

# save overhang data
overhang_df.to_csv('outputs/overhang_by_program.csv', index=False)
print(f"\nsaved overhang data to outputs/overhang_by_program.csv")


#visualizations

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# color palette and helpers
# order programs by total overhang for consistent color assignment
prog_totals = (
    overhang_df.groupby('Program')['Overhang']
    .sum()
    .sort_values(ascending=False)
)
all_programs = prog_totals.index.tolist()

palette = [
    '#2C5F2D', '#97BC62', '#0B3D91', '#6BAED6', '#E07B39',
    '#D4A843', '#8B4513', '#7B2D8E', '#C0392B', '#2C3E50',
    '#888888'
]
color_map = {prog: palette[i % len(palette)] for i, prog in enumerate(all_programs)}


def millions_fmt(x, _):
    return f'${x / 1e6:.0f}M'


def save_fig(fig, name):
    fig.savefig(f'outputs/{name}', dpi=200, bbox_inches='tight', facecolor='white')
    print(f'  saved outputs/{name}')
    plt.close(fig)


# aggregate yearly totals for charts
yearly = overhang_df.groupby('Award Year').agg(
    awarded=('Awarded', 'sum'),
    paid=('Paid Same Year', 'sum'),
    overhang=('Overhang', 'sum')
).reset_index()


# figure 1: stacked bar — awarded vs paid same year vs overhang
fig, ax = plt.subplots(figsize=(12, 6))

x = yearly['Award Year']
bar_width = 0.7

ax.bar(x, yearly['paid'], width=bar_width, label='Paid Same Year',
       color='#2C5F2D', edgecolor='white', linewidth=0.5)
ax.bar(x, yearly['overhang'], width=bar_width, bottom=yearly['paid'],
       label='Overhang', color='#E07B39', edgecolor='white', linewidth=0.5)

for _, row in yearly.iterrows():
    pct = row['overhang'] / row['awarded'] * 100
    ax.text(row['Award Year'], row['awarded'] + 5e6, f'{pct:.0f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E07B39')

ax.set_xlabel('Award Year', fontsize=11)
ax.set_ylabel('Amount', fontsize=11)
ax.set_title('Annual Grant Awards: Paid Same Year vs. Overhang',
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax.set_xticks(x)
ax.legend(loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, '01_awards_vs_overhang_by_year.png')


# figure 2: stacked area — overhang by program over time
top_n = 7
top_viz_programs = prog_totals.head(top_n).index.tolist()

pivot = (
    overhang_df.pivot_table(
        index='Award Year', columns='Program', values='Overhang', aggfunc='sum'
    )
    .fillna(0)
)
other_cols = [c for c in pivot.columns if c not in top_viz_programs]
pivot['Other'] = pivot[other_cols].sum(axis=1)
plot_cols = top_viz_programs + ['Other']
pivot_plot = pivot[plot_cols]

fig, ax = plt.subplots(figsize=(12, 6))

colors = [color_map.get(p, '#888888') for p in top_viz_programs] + ['#CCCCCC']
ax.stackplot(pivot_plot.index, *[pivot_plot[col] for col in plot_cols],
             labels=plot_cols, colors=colors, alpha=0.85)

ax.set_xlabel('Award Year', fontsize=11)
ax.set_ylabel('Overhang Amount', fontsize=11)
ax.set_title('Grant Overhang by Program (Stacked Area)',
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax.set_xticks(pivot_plot.index)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, '02_overhang_by_program_stacked.png')


# figure 3: heatmap — overhang rate (%) by program and year
heat_data = overhang_df.pivot_table(
    index='Program', columns='Award Year',
    values=['Overhang', 'Awarded'], aggfunc='sum'
).fillna(0)

overhang_pct_heat = (heat_data['Overhang'] / heat_data['Awarded'] * 100).fillna(0)

# sort programs by mean overhang rate
prog_order = overhang_pct_heat.mean(axis=1).sort_values(ascending=True).index
overhang_pct_heat = overhang_pct_heat.loc[prog_order]

fig, ax = plt.subplots(figsize=(13, 7))

im = ax.imshow(overhang_pct_heat.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=65)

ax.set_xticks(range(len(overhang_pct_heat.columns)))
ax.set_xticklabels(overhang_pct_heat.columns.astype(int), fontsize=10)
ax.set_yticks(range(len(overhang_pct_heat.index)))
ax.set_yticklabels(overhang_pct_heat.index, fontsize=10)

for i in range(len(overhang_pct_heat.index)):
    for j in range(len(overhang_pct_heat.columns)):
        val = overhang_pct_heat.iloc[i, j]
        text_color = 'white' if val > 40 else 'black'
        if val > 0:
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

ax.set_title('Overhang Rate (%) by Program and Award Year',
             fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, shrink=0.8, label='Overhang %')

save_fig(fig, '03_overhang_rate_heatmap.png')


# figure 4: horizontal bar — total overhang by program (10-year sum)
viz_prog_summary = overhang_df.groupby('Program').agg(
    total_awarded=('Awarded', 'sum'),
    total_overhang=('Overhang', 'sum'),
    total_paid=('Paid Same Year', 'sum')
).sort_values('total_overhang', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))

y_pos = range(len(viz_prog_summary))
ax.barh(y_pos, viz_prog_summary['total_overhang'],
        color=[color_map.get(p, '#888888') for p in viz_prog_summary.index],
        edgecolor='white', linewidth=0.5, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(viz_prog_summary.index, fontsize=10)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax.set_xlabel(f'Total Overhang ({START_YEAR}-{END_YEAR})', fontsize=11)
ax.set_title('Cumulative Overhang by Program (10-Year Total)',
             fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for i, (prog, row) in enumerate(viz_prog_summary.iterrows()):
    pct = (row['total_overhang'] / row['total_awarded'] * 100
           if row['total_awarded'] > 0 else 0)
    ax.text(row['total_overhang'] + 3e6, i, f'{pct:.0f}%',
            va='center', fontsize=9, color='#555555')

save_fig(fig, '04_total_overhang_by_program.png')


# figure 5: line chart — overhang rate trend for major programs
line_programs = [p for p in top_viz_programs if p != 'Special Projects']

fig, ax = plt.subplots(figsize=(12, 6))

for prog in line_programs:
    if prog in overhang_pct_heat.index:
        prog_data = overhang_pct_heat.loc[prog]
        mask = heat_data['Awarded'].loc[prog] > 0
        years = prog_data[mask].index
        vals = prog_data[mask].values
        ax.plot(years, vals, marker='o', linewidth=2, markersize=5,
                label=prog, color=color_map.get(prog, '#888888'))

# foundation-wide average
yearly_pct = yearly['overhang'] / yearly['awarded'] * 100
ax.plot(yearly['Award Year'], yearly_pct, marker='s', linewidth=2.5,
        markersize=6, label='Foundation Average', color='black',
        linestyle='--', zorder=10)

ax.set_xlabel('Award Year', fontsize=11)
ax.set_ylabel('Overhang Rate (%)', fontsize=11)
ax.set_title('Overhang Rate Trend by Program',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(START_YEAR, END_YEAR + 1))
ax.set_ylim(0, 70)
ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

save_fig(fig, '05_overhang_rate_trend.png')


# figure 6: grouped bar — awarded, paid, overhang side by side
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(yearly))
w = 0.25

ax.bar(x - w, yearly['awarded'], width=w, label='Awarded',
       color='#0B3D91', edgecolor='white')
ax.bar(x, yearly['paid'], width=w, label='Paid Same Year',
       color='#2C5F2D', edgecolor='white')
ax.bar(x + w, yearly['overhang'], width=w, label='Overhang',
       color='#E07B39', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(yearly['Award Year'].astype(int))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax.set_xlabel('Award Year', fontsize=11)
ax.set_ylabel('Amount', fontsize=11)
ax.set_title('Awards, Same-Year Payments, and Overhang by Year',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, '06_awards_paid_overhang_grouped.png')


# figure 7: stacked bar — total payments each year by award cohort age
# this shows the PAYMENT-YEAR view: all cash going out the door in a given
# year, broken down by how old the underlying grant award is.
# "yr0" = paid on grants awarded that same year (same-year payments)
# "yr1" = paid on grants awarded the prior year (overhang coming due)
# "yr2+" = paid on grants awarded two or more years ago

# join payments to requests to get award year on each payment
pay_with_award_yr = payments.merge(
    grants_only[['ref_norm', 'Award Year']],
    on='ref_norm',
    how='inner'
)

cohort_age_rows = []
for pay_year in range(START_YEAR, END_YEAR + 1):
    yr_pays = pay_with_award_yr[
        pay_with_award_yr['Effective Year'] == pay_year
    ].copy()
    yr_pays['Cohort Age'] = pay_year - yr_pays['Award Year']

    # bucket into yr0, yr1, yr2+
    for _, row in yr_pays.iterrows():
        age = row['Cohort Age']
        if age == 0:
            bucket = 'Same Year (yr0)'
        elif age == 1:
            bucket = 'Prior Year (yr1)'
        else:
            bucket = 'Older (yr2+)'
        cohort_age_rows.append({
            'Payment Year': pay_year,
            'Cohort Bucket': bucket,
            'Payment Amount': row['Payment Amount']
        })

cohort_age_df = pd.DataFrame(cohort_age_rows)
cohort_pivot = (
    cohort_age_df.groupby(['Payment Year', 'Cohort Bucket'])['Payment Amount']
    .sum()
    .unstack(fill_value=0)
)

# column order
bucket_order = ['Same Year (yr0)', 'Prior Year (yr1)', 'Older (yr2+)']
cohort_pivot = cohort_pivot[[b for b in bucket_order if b in cohort_pivot.columns]]

bucket_colors = {
    'Same Year (yr0)': '#2C5F2D',
    'Prior Year (yr1)': '#6BAED6',
    'Older (yr2+)': '#D4A843'
}

fig, ax = plt.subplots(figsize=(12, 6))

bottom = np.zeros(len(cohort_pivot))
for col in cohort_pivot.columns:
    ax.bar(cohort_pivot.index, cohort_pivot[col], width=0.7, bottom=bottom,
           label=col, color=bucket_colors.get(col, '#888888'),
           edgecolor='white', linewidth=0.5)
    bottom += cohort_pivot[col].values

# add total labels on top
for pay_year in cohort_pivot.index:
    total = cohort_pivot.loc[pay_year].sum()
    ax.text(pay_year, total + 5e6, f'${total / 1e6:.0f}M',
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

ax.set_xlabel('Payment Year', fontsize=11)
ax.set_ylabel('Amount Paid', fontsize=11)
ax.set_title('Total Payments by Year, by Award Cohort Age',
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax.set_xticks(cohort_pivot.index)
ax.legend(loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, '07_payments_by_cohort_age.png')


# figure 8: side-by-side — award-year view vs payment-year view

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# left: award-year view (same as figure 1)
ax1.bar(yearly['Award Year'], yearly['paid'], width=0.7,
        label='Paid Same Year', color='#2C5F2D', edgecolor='white', linewidth=0.5)
ax1.bar(yearly['Award Year'], yearly['overhang'], width=0.7,
        bottom=yearly['paid'], label='Overhang Created',
        color='#E07B39', edgecolor='white', linewidth=0.5)

for _, row in yearly.iterrows():
    ax1.text(row['Award Year'], row['awarded'] + 5e6,
             f'${row["awarded"] / 1e6:.0f}M',
             ha='center', va='bottom', fontsize=8, color='#333333')

ax1.set_xlabel('Award Year', fontsize=11)
ax1.set_ylabel('Amount', fontsize=11)
ax1.set_title('Award-Year View\n(What was committed each year)',
              fontsize=12, fontweight='bold')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax1.set_xticks(yearly['Award Year'])
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# right: payment-year view (same as figure 7)
bottom = np.zeros(len(cohort_pivot))
for col in cohort_pivot.columns:
    ax2.bar(cohort_pivot.index, cohort_pivot[col], width=0.7, bottom=bottom,
            label=col, color=bucket_colors.get(col, '#888888'),
            edgecolor='white', linewidth=0.5)
    bottom += cohort_pivot[col].values

for pay_year in cohort_pivot.index:
    total = cohort_pivot.loc[pay_year].sum()
    ax2.text(pay_year, total + 5e6, f'${total / 1e6:.0f}M',
             ha='center', va='bottom', fontsize=8, color='#333333')

ax2.set_xlabel('Payment Year', fontsize=11)
ax2.set_title('Payment-Year View\n(What actually went out the door)',
              fontsize=12, fontweight='bold')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(millions_fmt))
ax2.set_xticks(cohort_pivot.index)
ax2.tick_params(axis='x', rotation=45)
ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Two Views of the Same Money',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()

save_fig(fig, '08_award_vs_payment_year_comparison.png')

# gos vs non-gos overhang analysis

print("\n" + "=" * 70)
print("GOS VS NON-GOS OVERHANG ANALYSIS")
print("=" * 70)

# tag grants with gos flag
grants_only['GOS'] = grants_only['Type of Support'].apply(
    lambda x: 'GOS' if x == 'General Support/Organization' else 'Non-GOS'
)

# rebuild payment join with gos flag included
pay_grants_gos = payments.merge(
    grants_only[['ref_norm', 'Top Level Primary Program', 'Award Amount',
                 'Award Year', 'GOS']],
    on='ref_norm',
    how='inner'
)

# build overhang table by program, year, and gos status
gos_overhang_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    cohort = grants_only[grants_only['Award Year'] == year]
    for gos_label in ['GOS', 'Non-GOS']:
        cohort_gos = cohort[cohort['GOS'] == gos_label]
        awards_by_prog = cohort_gos.groupby('Top Level Primary Program')['Award Amount'].sum()

        cohort_pays = pay_grants_gos[
            (pay_grants_gos['Award Year'] == year) &
            (pay_grants_gos['Effective Year'] == year) &
            (pay_grants_gos['GOS'] == gos_label)
        ]
        paid_by_prog = cohort_pays.groupby('Top Level Primary Program')['Payment Amount'].sum()

        for prog in awards_by_prog.index:
            awarded = awards_by_prog.get(prog, 0)
            paid = paid_by_prog.get(prog, 0)
            overhang = awarded - paid
            gos_overhang_rows.append({
                'Award Year': year,
                'Program': prog,
                'GOS': gos_label,
                'Awarded': awarded,
                'Paid Same Year': paid,
                'Overhang': overhang,
                'Overhang %': (overhang / awarded * 100) if awarded > 0 else 0,
                'Grant Count': len(cohort_gos[cohort_gos['Top Level Primary Program'] == prog])
            })

gos_overhang_df = pd.DataFrame(gos_overhang_rows)

gos_summary = (
    gos_overhang_df.groupby(['Program', 'GOS'])
    .agg(
        total_awarded=('Awarded', 'sum'),
        total_overhang=('Overhang', 'sum'),
        total_grants=('Grant Count', 'sum')
    )
)
gos_summary['overhang_pct'] = (
    gos_summary['total_overhang'] / gos_summary['total_awarded'] * 100
)

print("\noverhang summary by program and support type:")
print("-" * 95)
for prog in prog_summary.index:
    for gos_label in ['GOS', 'Non-GOS']:
        if (prog, gos_label) in gos_summary.index:
            row = gos_summary.loc[(prog, gos_label)]
            print(
                f"  {prog:35s}  {gos_label:8s}"
                f"  awarded: ${row['total_awarded']:>13,.0f}"
                f"  overhang: ${row['total_overhang']:>13,.0f}"
                f"  ({row['overhang_pct']:>4.1f}%)"
                f"  [{int(row['total_grants']):,} grants]"
            )

# save gos overhang data
gos_overhang_df.to_csv('outputs/overhang_by_program_gos.csv', index=False)
print(f"\nsaved gos overhang data to outputs/overhang_by_program_gos.csv")

# hewlett program colors
hewlett_colors = {
    'Education': '#1A254E',
    'Environment': '#778218',
    'Gender Equity & Governance': '#E89829',
    'Performing Arts': '#4A0F3E',
    'U.S. Democracy': '#3E006C',
    'Philanthropy': '#214240',
    'Culture Race and Equity': '#C15811',
    'Economy and Society': '#184319',
    'Special Projects': '#C5D0C5',
    'SBAC': '#414B3F',
}

# figure 9: small multiples — gos vs non-gos overhang rate by program over time
top_gos_programs = prog_summary.head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
axes_flat = axes.flatten()

for i, prog in enumerate(top_gos_programs):
    ax = axes_flat[i]
    prog_color = hewlett_colors.get(prog, '#888888')

    for gos_label, ls, alpha in [('GOS', '-', 1.0), ('Non-GOS', '--', 0.7)]:
        subset = gos_overhang_df[
            (gos_overhang_df['Program'] == prog) &
            (gos_overhang_df['GOS'] == gos_label)
        ].copy()

        # only plot years where there was actual award activity
        subset = subset[subset['Awarded'] > 0]
        if subset.empty:
            continue

        ax.plot(subset['Award Year'], subset['Overhang %'],
                marker='o', linewidth=2, markersize=4,
                linestyle=ls, alpha=alpha, color=prog_color,
                label=gos_label)

    ax.set_title(prog, fontsize=10, fontweight='bold', color=prog_color)
    ax.set_xticks(range(START_YEAR, END_YEAR + 1))
    ax.set_xticklabels([str(y)[2:] for y in range(START_YEAR, END_YEAR + 1)],
                       fontsize=8)
    ax.set_ylim(0, 75)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if i == 0:
        ax.legend(fontsize=8, loc='upper right')
    if i % 4 == 0:
        ax.set_ylabel('Overhang Rate (%)', fontsize=10)

fig.suptitle('Overhang Rate by Program: GOS vs Non-GOS Grants',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])

save_fig(fig, '09_gos_vs_nongos_overhang_by_program.png')


# figure 10: grouped bar — aggregate gos vs non-gos overhang rate by program
agg = (
    gos_overhang_df.groupby(['Program', 'GOS'])
    .agg(awarded=('Awarded', 'sum'), overhang=('Overhang', 'sum'))
)
agg['pct'] = agg['overhang'] / agg['awarded'] * 100

# reshape for plotting
bar_data = agg['pct'].unstack('GOS').reindex(prog_summary.index).dropna(how='all')

fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(bar_data))
bar_h = 0.35

gos_vals = bar_data.get('GOS', pd.Series(0, index=bar_data.index)).fillna(0)
nongos_vals = bar_data.get('Non-GOS', pd.Series(0, index=bar_data.index)).fillna(0)

prog_colors = [hewlett_colors.get(p, '#888888') for p in bar_data.index]

ax.barh(y_pos + bar_h / 2, gos_vals, height=bar_h,
        color=prog_colors, edgecolor='white', linewidth=0.5, label='GOS')
ax.barh(y_pos - bar_h / 2, nongos_vals, height=bar_h,
        color=prog_colors, edgecolor='white', linewidth=0.5,
        alpha=0.45, label='Non-GOS', hatch='///')

# add percentage labels
for i, prog in enumerate(bar_data.index):
    gv = gos_vals.get(prog, 0)
    nv = nongos_vals.get(prog, 0)
    if gv > 0:
        ax.text(gv + 0.5, i + bar_h / 2, f'{gv:.0f}%',
                va='center', fontsize=9, color='#333333')
    if nv > 0:
        ax.text(nv + 0.5, i - bar_h / 2, f'{nv:.0f}%',
                va='center', fontsize=9, color='#333333')

ax.set_yticks(y_pos)
ax.set_yticklabels(bar_data.index, fontsize=10)
ax.set_xlabel('Overhang Rate (%)', fontsize=11)
ax.set_title('Aggregate Overhang Rate by Program: GOS vs Non-GOS (2016-2025)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(gos_vals.max(), nongos_vals.max()) + 8)

save_fig(fig, '10_gos_vs_nongos_aggregate_by_program.png')


# expenditure responsibility overhang analysis

print("\n" + "=" * 70)
print("EXPENDITURE RESPONSIBILITY OVERHANG ANALYSIS")
print("=" * 70)

# tag grants with er flag
grants_only['ER'] = grants_only['Expenditure Responsibility'].apply(
    lambda x: 'ER' if x == 1 else 'Non-ER'
)

# rebuild payment join with er flag included
pay_grants_er = payments.merge(
    grants_only[['ref_norm', 'Top Level Primary Program', 'Award Amount',
                 'Award Year', 'ER']],
    on='ref_norm',
    how='inner'
)

# build overhang table by program, year, and er status
er_overhang_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    cohort = grants_only[grants_only['Award Year'] == year]
    for er_label in ['ER', 'Non-ER']:
        cohort_er = cohort[cohort['ER'] == er_label]
        awards_by_prog = cohort_er.groupby('Top Level Primary Program')['Award Amount'].sum()

        cohort_pays = pay_grants_er[
            (pay_grants_er['Award Year'] == year) &
            (pay_grants_er['Effective Year'] == year) &
            (pay_grants_er['ER'] == er_label)
        ]
        paid_by_prog = cohort_pays.groupby('Top Level Primary Program')['Payment Amount'].sum()

        for prog in awards_by_prog.index:
            awarded = awards_by_prog.get(prog, 0)
            paid = paid_by_prog.get(prog, 0)
            overhang = awarded - paid
            er_overhang_rows.append({
                'Award Year': year,
                'Program': prog,
                'ER': er_label,
                'Awarded': awarded,
                'Paid Same Year': paid,
                'Overhang': overhang,
                'Overhang %': (overhang / awarded * 100) if awarded > 0 else 0,
                'Grant Count': len(cohort_er[cohort_er['Top Level Primary Program'] == prog])
            })

er_overhang_df = pd.DataFrame(er_overhang_rows)

er_summary = (
    er_overhang_df.groupby(['Program', 'ER'])
    .agg(
        total_awarded=('Awarded', 'sum'),
        total_overhang=('Overhang', 'sum'),
        total_grants=('Grant Count', 'sum')
    )
)
er_summary['overhang_pct'] = (
    er_summary['total_overhang'] / er_summary['total_awarded'] * 100
)

print("\noverhang summary by program and expenditure responsibility:")
print("-" * 95)
for prog in prog_summary.index:
    for er_label in ['ER', 'Non-ER']:
        if (prog, er_label) in er_summary.index:
            row = er_summary.loc[(prog, er_label)]
            print(
                f"  {prog:35s}  {er_label:8s}"
                f"  awarded: ${row['total_awarded']:>13,.0f}"
                f"  overhang: ${row['total_overhang']:>13,.0f}"
                f"  ({row['overhang_pct']:>4.1f}%)"
                f"  [{int(row['total_grants']):,} grants]"
            )

# save er overhang data
er_overhang_df.to_csv('outputs/overhang_by_program_er.csv', index=False)
print(f"\nsaved er overhang data to outputs/overhang_by_program_er.csv")

# figure 11: small multiples — er vs non-er overhang rate by program over time
top_er_programs = prog_summary.head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
axes_flat = axes.flatten()

for i, prog in enumerate(top_er_programs):
    ax = axes_flat[i]
    prog_color = hewlett_colors.get(prog, '#888888')

    for er_label, ls, alpha in [('ER', '-', 1.0), ('Non-ER', '--', 0.7)]:
        subset = er_overhang_df[
            (er_overhang_df['Program'] == prog) &
            (er_overhang_df['ER'] == er_label)
        ].copy()

        # only plot years where there was actual award activity
        subset = subset[subset['Awarded'] > 0]
        if subset.empty:
            continue

        ax.plot(subset['Award Year'], subset['Overhang %'],
                marker='o', linewidth=2, markersize=4,
                linestyle=ls, alpha=alpha, color=prog_color,
                label=er_label)

    # get er grant count for this program
    er_n = 0
    if (prog, 'ER') in er_summary.index:
        er_n = int(er_summary.loc[(prog, 'ER'), 'total_grants'])
    ax.set_title(f'{prog} ({er_n} ER grants)', fontsize=10, fontweight='bold',
                 color=prog_color)
    ax.set_xticks(range(START_YEAR, END_YEAR + 1))
    ax.set_xticklabels([str(y)[2:] for y in range(START_YEAR, END_YEAR + 1)],
                       fontsize=8)
    ax.set_ylim(0, 75)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if i == 0:
        ax.legend(fontsize=8, loc='upper right')
    if i % 4 == 0:
        ax.set_ylabel('Overhang Rate (%)', fontsize=10)

fig.suptitle('Overhang Rate by Program: ER vs Non-ER Grants',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])

save_fig(fig, '11_er_vs_noner_overhang_by_program.png')


# figure 12: grouped bar — aggregate er vs non-er overhang rate by program
agg_er = (
    er_overhang_df.groupby(['Program', 'ER'])
    .agg(awarded=('Awarded', 'sum'), overhang=('Overhang', 'sum'))
)
agg_er['pct'] = agg_er['overhang'] / agg_er['awarded'] * 100

bar_data_er = agg_er['pct'].unstack('ER').reindex(prog_summary.index).dropna(how='all')

fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(bar_data_er))
bar_h = 0.35

er_vals = bar_data_er.get('ER', pd.Series(0, index=bar_data_er.index)).fillna(0)
noner_vals = bar_data_er.get('Non-ER', pd.Series(0, index=bar_data_er.index)).fillna(0)

prog_colors_er = [hewlett_colors.get(p, '#888888') for p in bar_data_er.index]

ax.barh(y_pos + bar_h / 2, er_vals, height=bar_h,
        color=prog_colors_er, edgecolor='white', linewidth=0.5, label='ER')
ax.barh(y_pos - bar_h / 2, noner_vals, height=bar_h,
        color=prog_colors_er, edgecolor='white', linewidth=0.5,
        alpha=0.45, label='Non-ER', hatch='///')

for i, prog in enumerate(bar_data_er.index):
    ev = er_vals.get(prog, 0)
    nv = noner_vals.get(prog, 0)
    if ev > 0:
        ax.text(ev + 0.5, i + bar_h / 2, f'{ev:.0f}%',
                va='center', fontsize=9, color='#333333')
    if nv > 0:
        ax.text(nv + 0.5, i - bar_h / 2, f'{nv:.0f}%',
                va='center', fontsize=9, color='#333333')

ax.set_yticks(y_pos)
ax.set_yticklabels(bar_data_er.index, fontsize=10)
ax.set_xlabel('Overhang Rate (%)', fontsize=11)
ax.set_title('Aggregate Overhang Rate by Program: ER vs Non-ER (2016-2025)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(er_vals.max(), noner_vals.max()) + 8)

save_fig(fig, '12_er_vs_noner_aggregate_by_program.png')


# payment timing accuracy — scheduled vs actual dates

print("\n" + "=" * 70)
print("PAYMENT TIMING ACCURACY (SCHEDULED VS ACTUAL)")
print("=" * 70)

# compute gap in days (positive = late, negative = early)
payments['gap_days'] = (
    payments['Payment Date'] - payments['Scheduled Date Parsed']
).dt.days

# join payments to grants to get program
pay_timing = payments.merge(
    grants_only[['ref_norm', 'Top Level Primary Program']],
    on='ref_norm',
    how='inner'
).copy()

# filter to analysis window
pay_timing = pay_timing[
    (pay_timing['Effective Year'] >= START_YEAR) &
    (pay_timing['Effective Year'] <= END_YEAR) &
    (pay_timing['gap_days'].notna())
].copy()

print(f"\npayments with both scheduled and actual dates: {len(pay_timing):,}")

# summary by program
print(f"\npayment timing gap by program (days, positive = late):")
print("-" * 100)
print(
    f"  {'Program':35s}  {'N':>6s}  {'Mean':>6s}  {'Median':>6s}"
    f"  {'P75':>6s}  {'P90':>6s}  {'% Late':>7s}  {'% >30d':>7s}"
)
print("-" * 100)

timing_by_prog = {}
for prog, grp in pay_timing.groupby('Top Level Primary Program'):
    g = grp['gap_days']
    timing_by_prog[prog] = g
    print(
        f"  {prog:35s}  {len(g):>6,}  {g.mean():>6.1f}  {g.median():>6.0f}"
        f"  {g.quantile(.75):>6.0f}  {g.quantile(.90):>6.0f}"
        f"  {(g > 0).mean() * 100:>6.1f}%  {(g > 30).mean() * 100:>6.1f}%"
    )

# summary by year
print(f"\npayment timing gap by year:")
print("-" * 60)
for yr, grp in pay_timing.groupby('Effective Year'):
    g = grp['gap_days']
    print(
        f"  {int(yr)}:  n={len(g):>5,}  mean={g.mean():>6.1f}"
        f"  median={g.median():>5.0f}  p75={g.quantile(.75):>5.0f}"
        f"  p90={g.quantile(.90):>5.0f}"
    )

# figure 13: box plot — gap distribution by program
# use the top programs from the main analysis, sorted by median gap
top_timing_programs = [p for p in prog_summary.index if p in timing_by_prog]
prog_medians = {p: timing_by_prog[p].median() for p in top_timing_programs}
top_timing_programs = sorted(top_timing_programs, key=lambda p: prog_medians[p])

fig, ax = plt.subplots(figsize=(12, 7))

box_data = [pay_timing[pay_timing['Top Level Primary Program'] == p]['gap_days']
            for p in top_timing_programs]

bp = ax.boxplot(
    box_data, vert=False, patch_artist=True, widths=0.6,
    showfliers=False,
    medianprops=dict(color='white', linewidth=2),
    whiskerprops=dict(color='#666666'),
    capprops=dict(color='#666666')
)

for i, (patch, prog) in enumerate(zip(bp['boxes'], top_timing_programs)):
    color = hewlett_colors.get(prog, '#888888')
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
    patch.set_edgecolor('white')

    med = prog_medians[prog]
    ax.text(ax.get_xlim()[1] + 2, i + 1, f'{med:.0f}d',
            va='center', fontsize=9, color='#555555')

ax.set_yticks(range(1, len(top_timing_programs) + 1))
ax.set_yticklabels(top_timing_programs, fontsize=10)
ax.axvline(x=0, color='#333333', linewidth=1, linestyle='-', alpha=0.5)
ax.set_xlabel('Days from Scheduled Date (negative = early, positive = late)', fontsize=11)
ax.set_title('Payment Timing Gap by Program (Whiskers = IQR x 1.5, Fliers Hidden)',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(-120, 200)

save_fig(fig, '13_payment_timing_gap_by_program.png')


# figure 14: small multiples — median gap trend by year per program
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
axes_flat = axes.flatten()

top_trend_programs = prog_summary.head(8).index.tolist()

for i, prog in enumerate(top_trend_programs):
    ax = axes_flat[i]
    prog_color = hewlett_colors.get(prog, '#888888')

    prog_data = pay_timing[pay_timing['Top Level Primary Program'] == prog]
    yearly_stats = prog_data.groupby('Effective Year')['gap_days'].agg(
        ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    yearly_stats.columns = ['median', 'p25', 'p75']

    years = yearly_stats.index
    ax.fill_between(years, yearly_stats['p25'], yearly_stats['p75'],
                    color=prog_color, alpha=0.2)
    ax.plot(years, yearly_stats['median'], marker='o', linewidth=2,
            markersize=4, color=prog_color)

    ax.axhline(y=0, color='#333333', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.set_title(prog, fontsize=10, fontweight='bold', color=prog_color)
    ax.set_xticks(range(START_YEAR, END_YEAR + 1))
    ax.set_xticklabels([str(y)[2:] for y in range(START_YEAR, END_YEAR + 1)],
                       fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if i % 4 == 0:
        ax.set_ylabel('Days from Scheduled', fontsize=10)

fig.suptitle('Payment Timing Gap Trend by Program (Median with IQR Band)',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])

save_fig(fig, '14_payment_timing_trend_by_program.png')


# figure 15: foundation-wide timing band chart
yearly_gap = pay_timing.groupby('Effective Year')['gap_days'].agg(
    ['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75),
     lambda x: (x > 30).mean() * 100]
)
yearly_gap.columns = ['median', 'mean', 'p25', 'p75', 'pct_over_30d']

fig, ax = plt.subplots(figsize=(12, 6))

ax.fill_between(yearly_gap.index, yearly_gap['p25'], yearly_gap['p75'],
                color='#0B3D91', alpha=0.15, label='IQR (P25-P75)')
ax.plot(yearly_gap.index, yearly_gap['median'], marker='o', linewidth=2.5,
        markersize=6, color='#0B3D91', label='Median')
ax.plot(yearly_gap.index, yearly_gap['mean'], marker='s', linewidth=1.5,
        markersize=5, color='#E07B39', linestyle='--', label='Mean')
ax.axhline(y=0, color='#333333', linewidth=0.8, linestyle='-', alpha=0.4)

ax.set_xlabel('Payment Year', fontsize=11)
ax.set_ylabel('Days from Scheduled Date', fontsize=11)
ax.set_title('Foundation-Wide Payment Timing Gap',
             fontsize=14, fontweight='bold')
ax.set_xticks(yearly_gap.index)
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

save_fig(fig, '15_foundation_wide_timing_band.png')


# figure 16: share of payments more than 30 days late
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(yearly_gap.index, yearly_gap['pct_over_30d'], width=0.7,
       color='#C0392B', edgecolor='white', linewidth=0.5, alpha=0.8)
for yr, pct in zip(yearly_gap.index, yearly_gap['pct_over_30d']):
    ax.text(yr, pct + 1, f'{pct:.0f}%', ha='center', va='bottom',
            fontsize=9, color='#333333')

ax.set_xlabel('Payment Year', fontsize=11)
ax.set_ylabel('% of Payments >30 Days Late', fontsize=11)
ax.set_title('Share of Payments More Than 30 Days Late',
             fontsize=14, fontweight='bold')
ax.set_xticks(yearly_gap.index)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(yearly_gap['pct_over_30d']) + 8)

save_fig(fig, '16_payments_over_30d_late.png')

# save timing data
pay_timing_out = pay_timing[[
    'Request', 'ref_norm', 'Top Level Primary Program', 'Effective Year',
    'Payment Amount', 'gap_days'
]].copy()
pay_timing_out.to_csv('outputs/payment_timing_gaps.csv', index=False)
print(f"\nsaved timing data to outputs/payment_timing_gaps.csv")


print("\ndone -- all outputs saved to outputs/")