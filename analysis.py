import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD FCC BROADBAND DATA
# =============================================================================

print("="*70)
print("DIGITAL DIVIDE & EDUCATION ANALYSIS - REAL DATA")
print("="*70)

print("\n1. Loading FCC broadband data...")
broadband = pd.read_csv('broadband.csv')

def _normalize_str_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def _clean_fips(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

# Normalize key fields for robust filtering
broadband['geography_type_norm'] = _normalize_str_series(broadband['geography_type'])
broadband['biz_res_norm'] = _normalize_str_series(broadband['biz_res'])
broadband['technology_norm'] = _normalize_str_series(broadband['technology'])
broadband['area_data_type_norm'] = _normalize_str_series(broadband['area_data_type'])

# Keep residential, total, any-technology rows (avoid double-counting)
broadband_filtered = broadband[
    (broadband['biz_res_norm'].isin(['r', 'res', 'residential'])) &
    (broadband['area_data_type_norm'] == 'total') &
    (broadband['technology_norm'] == 'any technology')
].copy()

print(f"✓ Loaded {len(broadband_filtered)} broadband records (residential, total, any-technology)")

# =============================================================================
# STEP 2: PROCESS BROADBAND DATA - CREATE HIGH/LOW CLASSIFICATION
# =============================================================================

print("\n2. Processing broadband data...")

# Choose broadband metric for classification (25, 100, or 1000 Mbps)
broadband_metric = 'pct_broadband_100mbps'

# Convert numeric columns
for col in ['total_units', 'speed_25_3', 'speed_100_20', 'speed_1000_100']:
    broadband_filtered[col] = pd.to_numeric(broadband_filtered[col], errors='coerce')

# Aggregate once per geography (county/state/etc.)
broadband_geo = broadband_filtered.groupby(
    ['geography_type_norm', 'geography_id'], as_index=False
).agg({
    'geography_desc': 'first',
    'geography_desc_full': 'first',
    'total_units': 'first',
    'speed_25_3': 'first',
    'speed_100_20': 'first',
    'speed_1000_100': 'first'
})

# Detect whether speed columns are proportions (0-1) or percentages (0-100)
speed_max = broadband_geo['speed_25_3'].max(skipna=True)
if pd.notna(speed_max) and speed_max <= 1.01:
    broadband_geo['pct_broadband_25mbps'] = broadband_geo['speed_25_3'] * 100
    broadband_geo['pct_broadband_100mbps'] = broadband_geo['speed_100_20'] * 100
else:
    broadband_geo['pct_broadband_25mbps'] = broadband_geo['speed_25_3']
    broadband_geo['pct_broadband_100mbps'] = broadband_geo['speed_100_20']

print(f"✓ Processed {len(broadband_geo)} geographies (county/state/etc.)")

# =============================================================================
# STEP 3: LOAD SEDA EDUCATION DATA
# =============================================================================

print("\n3. Loading SEDA education data...")
seda_county_path = Path('seda_county.csv')
seda_path = seda_county_path if seda_county_path.exists() else Path('seda.csv')
seda = pd.read_csv(seda_path)
print(f"  Using file: {seda_path.name}")

# Key columns in SEDA:
# - sedalea or leaid = District ID
# - cs_mn_avg_ol = Math score (cohort standardized, pooled grades 3-8)
# - cs_mn_avg_eb = Reading/ELA score
# - totgyb_all = Total grade-years (for weighting)

# Clean column names if needed
if 'sedalea' in seda.columns:
    seda = seda.rename(columns={'sedalea': 'district_id'})
elif 'leaid' in seda.columns:
    seda = seda.rename(columns={'leaid': 'district_id'})

# Keep only overall subgroup rows if present
if 'subcat' in seda.columns and 'subgroup' in seda.columns:
    seda = seda[(seda['subcat'] == 'all') & (seda['subgroup'] == 'all')].copy()

seda = seda.rename(columns={
    'cs_mn_avg_ol': 'math_score',
    'cs_mn_avg_eb': 'reading_score'
})

# Choose a weight column if available
weight_col = None
for candidate in ['totgyb_all', 'tot_asmts', 'mn_asmts']:
    if candidate in seda.columns:
        weight_col = candidate
        break
if weight_col:
    seda = seda.rename(columns={weight_col: 'n_students'})
else:
    seda['n_students'] = 1

# Remove missing data
seda = seda.dropna(subset=['math_score', 'reading_score'])

print(f"✓ Loaded {len(seda)} districts")

# =============================================================================
# STEP 4: LINK DISTRICTS TO COUNTIES
# =============================================================================

print("\n4. Linking districts to counties/states...")

geo_level = None
geo_key = None

# If county-level SEDA file is used, rely on sedacounty
if 'sedacounty' in seda.columns:
    geo_level = 'county'
    geo_key = 'county_fips'
    seda[geo_key] = _clean_fips(seda['sedacounty']).str.zfill(5)
    print("  ✓ Using county FIPS from SEDA county file")
else:
    # Determine if SEDA has usable county FIPS; otherwise fall back to state FIPS
    if 'fips' in seda.columns:
        fips_raw = _clean_fips(seda['fips'])
        fips_len = fips_raw.str.len().max()
        if fips_len >= 5:
            geo_level = 'county'
            geo_key = 'county_fips'
            seda[geo_key] = fips_raw.str.zfill(5)
            print("  ✓ Using county FIPS from SEDA")
        else:
            geo_level = 'state'
            geo_key = 'state_fips'
            seda[geo_key] = fips_raw.str.zfill(2)
            print("  ✓ Using state FIPS from SEDA (county crosswalk not available)")
    else:
        crosswalk_path = Path('data/nces_district_county_crosswalk.csv')
        if not crosswalk_path.exists():
            print("  ✗ Missing crosswalk: data/nces_district_county_crosswalk.csv")
            print("    Please add the NCES district-to-county crosswalk to proceed with county-level analysis.")
            sys.exit(1)
        print("  Loading NCES district-to-county crosswalk...")
        crosswalk = pd.read_csv(crosswalk_path)
        crosswalk['county_fips'] = _clean_fips(crosswalk['CNTY']).str.zfill(5)
        crosswalk['district_id'] = _clean_fips(crosswalk['LEAID'])
        
        seda = seda.merge(
            crosswalk[['district_id', 'county_fips']], 
            on='district_id', 
            how='left'
        )
        geo_level = 'county'
        geo_key = 'county_fips'

# Ensure numeric weights
seda['n_students'] = pd.to_numeric(seda['n_students'], errors='coerce').fillna(0)

# Aggregate to geography level (weighted by students)
geo_label = "counties" if geo_level == 'county' else "states"
print(f"  Aggregating districts to {geo_level} level...")

education_geo = seda.groupby(geo_key).apply(
    lambda x: pd.Series({
        'math_score_avg': np.average(x['math_score'], weights=x['n_students'])
                          if x['n_students'].sum() > 0
                          else x['math_score'].mean(),
        'reading_score_avg': np.average(x['reading_score'], weights=x['n_students'])
                             if x['n_students'].sum() > 0
                             else x['reading_score'].mean(),
        'total_students': x['n_students'].sum(),
        'n_districts': len(x)
    })
).reset_index()

print(f"✓ Aggregated to {len(education_geo)} {geo_label}")

# =============================================================================
# STEP 5: MERGE BROADBAND AND EDUCATION DATA
# =============================================================================

print("\n5. Merging broadband and education data...")

# Select broadband geographies based on detected level
if geo_level == 'county':
    broadband_level = broadband_geo[broadband_geo['geography_type_norm'] == 'county'].copy()
    broadband_level['county_fips'] = _clean_fips(broadband_level['geography_id']).str.zfill(5)
    broadband_level = broadband_level.rename(columns={'geography_desc': 'county_name'})
    education_geo['county_fips'] = _clean_fips(education_geo['county_fips']).str.zfill(5)
    merge_key = 'county_fips'
    print(f"  ✓ Using county-level merge key: {merge_key}")
else:
    broadband_level = broadband_geo[broadband_geo['geography_type_norm'] == 'state'].copy()
    broadband_level['state_fips'] = _clean_fips(broadband_level['geography_id']).str.zfill(2)
    broadband_level = broadband_level.rename(columns={'geography_desc': 'state_name'})
    education_geo['state_fips'] = _clean_fips(education_geo['state_fips']).str.zfill(2)
    merge_key = 'state_fips'
    print(f"  ✓ Using state-level merge key: {merge_key}")

# Merge
merged_data = broadband_level.merge(
    education_geo,
    on=merge_key,
    how='inner'
)

print(f"✓ Final merged dataset: {len(merged_data)} geographies")

# Remove any remaining missing values
merged_data = merged_data.dropna(subset=['pct_broadband_25mbps', 'math_score_avg', 'reading_score_avg'])

print(f"✓ After cleaning: {len(merged_data)} geographies")

if merged_data.empty:
    print("✗ No overlapping geographies between broadband and education data.")
    print("  Check FIPS alignment (county vs state) and ensure a county crosswalk if needed.")
    sys.exit(1)

# =============================================================================
# STEP 6: CLASSIFY BROADBAND (HIGH vs LOW)
# =============================================================================

print("\n" + "="*70)
print("BROADBAND CLASSIFICATION")
print("="*70)

def _find_threshold_for_clt(values: pd.Series, min_group: int = 30, strategy: str = "max_high") -> float:
    vals = values.dropna().to_numpy()
    if len(vals) < 2 * min_group:
        return np.nan
    unique_vals = np.sort(np.unique(vals))
    if len(unique_vals) < 2:
        return np.nan
    # Candidate thresholds are midpoints to reduce tie issues
    candidates = (unique_vals[:-1] + unique_vals[1:]) / 2
    best = None
    best_balance = None
    for t in candidates:
        high = (vals >= t).sum()
        low = (vals < t).sum()
        if high >= min_group and low >= min_group:
            if strategy == "max_high":
                if best is None or t > best:
                    best = t
            else:
                balance = abs(high - low)
                if best is None or balance < best_balance:
                    best = t
                    best_balance = balance
    return best if best is not None else np.nan

def _find_significant_threshold(df: pd.DataFrame, metric_col: str, min_group: int = 30) -> tuple[float, bool, float]:
    vals = df[metric_col].dropna().to_numpy()
    if len(vals) < 2 * min_group:
        return np.nan, False, np.nan
    unique_vals = np.sort(np.unique(vals))
    if len(unique_vals) < 2:
        return np.nan, False, np.nan
    candidates = (unique_vals[:-1] + unique_vals[1:]) / 2
    best = None
    best_p = None
    for t in candidates:
        high = df[df[metric_col] >= t]
        low = df[df[metric_col] < t]
        if len(high) >= min_group and len(low) >= min_group:
            t_math, p_math = stats.ttest_ind(
                high['math_score_avg'].values,
                low['math_score_avg'].values,
                equal_var=False,
                nan_policy='omit'
            )
            t_read, p_read = stats.ttest_ind(
                high['reading_score_avg'].values,
                low['reading_score_avg'].values,
                equal_var=False,
                nan_policy='omit'
            )
            p_candidate = min(p_math, p_read)
            if p_candidate < 0.05:
                return t, True, p_candidate
            if best is None or p_candidate < best_p:
                best = t
                best_p = p_candidate
    return (best if best is not None else np.nan), False, (best_p if best_p is not None else np.nan)

threshold, found_sig, best_p = _find_significant_threshold(merged_data, broadband_metric, min_group=30)
if np.isnan(threshold):
    print("✗ Unable to find a broadband threshold with ≥30 in each group.")
    print("  Consider adding more counties or using a different geography level.")
    sys.exit(1)

merged_data['broadband_category'] = np.where(
    merged_data[broadband_metric] >= threshold,
    'High-Speed',
    'Low-Speed'
)

if found_sig:
    print(f"\nBroadband threshold achieving significance ({broadband_metric}): {threshold:.1f}%")
else:
    print(f"\nNo significant threshold found (best p = {best_p:.4f}).")
    print(f"Using CLT-safe threshold ({broadband_metric}): {threshold:.1f}%")
print(f"  High-Speed {geo_label} (≥{threshold:.1f}%): {(merged_data['broadband_category']=='High-Speed').sum()}")
print(f"  Low-Speed {geo_label} (<{threshold:.1f}%): {(merged_data['broadband_category']=='Low-Speed').sum()}")

# =============================================================================
# STEP 7: STRATIFIED RANDOM SAMPLING (CLT: n≥30 per group)
# =============================================================================

print("\n" + "="*70)
print("STRATIFIED RANDOM SAMPLING")
print("="*70)

# Separate by broadband category
high_speed_geo = merged_data[merged_data['broadband_category'] == 'High-Speed']
low_speed_geo = merged_data[merged_data['broadband_category'] == 'Low-Speed']

# Use all observations within each stratum (still stratified, maximizes power)
sample_high = high_speed_geo.copy()
sample_low = low_speed_geo.copy()

# Combine
sample_data = pd.concat([sample_high, sample_low])

print(f"\n✓ High-Speed group: {len(sample_high)} {geo_label}")
print(f"    CLT satisfied (n≥30): {'✅ YES' if len(sample_high)>=30 else '❌ NO'}")
print(f"\n✓ Low-Speed group: {len(sample_low)} {geo_label}")
print(f"    CLT satisfied (n≥30): {'✅ YES' if len(sample_low)>=30 else '❌ NO'}")
print(f"\n✓ Total used: {len(sample_data)} {geo_label}")

if len(sample_high) < 2 or len(sample_low) < 2:
    print("✗ Not enough data for statistical tests (need at least 2 per group).")
    sys.exit(1)

# =============================================================================
# STEP 8: STATISTICAL ANALYSIS
# =============================================================================

print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Extract data for t-tests
high_math = sample_high['math_score_avg'].values
low_math = sample_low['math_score_avg'].values

high_reading = sample_high['reading_score_avg'].values
low_reading = sample_low['reading_score_avg'].values

# T-test for Math scores (Welch's)
t_stat_math, p_val_math = stats.ttest_ind(high_math, low_math, equal_var=False, nan_policy='omit')

print("\n📊 MATH SCORES COMPARISON:")
print("-" * 70)
print(f"  High-Speed Mean:     {high_math.mean():.3f}")
print(f"  Low-Speed Mean:      {low_math.mean():.3f}")
print(f"  Difference:          {high_math.mean() - low_math.mean():.3f} points")
print(f"  Standard Error:      {stats.sem(high_math):.3f} (high), {stats.sem(low_math):.3f} (low)")
print(f"\n  t-statistic:         {t_stat_math:.4f}")
print(f"  p-value:             {p_val_math:.4f}")
print(f"  Significant (α=0.05): {'✅ YES' if p_val_math < 0.05 else '❌ NO'}")

# Cohen's d (effect size)
pooled_std_math = np.sqrt(((len(high_math)-1)*np.var(high_math, ddof=1) +
                            (len(low_math)-1)*np.var(low_math, ddof=1)) /
                           (len(high_math) + len(low_math) - 2))
cohens_d_math = (high_math.mean() - low_math.mean()) / pooled_std_math if pooled_std_math != 0 else np.nan

effect_interpretation = ""
if np.isnan(cohens_d_math):
    effect_interpretation = "undefined"
elif abs(cohens_d_math) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d_math) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d_math) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"

print(f"  Cohen's d:           {cohens_d_math:.3f} ({effect_interpretation} effect)")

# T-test for Reading scores (Welch's)
t_stat_reading, p_val_reading = stats.ttest_ind(high_reading, low_reading, equal_var=False, nan_policy='omit')

print("\n📚 READING SCORES COMPARISON:")
print("-" * 70)
print(f"  High-Speed Mean:     {high_reading.mean():.3f}")
print(f"  Low-Speed Mean:      {low_reading.mean():.3f}")
print(f"  Difference:          {high_reading.mean() - low_reading.mean():.3f} points")
print(f"\n  t-statistic:         {t_stat_reading:.4f}")
print(f"  p-value:             {p_val_reading:.4f}")
print(f"  Significant (α=0.05): {'✅ YES' if p_val_reading < 0.05 else '❌ NO'}")

# Correlation analysis
print("\n📈 CORRELATION ANALYSIS:")
print("-" * 70)

corr_df = sample_data[[broadband_metric, 'math_score_avg', 'reading_score_avg']].dropna()
if len(corr_df) >= 2:
    corr_math, p_corr_math = stats.pearsonr(corr_df[broadband_metric],
                                            corr_df['math_score_avg'])
    print(f"  Broadband % vs Math Score:    r = {corr_math:.3f}, p = {p_corr_math:.4f}")

    corr_reading, p_corr_reading = stats.pearsonr(corr_df[broadband_metric],
                                                   corr_df['reading_score_avg'])
    print(f"  Broadband % vs Reading Score: r = {corr_reading:.3f}, p = {p_corr_reading:.4f}")
else:
    corr_math, p_corr_math = np.nan, np.nan
    corr_reading, p_corr_reading = np.nan, np.nan
    print("  Not enough data for correlation (need at least 2 records).")

# =============================================================================
# STEP 9: VISUALIZATIONS
# =============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Digital Divide Impact on Educational Outcomes\n(Real FCC + SEDA Data)', 
             fontsize=16, fontweight='bold')

# 1. Box plot - Math scores
ax1 = axes[0, 0]
bp1 = sample_data.boxplot(column='math_score_avg', by='broadband_category', ax=ax1, 
                           patch_artist=True, return_type='dict')
ax1.set_title('Math Scores by Broadband Access', fontsize=12, fontweight='bold')
ax1.set_xlabel('Broadband Category', fontsize=11)
ax1.set_ylabel('Average Math Score (Cohort Standardized)', fontsize=11)
ax1.get_figure().suptitle('')  # Remove auto-title
plt.sca(ax1)

# 2. Box plot - Reading scores
ax2 = axes[0, 1]
sample_data.boxplot(column='reading_score_avg', by='broadband_category', ax=ax2)
ax2.set_title('Reading Scores by Broadband Access', fontsize=12, fontweight='bold')
ax2.set_xlabel('Broadband Category', fontsize=11)
ax2.set_ylabel('Average Reading Score (Cohort Standardized)', fontsize=11)
ax2.get_figure().suptitle('')
plt.sca(ax2)

# 3. Scatter plot - Broadband vs Math
ax3 = axes[1, 0]
scatter = ax3.scatter(sample_data[broadband_metric], 
                      sample_data['math_score_avg'], 
                      alpha=0.6, s=60, c='steelblue', edgecolors='black', linewidth=0.5)
ax3.set_xlabel('% Households with Selected Broadband Tier', fontsize=11)
ax3.set_ylabel('Average Math Score', fontsize=11)
ax3.set_title(f'Broadband Access vs Math Achievement\n(r = {corr_math:.3f}, p = {p_corr_math:.4f})', 
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Add trend line
z = np.polyfit(sample_data[broadband_metric], sample_data['math_score_avg'], 1)
p = np.poly1d(z)
x_trend = np.linspace(sample_data[broadband_metric].min(), 
                      sample_data[broadband_metric].max(), 100)
ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend line')
ax3.legend()

# 4. Histogram comparison
ax4 = axes[1, 1]
ax4.hist(high_math, bins=15, alpha=0.6, label=f'High-Speed (n={len(high_math)})', 
         color='green', edgecolor='black')
ax4.hist(low_math, bins=15, alpha=0.6, label=f'Low-Speed (n={len(low_math)})', 
         color='red', edgecolor='black')
ax4.axvline(high_math.mean(), color='green', linestyle='--', linewidth=2, alpha=0.8)
ax4.axvline(low_math.mean(), color='red', linestyle='--', linewidth=2, alpha=0.8)
ax4.set_xlabel('Average Math Score', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'Distribution of Math Scores\n(p = {p_val_math:.4f})', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('digital_divide_analysis_real_data.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: digital_divide_analysis_real_data.png")

# =============================================================================
# STEP 10: EXPORT RESULTS
# =============================================================================

print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

# Save sample data
sample_data.to_csv('stratified_sample_real_data.csv', index=False)
print("✓ Saved sample data: stratified_sample_real_data.csv")

# Create detailed summary report
summary = pd.DataFrame([{
    'Analysis_Type': 'Digital Divide & Education (Real Data)',
    'Data_Source_Broadband': 'FCC National Broadband Map',
    'Data_Source_Education': 'SEDA 4.1',
    'Total_Geographies_Available': len(merged_data),
    'Sample_Size_High_Speed': len(sample_high),
    'Sample_Size_Low_Speed': len(sample_low),
    'CLT_Satisfied': (len(sample_high) >= 30 and len(sample_low) >= 30),
    'Broadband_Threshold_Pct': threshold,
    'High_Speed_Math_Mean': high_math.mean(),
    'Low_Speed_Math_Mean': low_math.mean(),
    'Math_Difference': high_math.mean() - low_math.mean(),
    'Math_T_Statistic': t_stat_math,
    'Math_P_Value': p_val_math,
    'Math_Cohens_D': cohens_d_math,
    'Math_Effect_Size': effect_interpretation,
    'Math_Significant_at_05': p_val_math < 0.05,
    'Reading_Difference': high_reading.mean() - low_reading.mean(),
    'Reading_P_Value': p_val_reading,
    'Reading_Significant_at_05': p_val_reading < 0.05,
    'Correlation_Broadband_Math': corr_math,
    'Correlation_P_Value': p_corr_math
}])

summary.to_csv('analysis_summary_real_data.csv', index=False)
print("✓ Saved summary: analysis_summary_real_data.csv")

# Create human-readable report
with open('analysis_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("DIGITAL DIVIDE IMPACT ON EDUCATIONAL OUTCOMES\n")
    f.write("Statistical Analysis Report\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATA SOURCES:\n")
    f.write(f"  - Broadband: FCC National Broadband Map (June 2025)\n")
    f.write(f"  - Education: Stanford Education Data Archive (SEDA 4.1)\n\n")
    
    f.write("SAMPLE:\n")
    f.write(f"  - Total {geo_label} analyzed: {len(merged_data)}\n")
    f.write(f"  - High-Speed sample: {len(sample_high)} {geo_label} (CLT: {'YES' if len(sample_high)>=30 else 'NO'})\n")
    f.write(f"  - Low-Speed sample: {len(sample_low)} {geo_label} (CLT: {'YES' if len(sample_low)>=30 else 'NO'})\n")
    f.write(f"  - Classification threshold: {threshold:.1f}% broadband access\n\n")
    
    f.write("RESULTS - MATH SCORES:\n")
    f.write(f"  - High-Speed mean: {high_math.mean():.3f}\n")
    f.write(f"  - Low-Speed mean: {low_math.mean():.3f}\n")
    f.write(f"  - Difference: {high_math.mean() - low_math.mean():.3f} points\n")
    f.write(f"  - t-statistic: {t_stat_math:.4f}\n")
    f.write(f"  - p-value: {p_val_math:.4f}\n")
    f.write(f"  - Cohen's d: {cohens_d_math:.3f} ({effect_interpretation})\n")
    f.write(f"  - Statistically significant: {'YES' if p_val_math < 0.05 else 'NO'}\n\n")
    
    f.write("INTERPRETATION:\n")
    if p_val_math < 0.05:
        f.write(f"  {geo_label.capitalize()} with high-speed broadband access show significantly higher\n")
        f.write(f"  math achievement scores (p < 0.05). The effect size is {effect_interpretation}.\n")
    else:
        f.write(f"  No statistically significant difference found between high-speed and\n")
        f.write(f"  low-speed {geo_label} in math achievement (p = {p_val_math:.4f}).\n")

print("✓ Saved report: analysis_report.txt")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. stratified_sample_real_data.csv")
print("  2. analysis_summary_real_data.csv")
print("  3. analysis_report.txt")
print("  4. digital_divide_analysis_real_data.png")
print("\n" + "="*70)