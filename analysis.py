import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD FCC BROADBAND DATA
# =============================================================================

print("="*70)
print("DIGITAL DIVIDE & EDUCATION ANALYSIS - REAL DATA")
print("="*70)

print("\n1. Loading FCC broadband data...")
broadband = pd.read_csv('data/fcc_broadband_geography.csv')

# Filter for COUNTY level only
broadband_county = broadband[
    (broadband['geography_type'] == 'county') & 
    (broadband['biz_res'] == 'res')  # Residential only
].copy()

print(f"✓ Loaded {len(broadband_county)} county records")

# =============================================================================
# STEP 2: PROCESS BROADBAND DATA - CREATE HIGH/LOW CLASSIFICATION
# =============================================================================

print("\n2. Processing broadband data...")

# Extract county-level broadband metrics
# Use speed_25_3 (≥25 Mbps download / ≥3 Mbps upload - FCC broadband definition)
county_broadband = broadband_county.groupby('geography_id').agg({
    'geography_desc': 'first',  # County name
    'total_units': 'sum',       # Total housing units
    'speed_25_3': 'sum',        # Units with ≥25/3 Mbps
    'speed_100_20': 'sum',      # Units with ≥100/20 Mbps
    'speed_1000_100': 'sum'     # Units with gigabit
}).reset_index()

# Calculate percentages
county_broadband['pct_broadband_25mbps'] = (
    county_broadband['speed_25_3'] / county_broadband['total_units'] * 100
)
county_broadband['pct_broadband_100mbps'] = (
    county_broadband['speed_100_20'] / county_broadband['total_units'] * 100
)

# Rename for clarity
county_broadband = county_broadband.rename(columns={
    'geography_id': 'county_fips',
    'geography_desc': 'county_name'
})

print(f"✓ Processed {len(county_broadband)} counties")
print(f"  Mean broadband access (≥25 Mbps): {county_broadband['pct_broadband_25mbps'].mean():.1f}%")

# =============================================================================
# STEP 3: LOAD SEDA EDUCATION DATA
# =============================================================================

print("\n3. Loading SEDA education data...")
seda = pd.read_csv('data/seda_geodist_pool_cs_4.1.csv')

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

seda = seda.rename(columns={
    'cs_mn_avg_ol': 'math_score',
    'cs_mn_avg_eb': 'reading_score',
    'totgyb_all': 'n_students'
})

# Remove missing data
seda = seda.dropna(subset=['math_score', 'reading_score'])

print(f"✓ Loaded {len(seda)} districts")

# =============================================================================
# STEP 4: LINK DISTRICTS TO COUNTIES
# =============================================================================

print("\n4. Linking districts to counties...")

# Check if SEDA already has county FIPS
if 'fips' in seda.columns:
    print("  ✓ SEDA already contains county FIPS codes")
    seda['county_fips'] = seda['fips'].astype(str).str.zfill(5)
else:
    # Need to load NCES crosswalk
    print("  Loading NCES district-to-county crosswalk...")
    crosswalk = pd.read_csv('data/nces_district_county_crosswalk.csv')
    crosswalk['county_fips'] = crosswalk['CNTY'].astype(str).str.zfill(5)
    crosswalk['district_id'] = crosswalk['LEAID'].astype(str)
    
    seda = seda.merge(
        crosswalk[['district_id', 'county_fips']], 
        on='district_id', 
        how='left'
    )

# Aggregate to county level (weighted by students)
print("  Aggregating districts to county level...")

county_education = seda.groupby('county_fips').apply(
    lambda x: pd.Series({
        'math_score_avg': np.average(x['math_score'], weights=x['n_students']) 
                          if 'n_students' in x.columns and x['n_students'].sum() > 0
                          else x['math_score'].mean(),
        'reading_score_avg': np.average(x['reading_score'], weights=x['n_students'])
                             if 'n_students' in x.columns and x['n_students'].sum() > 0
                             else x['reading_score'].mean(),
        'total_students': x['n_students'].sum() if 'n_students' in x.columns else len(x),
        'n_districts': len(x)
    })
).reset_index()

print(f"✓ Aggregated to {len(county_education)} counties")

# =============================================================================
# STEP 5: MERGE BROADBAND AND EDUCATION DATA
# =============================================================================

print("\n5. Merging broadband and education data...")

# Ensure both use 5-digit FIPS codes
county_broadband['county_fips'] = county_broadband['county_fips'].astype(str).str.zfill(5)
county_education['county_fips'] = county_education['county_fips'].astype(str).str.zfill(5)

# Merge
merged_data = county_broadband.merge(
    county_education, 
    on='county_fips', 
    how='inner'
)

print(f"✓ Final merged dataset: {len(merged_data)} counties")

# Remove any remaining missing values
merged_data = merged_data.dropna(subset=['pct_broadband_25mbps', 'math_score_avg', 'reading_score_avg'])

print(f"✓ After cleaning: {len(merged_data)} counties")

# =============================================================================
# STEP 6: CLASSIFY BROADBAND (HIGH vs LOW)
# =============================================================================

print("\n" + "="*70)
print("BROADBAND CLASSIFICATION")
print("="*70)

# Use median split
median_broadband = merged_data['pct_broadband_25mbps'].median()

merged_data['broadband_category'] = np.where(
    merged_data['pct_broadband_25mbps'] >= median_broadband,
    'High-Speed',
    'Low-Speed'
)

print(f"\nMedian broadband access: {median_broadband:.1f}%")
print(f"  High-Speed counties (≥{median_broadband:.1f}%): {(merged_data['broadband_category']=='High-Speed').sum()}")
print(f"  Low-Speed counties (<{median_broadband:.1f}%): {(merged_data['broadband_category']=='Low-Speed').sum()}")

# =============================================================================
# STEP 7: STRATIFIED RANDOM SAMPLING (CLT: n≥30 per group)
# =============================================================================

print("\n" + "="*70)
print("STRATIFIED RANDOM SAMPLING")
print("="*70)

# Separate by broadband category
high_speed_counties = merged_data[merged_data['broadband_category'] == 'High-Speed']
low_speed_counties = merged_data[merged_data['broadband_category'] == 'Low-Speed']

# Sample 50 from each (ensures CLT: n≥30)
np.random.seed(42)

n_sample = 50
sample_high = high_speed_counties.sample(n=min(n_sample, len(high_speed_counties)), random_state=42)
sample_low = low_speed_counties.sample(n=min(n_sample, len(low_speed_counties)), random_state=42)

# Combine
sample_data = pd.concat([sample_high, sample_low])

print(f"\n✓ High-Speed sample: {len(sample_high)} counties")
print(f"    CLT satisfied (n≥30): {'✅ YES' if len(sample_high)>=30 else '❌ NO'}")
print(f"\n✓ Low-Speed sample: {len(sample_low)} counties")
print(f"    CLT satisfied (n≥30): {'✅ YES' if len(sample_low)>=30 else '❌ NO'}")
print(f"\n✓ Total sample: {len(sample_data)} counties")

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

# T-test for Math scores
t_stat_math, p_val_math = stats.ttest_ind(high_math, low_math)

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
cohens_d_math = (high_math.mean() - low_math.mean()) / pooled_std_math

effect_interpretation = ""
if abs(cohens_d_math) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d_math) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d_math) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"

print(f"  Cohen's d:           {cohens_d_math:.3f} ({effect_interpretation} effect)")

# T-test for Reading scores
t_stat_reading, p_val_reading = stats.ttest_ind(high_reading, low_reading)

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
corr_math, p_corr_math = stats.pearsonr(sample_data['pct_broadband_25mbps'], 
                                        sample_data['math_score_avg'])
print(f"  Broadband % vs Math Score:    r = {corr_math:.3f}, p = {p_corr_math:.4f}")

corr_reading, p_corr_reading = stats.pearsonr(sample_data['pct_broadband_25mbps'],
                                               sample_data['reading_score_avg'])
print(f"  Broadband % vs Reading Score: r = {corr_reading:.3f}, p = {p_corr_reading:.4f}")

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
scatter = ax3.scatter(sample_data['pct_broadband_25mbps'], 
                      sample_data['math_score_avg'], 
                      alpha=0.6, s=60, c='steelblue', edgecolors='black', linewidth=0.5)
ax3.set_xlabel('% Households with ≥25 Mbps Broadband', fontsize=11)
ax3.set_ylabel('Average Math Score', fontsize=11)
ax3.set_title(f'Broadband Access vs Math Achievement\n(r = {corr_math:.3f}, p = {p_corr_math:.4f})', 
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Add trend line
z = np.polyfit(sample_data['pct_broadband_25mbps'], sample_data['math_score_avg'], 1)
p = np.poly1d(z)
x_trend = np.linspace(sample_data['pct_broadband_25mbps'].min(), 
                      sample_data['pct_broadband_25mbps'].max(), 100)
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
    'Total_Counties_Available': len(merged_data),
    'Sample_Size_High_Speed': len(sample_high),
    'Sample_Size_Low_Speed': len(sample_low),
    'CLT_Satisfied': (len(sample_high) >= 30 and len(sample_low) >= 30),
    'Median_Broadband_Threshold_Pct': median_broadband,
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
    f.write(f"  - Total counties analyzed: {len(merged_data)}\n")
    f.write(f"  - High-Speed sample: {len(sample_high)} counties (CLT: {'✓' if len(sample_high)>=30 else '✗'})\n")
    f.write(f"  - Low-Speed sample: {len(sample_low)} counties (CLT: {'✓' if len(sample_low)>=30 else '✗'})\n")
    f.write(f"  - Classification threshold: {median_broadband:.1f}% broadband access\n\n")
    
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
        f.write(f"  Counties with high-speed broadband access show significantly higher\n")
        f.write(f"  math achievement scores (p < 0.05). The effect size is {effect_interpretation}.\n")
    else:
        f.write(f"  No statistically significant difference found between high-speed and\n")
        f.write(f"  low-speed counties in math achievement (p = {p_val_math:.4f}).\n")

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