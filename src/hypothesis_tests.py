import pandas as pd
import numpy as np
from scipy import stats

def test_gender_risk(df):
    """Evaluates whether Underwriting Margin differs between Male and Female drivers."""
    local_df = df.copy()
    local_df['Margin'] = local_df['TotalPremium'] - local_df['TotalClaims']
    
    group_a = local_df[local_df['Gender'].str.upper() == 'FEMALE']['Margin'].dropna()
    group_b = local_df[local_df['Gender'].str.upper() == 'MALE']['Margin'].dropna()
    
    if len(group_a) == 0 or len(group_b) == 0:
        return "Gender Data Incomplete", np.nan, "N/A"
        
    t_stat, p_val = stats.ttest_ind(group_b, group_a, equal_var=False)
    decision = "Reject H₀" if p_val < 0.05 else "Fail to Reject H₀"
    return "Two-Sample Welch t-test", p_val, decision

def test_geographic_variance(df):
    """Evaluates whether Claim Frequency depends on localized Province boundaries."""
    local_df = df.copy()
    local_df['HasClaim'] = np.where(local_df['TotalClaims'] > 0, 1, 0)
    
    geo_df = local_df[local_df['Province'].str.upper().isin(['NC', 'GP'])]
    
    if len(geo_df) == 0:
        return "Geographic Data Incomplete", np.nan, "N/A"
        
    contingency_table = pd.crosstab(geo_df['Province'], geo_df['HasClaim'])
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    decision = "Reject H₀" if p_val < 0.05 else "Fail to Reject H₀"
    return "Chi-Square (χ²) Test", p_val, decision

def test_security_impact(df):
    """Evaluates if Tracking Device presence changes financial Claim Severity."""
    claims_subset = df[df['TotalClaims'] > 0].copy()
    
    group_a = claims_subset[claims_subset['TrackingDevice'] == 0]['TotalClaims'].dropna()
    group_b = claims_subset[claims_subset['TrackingDevice'] == 1]['TotalClaims'].dropna()
    
    if len(group_a) == 0 or len(group_b) == 0:
        return "Security Data Incomplete", np.nan, "N/A"
        
    t_stat, p_val = stats.ttest_ind(group_b, group_a, equal_var=False)
    decision = "Reject H₀" if p_val < 0.05 else "Fail to Reject H₀"
    return "Two-Sample Welch t-test", p_val, decision