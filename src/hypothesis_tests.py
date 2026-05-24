<<<<<<< HEAD
import numpy as np
import pandas as pd
from scipy import stats


# =========================================================
# CHI-SQUARE TEST
# Used for categorical KPIs
# Example:
# Province vs Claim Frequency
# =========================================================
def chi_square_test(df, group_col, target_col):

    contingency_table = pd.crosstab(
        df[group_col],
        df[target_col]
    )

    chi2, p_value, dof, expected = stats.chi2_contingency(
        contingency_table
    )

    return {
        "test": "Chi-square",
        "group_column": group_col,
        "target_column": target_col,
        "chi2_statistic": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "decision": (
            "Reject H0"
            if p_value < 0.05
            else "Fail to Reject H0"
        )
    }


# =========================================================
# WELCH T-TEST
# Used for numerical KPIs
# Example:
# Gender vs Margin
# =========================================================
def welch_ttest(
    df,
    group_col,
    value_col,
    group_a,
    group_b
):

    a = df[
        df[group_col] == group_a
    ][value_col].dropna()

    b = df[
        df[group_col] == group_b
    ][value_col].dropna()

    t_stat, p_value = stats.ttest_ind(
        a,
        b,
        equal_var=False
    )

    return {
        "test": "Welch t-test",
        "group_column": group_col,
        "value_column": value_col,
        "group_a": group_a,
        "group_b": group_b,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "decision": (
            "Reject H0"
            if p_value < 0.05
            else "Fail to Reject H0"
        )
    }


# =========================================================
# ADD MARGIN KPI
# Margin = Premium - Claims
# =========================================================
def add_margin(df):

    df = df.copy()

    df["margin"] = (
        df["TotalPremium"]
        - df["TotalClaims"]
    )

    return df


# =========================================================
# ADD CLAIM FREQUENCY KPI
# 1 = claim occurred
# 0 = no claim
# =========================================================
def add_claim_frequency(df):

    df = df.copy()

    df["has_claim"] = (
        df["TotalClaims"] > 0
    ).astype(int)

    return df


# =========================================================
# SUMMARY TABLE GENERATOR
# =========================================================
def results_table(results):

    return pd.DataFrame(results)
=======
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
>>>>>>> 2755f2b (Complete task 2 analysis)
