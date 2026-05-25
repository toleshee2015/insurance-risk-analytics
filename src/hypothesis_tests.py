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