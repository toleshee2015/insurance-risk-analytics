import pandas as pd
from scipy import stats
import numpy as np


# -----------------------------
# CHI-SQUARE TEST
# -----------------------------
def chi_square_test(df, group_col, target_col):
    table = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, exp = stats.chi2_contingency(table)

    return {
        "test": "Chi-square",
        "statistic": chi2,
        "p_value": p,
        "decision": "Reject H0" if p < 0.05 else "Fail to Reject H0"
    }


# -----------------------------
# WELCH T-TEST
# -----------------------------
def welch_ttest(df, group_col, value_col, group_a, group_b):
    a = df[df[group_col] == group_a][value_col].dropna()
    b = df[df[group_col] == group_b][value_col].dropna()

    t_stat, p = stats.ttest_ind(a, b, equal_var=False)

    return {
        "test": "Welch t-test",
        "statistic": t_stat,
        "p_value": p,
        "decision": "Reject H0" if p < 0.05 else "Fail to Reject H0"
    }


# -----------------------------
# MARGIN CALCULATION
# -----------------------------
def add_margin(df):
    df = df.copy()
    df["margin"] = df["premium"] - df["claim_amount"]
    return df