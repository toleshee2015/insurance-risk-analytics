import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway


# -----------------------------
# 1. Chi-square test (categorical vs categorical)
# -----------------------------
def chi_square_test(df, feature, target):
    table = pd.crosstab(df[feature], df[target])
    chi2, p, dof, expected = chi2_contingency(table)
    return {
        "test": "chi-square",
        "p_value": p,
        "statistic": chi2,
        "dof": dof
    }


# -----------------------------
# 2. T-test (two groups only)
# -----------------------------
def t_test(df, group_col, value_col, group_a, group_b):
    a = df[df[group_col] == group_a][value_col]
    b = df[df[group_col] == group_b][value_col]

    t_stat, p = ttest_ind(a, b, equal_var=False)

    return {
        "test": "t-test",
        "p_value": p,
        "statistic": t_stat
    }


# -----------------------------
# 3. ANOVA (multi-group numeric comparison)
# -----------------------------
def anova_test(df, group_col, value_col):
    groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
    f_stat, p = f_oneway(*groups)

    return {
        "test": "anova",
        "p_value": p,
        "statistic": f_stat
    }


# -----------------------------
# 4. Decision helper
# -----------------------------
def decision_rule(p_value, alpha=0.05):
    return "Reject H0" if p_value < alpha else "Fail to reject H0"