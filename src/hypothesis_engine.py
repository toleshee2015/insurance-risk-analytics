import numpy as np
import pandas as pd
from scipy import stats


class HypothesisEngine:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = []

    # =========================================================
    # DECISION RULE (EMBEDDED)
    # =========================================================
    def interpret(self, p_value):
        return "Reject H0" if p_value < self.alpha else "Fail to Reject H0"

    # =========================================================
    # EFFECT SIZE
    # =========================================================
    def cohens_d(self, a, b):
        a, b = np.array(a), np.array(b)

        na, nb = len(a), len(b)
        sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)

        pooled = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))

        if pooled == 0:
            return 0

        return (np.mean(a) - np.mean(b)) / pooled

    # =========================================================
    # ASSUMPTION CHECKS
    # =========================================================
    def is_normal(self, data):
        if len(data) < 8:
            return False
        _, p = stats.shapiro(data)
        return p > self.alpha

    def has_equal_variance(self, a, b):
        _, p = stats.levene(a, b)
        return p > self.alpha

    # =========================================================
    # MATCHED GROUP SELECTION
    # =========================================================
    def find_matched_groups(self, df, group_col, control_features):
        summary = df.groupby(group_col)[control_features].mean()

        min_pair = None
        min_dist = float("inf")

        for i in summary.index:
            for j in summary.index:
                if i == j:
                    continue

                dist = np.linalg.norm(summary.loc[i] - summary.loc[j])

                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)

        return min_pair

    # =========================================================
    # CHI-SQUARE (categorical KPI)
    # =========================================================
    def chi_square(self, contingency_table):
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        result = {
            "test": "Chi-square",
            "stat": float(chi2),
            "p_value": float(p),
            "decision": self.interpret(p)
        }

        self.results.append(result)
        return result

    # =========================================================
    # ADAPTIVE NUMERICAL TEST
    # =========================================================
    def adaptive_ttest(self, a, b):

        a = pd.Series(a).dropna()
        b = pd.Series(b).dropna()

        normal_a = self.is_normal(a)
        normal_b = self.is_normal(b)
        equal_var = self.has_equal_variance(a, b)

        if normal_a and normal_b:
            if equal_var:
                stat, p = stats.ttest_ind(a, b, equal_var=True)
                test_name = "Student t-test"
            else:
                stat, p = stats.ttest_ind(a, b, equal_var=False)
                test_name = "Welch t-test"
        else:
            stat, p = stats.mannwhitneyu(a, b)
            test_name = "Mann-Whitney U"

        return {
            "test": test_name,
            "stat": float(stat),
            "p_value": float(p),
            "decision": self.interpret(p)
        }

    # =========================================================
    # MATCHED T-TEST
    # =========================================================
    def matched_ttest(self, df, group_col, value_col, control_features):

        g1, g2 = self.find_matched_groups(df, group_col, control_features)

        a = df[df[group_col] == g1][value_col].dropna()
        b = df[df[group_col] == g2][value_col].dropna()

        t_stat, p = stats.ttest_ind(a, b, equal_var=False)

        return {
            "test": "Matched Welch t-test",
            "group_a": g1,
            "group_b": g2,
            "t_stat": float(t_stat),
            "p_value": float(p),
            "effect_size": float(self.cohens_d(a, b)),
            "decision": self.interpret(p)
        }

    # =========================================================
    # MAIN DISPATCHER
    # =========================================================
    def run_test(self, df, group_col, value_col, kpi_type,
                 baseline=None, control_features=None):

        result = {
            "group_col": group_col,
            "value_col": value_col,
            "kpi_type": kpi_type
        }

        # -------------------------
        # CATEGORICAL KPI
        # -------------------------
        if kpi_type == "categorical":
            table = pd.crosstab(df[group_col], df[value_col])
            res = self.chi_square(table)
            result.update(res)

        # -------------------------
        # NUMERICAL KPI
        # -------------------------
        elif kpi_type == "numerical":

            if control_features is not None:
                res = self.matched_ttest(
                    df, group_col, value_col, control_features)
            else:
                if baseline is None:
                    raise ValueError("baseline required")

                a = df[df[group_col] == baseline][value_col]
                b = df[df[group_col] != baseline][value_col]

                res = self.adaptive_ttest(a, b)

            result.update(res)

        else:
            raise ValueError("Unknown KPI type")

        self.results.append(result)
        return result

    # =========================================================
    # SUMMARY
    # =========================================================
    def summary(self):
        return pd.DataFrame(self.results)
