from hypothesis_registry import HYPOTHESIS_REGISTRY
from hypothesis_engine import HypothesisEngine


class HypothesisRunner:
    def __init__(self, df):
        self.df = df
        self.engine = HypothesisEngine()

    def run_all(self):
        results = []

        for name, config in HYPOTHESIS_REGISTRY.items():
            group_col = config["group_col"]

            # --------------------------
            # Multi-KPI hypotheses
            # --------------------------
            if "kpis" in config:
                for kpi in config["kpis"]:

                    if kpi == "claim_frequency":

                        contingency = self._to_contingency(group_col)

                        res = self.engine.chi_square(
                            contingency,
                            label=f"{name} - frequency"
                        )

                    elif kpi == "claim_severity":

                        groups = self.df[group_col].dropna().unique()
                        group_a, group_b = groups[0], groups[1]

                        res = self.engine.welch_ttest(
                            self.df,
                            group_col=group_col,
                            value_col="claim_amount",
                            group_a=group_a,
                            group_b=group_b,
                            label=f"{name} - severity"
                        )

                    else:
                        continue

                    results.append(res)

            # --------------------------
            # Single KPI (margin case)
            # --------------------------
            else:
                if config["kpi"] == "margin":

                    self.df["margin"] = (
                        self.df["premium"] - self.df["claim_amount"]
                    )

                    groups = self.df[group_col].dropna().unique()
                    group_a, group_b = groups[0], groups[1]

                    res = self.engine.welch_ttest(
                        self.df,
                        group_col=group_col,
                        value_col="margin",
                        group_a=group_a,
                        group_b=group_b,
                        label=f"{name} - margin"
                    )

                    results.append(res)

        return results

    def _to_contingency(self, group_col):
        return (
            self.df.groupby(group_col)["has_claim"]
            .value_counts()
            .unstack(fill_value=0)
        )