import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        pass

    # =====================================================
    # VEHICLE AGE
    # =====================================================
    def add_vehicle_age(self, df, current_year=None):

        df = df.copy()

        if current_year is None:
            current_year = pd.Timestamp.today().year

        # adjust column name if needed (common variants exist)
        if "VehicleYear" in df.columns:
            df["vehicle_age"] = current_year - df["VehicleYear"]

        elif "RegistrationYear" in df.columns:
            df["vehicle_age"] = current_year - df["RegistrationYear"]

        else:
            df["vehicle_age"] = np.nan

        return df

    # =====================================================
    # POLICY DURATION
    # =====================================================
    def add_policy_duration(self, df):

        df = df.copy()

        # common insurance datasets use inception / start date
        date_cols = ["PolicyStartDate", "InceptionDate", "StartDate"]

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

                df["policy_duration_days"] = (
                    pd.Timestamp.today() - df[col]
                ).dt.days

                break

        return df

    # =====================================================
    # CLAIM INDICATOR
    # =====================================================
    def add_claim_indicator(self, df):

        df = df.copy()

        if "TotalClaims" in df.columns:
            df["has_claim"] = (df["TotalClaims"] > 0).astype(int)

        return df

    # =====================================================
    # LOSS RATIO (VERY IMPORTANT FOR INSURANCE)
    # =====================================================
    def add_loss_ratio(self, df):

        df = df.copy()

        if (
            "TotalClaims" in df.columns
            and "TotalPremium" in df.columns
        ):
            df["loss_ratio"] = (
                df["TotalClaims"] /
                df["TotalPremium"].replace(0, np.nan)
            )

        return df

    # =====================================================
    # PREMIUM PER YEAR OF VEHICLE AGE
    # =====================================================
    def add_premium_per_vehicle_age(self, df):

        df = df.copy()

        if "vehicle_age" in df.columns:

            df["premium_per_vehicle_age"] = (
                df["TotalPremium"] /
                df["vehicle_age"].replace(0, np.nan)
            )

        return df

    # =====================================================
    # FULL PIPELINE
    # =====================================================
    def transform(self, df):

        df = df.copy()

        df = self.add_vehicle_age(df)
        df = self.add_policy_duration(df)
        df = self.add_claim_indicator(df)
        df = self.add_loss_ratio(df)
        df = self.add_premium_per_vehicle_age(df)

        return df