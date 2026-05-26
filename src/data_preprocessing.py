import pandas as pd
import numpy as np


class DataPreprocessor:

    def __init__(self, missing_threshold=0.4):
        """
        missing_threshold:
        Drop columns with missing ratio above this threshold.
        """
        self.missing_threshold = missing_threshold

    # =====================================================
    # MISSING VALUE SUMMARY
    # =====================================================
    def missing_summary(self, df):

        summary = pd.DataFrame({
            "missing_count": df.isnull().sum(),
            "missing_ratio": df.isnull().mean()
        })

        summary = summary.sort_values(
            by="missing_ratio",
            ascending=False
        )

        return summary

    # =====================================================
    # DROP HIGH-MISSING COLUMNS
    # =====================================================
    def drop_high_missing_columns(self, df):

        missing_ratio = df.isnull().mean()

        cols_to_drop = missing_ratio[
            missing_ratio > self.missing_threshold
        ].index.tolist()

        df = df.drop(columns=cols_to_drop)

        return df, cols_to_drop

    # =====================================================
    # NUMERICAL IMPUTATION
    # =====================================================
    def impute_numeric(self, df):

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_cols:

            if df[col].isnull().sum() > 0:

                median_value = df[col].median()

                df[col] = df[col].fillna(
                    median_value
                )

        return df

    # =====================================================
    # CATEGORICAL IMPUTATION
    # =====================================================
    def impute_categorical(self, df):

        categorical_cols = df.select_dtypes(
            include="object"
        ).columns

        for col in categorical_cols:

            if df[col].isnull().sum() > 0:

                mode_value = df[col].mode()[0]

                df[col] = df[col].fillna(
                    mode_value
                )

        return df

    # =====================================================
    # FULL PIPELINE
    # =====================================================
    def fit_transform(self, df):

        print("Initial shape:", df.shape)

        # Drop heavily missing columns
        df, dropped_cols = self.drop_high_missing_columns(df)

        print("Dropped columns:", dropped_cols)

        # Impute numeric columns
        df = self.impute_numeric(df)

        # Impute categorical columns
        df = self.impute_categorical(df)

        print("Final shape:", df.shape)

        return df