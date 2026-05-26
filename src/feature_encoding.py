import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEncoder:

    def __init__(self):
        self.label_encoders = {}

    # =====================================================
    # ONE-HOT ENCODING
    # =====================================================
    def one_hot_encode(self, df, columns):

        df = df.copy()

        df = pd.get_dummies(
            df,
            columns=columns,
            drop_first=True
        )

        return df

    # =====================================================
    # LABEL ENCODING
    # =====================================================
    def label_encode(self, df, columns):

        df = df.copy()

        for col in columns:

            le = LabelEncoder()

            df[col] = df[col].astype(str)

            df[col] = le.fit_transform(df[col])

            self.label_encoders[col] = le

        return df

    # =====================================================
    # AUTO ENCODING STRATEGY
    # =====================================================
    def transform(self, df):

        df = df.copy()

        # ---------------------------
        # COMMON INSURANCE COLUMNS
        # ---------------------------
        one_hot_cols = []
        label_cols = []

        # Example logic (adjust if needed)
        for col in df.select_dtypes(include="object").columns:

            unique_vals = df[col].nunique()

            # low cardinality → one-hot
            if unique_vals <= 10:
                one_hot_cols.append(col)

            # high cardinality → label encoding
            else:
                label_cols.append(col)

        df = self.one_hot_encode(df, one_hot_cols)
        df = self.label_encode(df, label_cols)

        return df