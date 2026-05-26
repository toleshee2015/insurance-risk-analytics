import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class ModelTrainer:

    def __init__(self, test_size=0.2, random_state=42, run_shap=False):

        self.test_size = test_size
        self.random_state = random_state
        self.run_shap = run_shap

        self.models = {}
        self.results = []

        self.best_model_name = None
        self.best_model = None
        self.shap_values = None

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================
    def feature_engineering(self, df):

        df = df.copy()
        current_year = pd.Timestamp.now().year

        if "RegistrationYear" in df.columns:
            df["VehicleAge"] = (current_year - df["RegistrationYear"]).clip(lower=0)

        if "TotalPremium" in df.columns and "SumInsured" in df.columns:
            df["PremiumToInsuredRatio"] = df["TotalPremium"] / (df["SumInsured"] + 1)

        if "TotalClaims" in df.columns and "TotalPremium" in df.columns:
            df["ClaimToPremiumRatio"] = df["TotalClaims"] / (df["TotalPremium"] + 1)

        if "NumberOfVehiclesInFleet" in df.columns:
            df["IsFleet"] = (df["NumberOfVehiclesInFleet"] > 1).astype(int)

        return df

    # =====================================================
    # CLEANING (FIXED + WARNING FREE)
    # =====================================================
    def clean_features(self, df):

        df = df.copy()

        df = df.dropna(axis=1, how="all")

        # -------------------------
        # DATE HANDLING (FIXED)
        # -------------------------
        date_cols = ["TransactionMonth", "VehicleIntroDate"]

        for col in date_cols:
            if col in df.columns:

                df[col] = pd.to_datetime(
                    df[col],
                    format="%Y-%m-%d %H:%M:%S",
                    errors="coerce"
                )

                df[col + "_year"] = df[col].dt.year
                df[col + "_month"] = df[col].dt.month

                df = df.drop(columns=[col])

        # -------------------------
        # BOOLEAN → INT
        # -------------------------
        bool_cols = df.select_dtypes(include=["bool"]).columns
        df[bool_cols] = df[bool_cols].astype(int)

        # -------------------------
        # CATEGORICAL → ONE HOT (FIXED FOR PANDAS 2/3)
        # -------------------------
        cat_cols = df.select_dtypes(include=["object", "string"]).columns

        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # -------------------------
        # NUMERIC CLEANUP
        # -------------------------
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        return df

    # =====================================================
    # SPLIT
    # =====================================================
    def split_data(self, df, target):

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")

        X = df.drop(columns=[target])
        y = df[target]

        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    # =====================================================
    # EVALUATION
    # =====================================================
    def evaluate(self, name, y_true, y_pred):

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        self.results.append({
            "model": name,
            "rmse": rmse,
            "r2_score": r2
        })

    # =====================================================
    # MAIN PIPELINE
    # =====================================================
    def run_all(self, df, target="TotalPremium"):

        self.results = []

        # STEP 1: feature engineering
        df = self.feature_engineering(df)

        # STEP 2: cleaning
        df = self.clean_features(df)

        # STEP 3: split
        X_train, X_test, y_train, y_test = self.split_data(df, target)

        # -------------------------
        # MODELS
        # -------------------------
        models = {
            "LinearRegression": LinearRegression(),

            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),

            "XGBoost": XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                objective="reg:squarederror"
            )
        }

        # TRAIN MODELS
        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            self.models[name] = model
            self.evaluate(name, y_test, preds)

        # BEST MODEL
        best = max(self.results, key=lambda x: x["r2_score"])

        self.best_model_name = best["model"]
        self.best_model = self.models[self.best_model_name]

        return {
            "results": pd.DataFrame(self.results),
            "best_model": self.best_model_name,
            "X_train": X_train,
            "X_test": X_test
        }