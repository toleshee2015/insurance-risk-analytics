import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluator:

    def evaluate(self, y_true, y_pred, task_type="regression"):

        # =====================================================
        # REGRESSION METRICS
        # =====================================================
        if task_type == "regression":

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            results = {
                "RMSE": rmse,
                "R2_Score": r2
            }

        # =====================================================
        # CLASSIFICATION METRICS
        # =====================================================
        elif task_type == "classification":

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

            results = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1
            }

        else:
            raise ValueError(
                "task_type must be either 'regression' or 'classification'"
            )

        return pd.DataFrame([results])