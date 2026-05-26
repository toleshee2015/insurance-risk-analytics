import numpy as np
import pandas as pd

def claim_frequency(df, group_col):
    return df.groupby(group_col)["has_claim"].mean()

def claim_severity(df):
    return df[df["has_claim"] == 1]["claim_amount"].mean()

def margin(df):
    return df["premium"] - df["claim_amount"]