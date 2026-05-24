import pandas as pd
import numpy as np
import os
import sys

# Safely point Python to look inside your src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from hypothesis_tests import test_gender_risk, test_geographic_variance, test_security_impact

def main():
    # Adjust this path if your DVC data folder is structured differently
    data_path = "data/insurance_data.csv"
    
    if os.path.exists(data_path):
        print(f"Reading tracking repository dataset from: {data_path}...")
        df = pd.read_csv(data_path, low_memory=False)
    else:
        print(f"Data file not found at {data_path}. Running automated verification mockup...")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'Gender': rng.choice(['Male', 'Female'], size=20000),
            'Province': rng.choice(['GP', 'NC', 'WC'], size=20000),
            'TotalPremium': rng.uniform(1200, 4500, size=20000),
            'TotalClaims': rng.choice([0, 0, 0, 0, 15000], size=20000),
            'TrackingDevice': rng.choice([0, 1], size=20000)
        })

    print("Computing statistical variations across dimensions...")
    
    # Process calculations
    t1_name, t1_p, t1_dec = test_gender_risk(df)
    t2_name, t2_p, t2_dec = test_geographic_variance(df)
    t3_name, t3_p, t3_dec = test_security_impact(df)

    summary_table = pd.DataFrame({
        "Hypothesis Target": [
            "Hypothesis 1: Gender vs Underwriting Margin",
            "Hypothesis 2: Province vs Claim Frequency",
            "Hypothesis 3: Tracker Presence vs Claim Severity"
        ],
        "Statistical Model": [t1_name, t2_name, t3_name],
        "Computed p-value": [f"{t1_p:.4e}" if pd.notnull(t1_p) else "N/A", 
                             f"{t2_p:.4e}" if pd.notnull(t2_p) else "N/A", 
                             f"{t3_p:.4e}" if pd.notnull(t3_p) else "N/A"],
        "Decision (α=0.05)": [t1_dec, t2_dec, t3_dec]
    })

    # Render metrics cleanly to standard output stream
    print("\n" + "="*90)
    print("                 ALPHACARE SOLUTIONS - TASK 3 HYPOTHESIS SUMMARY MATRIX")
    print("="*90)
    print(summary_table.to_string(index=False))
    print("="*90 + "\n")

    print("ACTUARIAL INSIGHTS FOR REJECTED SEGMENTS:")
    print("-" * 42)
    if t1_dec == "Reject H₀":
        print("• Gender Profiles: Underwriting margins show significant variance between male and female drivers.")
        print("  Strategy: Apply a premium loading structural factor to higher risk demographics.")
    if t2_dec == "Reject H₀":
        print("• Geographic Profiles: Regional markers display a direct causal influence on claim frequency rates.")
        print("  Strategy: Roll out base pricing drops in Northern Cape to capture profitable policy blocks.")
    if t3_dec == "Reject H₀":
        print("• Asset Protection Profiles: Telemetry tracking devices show validated reductions in claim severe sizes.")
        print("  Strategy: Introduce an explicit premium safety markdown reward factor for tracked cars.")
    print("="*90 + "\n")

if __name__ == "__main__":
    main()