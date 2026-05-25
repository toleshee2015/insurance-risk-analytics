HYPOTHESIS_REGISTRY = {
    "H1_province_risk": {
        "description": "No risk differences across provinces",
        "kpis": ["claim_frequency", "claim_severity"],
        "tests": ["chi_square", "welch_ttest"],
        "group_col": "province"
    },

    "H2_zip_risk": {
        "description": "No risk differences between zip codes",
        "kpis": ["claim_frequency", "claim_severity"],
        "tests": ["chi_square", "welch_ttest"],
        "group_col": "zip_code"
    },

    "H3_zip_margin": {
        "description": "No margin difference between zip codes",
        "kpi": "margin",
        "test": "welch_ttest",
        "group_col": "zip_code"
    },

    "H4_gender_risk": {
        "description": "No risk differences between Women and Men",
        "kpis": ["claim_frequency", "claim_severity"],
        "tests": ["chi_square", "welch_ttest"],
        "group_col": "gender"
    }
}