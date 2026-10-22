"""
04_dcca_export.py - Summary Report & Model Validation Export
"""

import numpy as np
from scipy.stats import spearmanr, skew

from config import DCCA_CACHE_FILE, OUTPUT_DIR

def get_stats_str(arr, label):
    return (
        f"Stats for {label}:\n"
        f" Mean    : {np.mean(arr):.4f}\n"
        f" Median  : {np.median(arr):.4f}\n"
        f" Std Dev : {np.std(arr):.4f}\n"
        f" Min     : {np.min(arr):.4f}\n"
        f" Max     : {np.max(arr):.4f}\n"
        f" Skewness: {skew(arr):.4f}\n"
        f"{'-'*30}\n"
    )

def main():
    print("Generating comprehensive dCCA execution report...")
    cache = np.load(DCCA_CACHE_FILE)
    C_1_hat = cache["C_1_hat"]
    C_2_hat = cache["C_2_hat"]
    D_1_hat = cache["D_1_hat"]
    D_2_hat = cache["D_2_hat"]
    Gamma_1 = cache["Gamma_1"]
    Gamma_2 = cache["Gamma_2"]
    age = cache["age"]

    weights_mri = Gamma_1[:, 0]
    weights_pet = Gamma_2[:, 0]

    c_12 = spearmanr(C_1_hat[0, :], age)[0]
    c_21 = spearmanr(C_2_hat[0, :], age)[0]
    c_33 = spearmanr(C_1_hat[0, :], C_2_hat[0, :])[0]
    c_44 = spearmanr(D_1_hat[0, :], D_2_hat[0, :])[0]

    report_path = OUTPUT_DIR / "dcca_summary_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("          dCCA SUMMARY REPORT                        \n")
        f.write("=====================================================\n\n")

        f.write("1. MODEL DIMENSIONS & CORRELATIONS:\n")
        f.write(f" - C_1_hat shape: {C_1_hat.shape}\n")
        f.write(f" - C_2_hat shape: {C_2_hat.shape}\n")
        f.write(f" - MRI 1st Canonical vs. Age Spearman r : {c_12:.4f}\n")
        f.write(f" - PET 1st Canonical vs. Age Spearman r : {c_21:.4f}\n")
        f.write(f" - Intermodal Correlation (MRI vs PET)  : {c_33:.4f}\n")
        f.write(f" - Distinct Component Check Correlation : {c_44:.4f}\n\n")

        f.write("2. CANONICAL WEIGHTS DESCRIPTIVE STATS:\n")
        f.write(get_stats_str(weights_mri, "MRI Weights"))
        f.write(get_stats_str(weights_pet, "PET Weights"))

    print(f"Summary report saved successfully at: {report_path}")

if __name__ == "__main__":
    main()
