"""
04_rcca_export.py - Summary Report & Model Validation Export
"""

import numpy as np
from scipy.stats import spearmanr, skew

from config import RCCA_CACHE_FILE, OUTPUT_DIR

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
    print("Generating comprehensive rCCA execution report...")
    cache = np.load(RCCA_CACHE_FILE)
    mri_scores = cache["mri_scores"]
    pet_scores = cache["pet_scores"]
    mri_weights = cache["mri_weights"]
    pet_weights = cache["pet_weights"]
    age = cache["age"]

    c_12 = spearmanr(mri_scores[:, 0], age)[0]
    c_21 = spearmanr(pet_scores[:, 0], age)[0]
    c_33 = spearmanr(mri_scores[:, 0], pet_scores[:, 0])[0]

    report_path = OUTPUT_DIR / "rcca_summary_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("          rCCA SUMMARY REPORT                        \n")
        f.write("=====================================================\n\n")

        f.write("1. MODEL CORRELATIONS:\n")
        f.write(f" - MRI rCC1 vs. Age Spearman r        : {c_12:.4f}\n")
        f.write(f" - PET rCC1 vs. Age Spearman r        : {c_21:.4f}\n")
        f.write(f" - Intermodal Correlation (MRI vs PET): {c_33:.4f}\n\n")

        f.write("2. CANONICAL WEIGHTS DESCRIPTIVE STATS:\n")
        f.write(get_stats_str(mri_weights, "MRI Weights"))
        f.write(get_stats_str(pet_weights, "PET Weights"))

    print(f"Summary report saved at: {report_path}")

if __name__ == "__main__":
    main()
