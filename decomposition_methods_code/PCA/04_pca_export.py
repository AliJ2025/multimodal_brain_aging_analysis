"""
04_pca_export.py - Summary Report & Model Validation Export
"""

import numpy as np
from scipy.stats import spearmanr, skew

from config import PCA_CACHE_FILE, OUTPUT_DIR

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
    print("Generating comprehensive PCA execution report...")
    cache = np.load(PCA_CACHE_FILE)
    mri_scores = cache["mri_scores"]
    pet_scores = cache["pet_scores"]
    mri_components = cache["mri_components"]
    pet_components = cache["pet_components"]
    mri_exp_var = cache["mri_exp_var"]
    pet_exp_var = cache["pet_exp_var"]
    age = cache["age"]

    r_mri_age = spearmanr(mri_scores[:, 0], age)[0]
    r_pet_age = spearmanr(pet_scores[:, 0], age)[0]
    r_intermodal = spearmanr(mri_scores[:, 0], pet_scores[:, 0])[0]

    report_path = OUTPUT_DIR / "pca_summary_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("          PCA SUMMARY REPORT                         \n")
        f.write("=====================================================\n\n")

        f.write("1. EXPLAINED VARIANCE RATIOS:\n")
        f.write(f" - MRI PC1 Explained Variance        : {mri_exp_var[0]:.4f}\n")
        f.write(f" - PET PC1 Explained Variance        : {pet_exp_var[0]:.4f}\n")
        f.write(f" - MRI Cumulative Variance (3 PCs)   : {np.sum(mri_exp_var):.4f}\n")
        f.write(f" - PET Cumulative Variance (3 PCs)   : {np.sum(pet_exp_var):.4f}\n\n")

        f.write("2. MODEL CORRELATIONS:\n")
        f.write(f" - MRI PC1 vs. Age Spearman r        : {r_mri_age:.4f}\n")
        f.write(f" - PET PC1 vs. Age Spearman r        : {r_pet_age:.4f}\n")
        f.write(f" - Intermodal Correlation (MRI vs PET): {r_intermodal:.4f}\n\n")

        f.write("3. PRINCIPAL COMPONENT WEIGHTS DESCRIPTIVE STATS:\n")
        f.write(get_stats_str(mri_components[0], "MRI PC1 Weights"))
        f.write(get_stats_str(pet_components[0], "PET PC1 Weights"))

    print(f"Summary report saved at: {report_path}")

if __name__ == "__main__":
    main()
