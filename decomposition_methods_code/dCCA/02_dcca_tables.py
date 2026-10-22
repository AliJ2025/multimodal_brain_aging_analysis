"""
02_dcca_tables.py - Statistical Analyses, Feature Matching & CSV Exports
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    INPUT_FILE_MRI, INPUT_FILE_PET, INPUT_FILE_MOCA, INPUT_FILE_PSYCH,
    DCCA_CACHE_FILE, TABLES_DIR
)

def main():
    print("Loading cached dCCA model results...")
    cache = np.load(DCCA_CACHE_FILE)
    C_1_hat = cache["C_1_hat"]
    C_2_hat = cache["C_2_hat"]
    Gamma_1 = cache["Gamma_1"]
    Gamma_2 = cache["Gamma_2"]
    scaled_mri = cache["scaled_mri"]
    scaled_pet = cache["scaled_pet"]

    df_mri = pd.read_csv(INPUT_FILE_MRI)
    df_pet = pd.read_csv(INPUT_FILE_PET)
    df_mri_info = df_mri[['name', 'sess', 'age', 'Dataset']].copy()
    df_mri_cols = df_mri.drop(['name', 'sess', 'age', 'Dataset'], axis=1).iloc[:, :-16].columns
    df_pet_cols = df_pet.drop(['name', 'sess', 'age', 'Dataset'], axis=1).iloc[:, :-16].columns

    # 1. Feature vs Score Correlations
    mri_score = C_1_hat[0, :]
    pet_score = C_2_hat[0, :]
    rows = []

    for col in df_mri_cols:
        r, p = spearmanr(pet_score, df_mri[col].values)
        rows.append({"correlation_type": "PET_score_vs_original_MRI", "feature": col, "spearman_r": r, "p_value": p})

    for col in df_pet_cols:
        r, p = spearmanr(mri_score, df_pet[col].values)
        rows.append({"correlation_type": "MRI_score_vs_original_PET", "feature": col, "spearman_r": r, "p_value": p})

    pd.DataFrame(rows).to_csv(TABLES_DIR / "score_vs_original_feature_spearman.csv", index=False)

    # 2. Weights & Loadings Export
    weights_mri = Gamma_1[:, 0]
    weights_pet = Gamma_2[:, 0]
    pd.DataFrame({'MRI_weights': weights_mri, 'PET_weights': weights_pet}).to_csv(TABLES_DIR / "weights.csv", index=False)

    loadings_mri = np.array([spearmanr(scaled_mri[:, i], C_1_hat[0, :])[0] for i in range(scaled_mri.shape[1])])
    loadings_pet = np.array([spearmanr(scaled_pet[:, i], C_2_hat[0, :])[0] for i in range(scaled_pet.shape[1])])
    pd.DataFrame({'MRI_loadings': loadings_mri, 'PET_loadings': loadings_pet, 'Region': df_mri_cols}).to_csv(TABLES_DIR / "loadings.csv", index=False)

    # 3. Clinical & Cognitive Matching
    moca = pd.read_csv(INPUT_FILE_MOCA)
    psych = pd.read_csv(INPUT_FILE_PSYCH)

    moca.columns = moca.columns.str.strip()
    df_mri_info.columns = df_mri_info.columns.str.strip()
    moca['PTID'] = moca['PTID'].astype(str).str.strip().str.upper()
    df_mri_info['name'] = df_mri_info['name'].astype(str).str.strip().str.upper()

    moca_subset = moca[['PTID', 'Phase', 'RID', 'VISCODE2']]
    moca_first = moca_subset.sort_values('VISCODE2').drop_duplicates(subset=['PTID'], keep='first')
    df_merged = df_mri_info.merge(moca_first, left_on='name', right_on='PTID', how='left')

    psych_renamed = psych.rename(columns={'PHASE': 'Phase'})
    psych_renamed['RID'] = psych_renamed['RID'].astype('Int64')
    psych_subset = psych_renamed[['RID', 'VISCODE2', 'Phase', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'ADNI_EF2']]

    matched_both = df_merged.merge(psych_subset, on=['RID', 'VISCODE2', 'Phase'], how='inner')

    df_mri_info["U_mri"] = C_1_hat[0, :]
    df_mri_info["U_pet"] = C_2_hat[0, :]
    matched_scores = df_mri_info.merge(matched_both[["name", "sess"]], on=["name", "sess"], how="inner")

    print("\nCognitive Domain Correlations with dCCA Scores:")
    for domain in ["ADNI_MEM", "ADNI_LAN", "ADNI_EF", "ADNI_EF2", "ADNI_VS"]:
        if domain in matched_both.columns:
            r_mri = spearmanr(matched_both[domain], matched_scores["U_mri"])[0]
            r_pet = spearmanr(matched_both[domain], matched_scores["U_pet"])[0]
            print(f" {domain:9s} -> MRI: {r_mri:.4f} | PET: {r_pet:.4f}")

    print(f"\nTable exports finished successfully in: {TABLES_DIR}")

if __name__ == "__main__":
    main()
