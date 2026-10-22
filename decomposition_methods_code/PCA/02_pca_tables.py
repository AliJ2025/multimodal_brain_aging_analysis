"""
02_pca_tables.py - Statistical Analyses, Feature Matching & CSV Exports
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    INPUT_FILE_MRI, INPUT_FILE_PET, INPUT_FILE_MOCA, INPUT_FILE_PSYCH,
    PCA_CACHE_FILE, TABLES_DIR
)

def main():
    print("Loading cached PCA model results...")
    cache = np.load(PCA_CACHE_FILE)
    mri_scores = cache["mri_scores"]
    pet_scores = cache["pet_scores"]
    mri_components = cache["mri_components"]
    pet_components = cache["pet_components"]
    scaled_mri = cache["scaled_mri"]
    scaled_pet = cache["scaled_pet"]

    df_mri = pd.read_csv(INPUT_FILE_MRI)
    df_mri_info = df_mri[['name', 'sess', 'age', 'Dataset']].copy()
    df_mri_cols = df_mri.drop(['name', 'sess', 'age', 'Dataset'], axis=1).iloc[:, :-16].columns

    # 1. Weights & Loadings Export
    mri_weights = mri_components[0]
    pet_weights = pet_components[0]
    pd.DataFrame({'MRI_weights': mri_weights, 'PET_weights': pet_weights}).to_csv(TABLES_DIR / "weights.csv", index=False)

    loadings_mri = np.array([spearmanr(scaled_mri[:, i], mri_scores[:, 0])[0] for i in range(scaled_mri.shape[1])])
    loadings_pet = np.array([spearmanr(scaled_pet[:, i], pet_scores[:, 0])[0] for i in range(scaled_pet.shape[1])])
    pd.DataFrame({'MRI_loadings': loadings_mri, 'PET_loadings': loadings_pet, 'Region': df_mri_cols}).to_csv(TABLES_DIR / "loadings.csv", index=False)

    # 2. Clinical & Cognitive Matching
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

    df_mri_info["U_mri"] = mri_scores[:, 0]
    df_mri_info["U_pet"] = pet_scores[:, 0]
    matched_scores = df_mri_info.merge(matched_both[["name", "sess"]], on=["name", "sess"], how="inner")

    print("\nCognitive Domain Correlations with PC1 Scores:")
    for domain in ["ADNI_MEM", "ADNI_LAN", "ADNI_EF", "ADNI_EF2", "ADNI_VS"]:
        if domain in matched_both.columns:
            r_mri = spearmanr(matched_both[domain], matched_scores["U_mri"])[0]
            r_pet = spearmanr(matched_both[domain], matched_scores["U_pet"])[0]
            print(f" {domain:9s} -> MRI: {r_mri:.4f} | PET: {r_pet:.4f}")

    print(f"\nTable exports finished successfully in: {TABLES_DIR}")

if __name__ == "__main__":
    main()
