"""
01_pca_fit.py - Data Preprocessing & PCA Model Fitting

Loads MRI and PET neuroimaging data, applies standardization, and fits separate PCA models.
Saves intermediate results and explained variance ratios to a compressed numpy cache (.npz).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import INPUT_FILE_MRI, INPUT_FILE_PET, PCA_CACHE_FILE, NUMBER_OF_COMPONENTS

def load_and_preprocess_data():
    """Reads raw CSVs, extracts subject demographics, removes subcortical regions, and scales features."""
    print("Loading MRI and PET data for PCA...")
    df_mri = pd.read_csv(INPUT_FILE_MRI)
    df_pet = pd.read_csv(INPUT_FILE_PET)

    age = df_mri["age"].values
    df_mri_info = df_mri[['name', 'sess', 'age', 'Dataset']]

    df_mri_features = df_mri.drop(['name', 'sess', 'age', 'Dataset'], axis=1)
    df_pet_features = df_pet.drop(['name', 'sess', 'age', 'Dataset'], axis=1)

    # Remove subcortical regions (last 16 columns)
    df_mri_features = df_mri_features.iloc[:, :-16]
    df_pet_features = df_pet_features.iloc[:, :-16]

    scaler = StandardScaler()
    scaled_mri = scaler.fit_transform(df_mri_features)
    scaled_pet = scaler.fit_transform(df_pet_features)

    return df_mri_info, df_mri_features, df_pet_features, scaled_mri, scaled_pet, age

def main():
    df_mri_info, df_mri_features, df_pet_features, scaled_mri, scaled_pet, age = load_and_preprocess_data()

    print(f"Fitting PCA models with {NUMBER_OF_COMPONENTS} components...")
    pca_mri = PCA(n_components=NUMBER_OF_COMPONENTS)
    mri_scores = pca_mri.fit_transform(scaled_mri)

    pca_pet = PCA(n_components=NUMBER_OF_COMPONENTS)
    pet_scores = pca_pet.fit_transform(scaled_pet)

    print("Saving fitted model outputs to cache...")
    np.savez(
        PCA_CACHE_FILE,
        mri_scores=mri_scores,
        pet_scores=pet_scores,
        mri_components=pca_mri.components_,
        pet_components=pca_pet.components_,
        mri_exp_var=pca_mri.explained_variance_ratio_,
        pet_exp_var=pca_pet.explained_variance_ratio_,
        scaled_mri=scaled_mri,
        scaled_pet=scaled_pet,
        age=age
    )
    print(f"PCA fitting complete. Cached at: {PCA_CACHE_FILE}")

if __name__ == "__main__":
    main()
