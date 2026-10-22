"""
01_cca_fit.py - Data Preprocessing & Standard CCA Model Fitting

Loads MRI and PET neuroimaging data, applies standardization, and fits the CCA model.
Saves intermediate results to a compressed numpy cache (.npz).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

from config import INPUT_FILE_MRI, INPUT_FILE_PET, CCA_CACHE_FILE, NUMBER_OF_COMPONENTS

def load_and_preprocess_data():
    """Reads raw CSVs, extracts subject demographics, removes subcortical regions, and scales features."""
    print("Loading MRI and PET data for CCA...")
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

    print(f"Fitting Standard CCA model with {NUMBER_OF_COMPONENTS} components...")
    cca = CCA(n_components=NUMBER_OF_COMPONENTS)
    cca.fit(scaled_mri, scaled_pet)
    mri_scores, pet_scores = cca.transform(scaled_mri, scaled_pet)

    print("Saving fitted model outputs to cache...")
    np.savez(
        CCA_CACHE_FILE,
        mri_scores=mri_scores,
        pet_scores=pet_scores,
        mri_weights=cca.x_weights_[:, 0],
        pet_weights=cca.y_weights_[:, 0],
        scaled_mri=scaled_mri,
        scaled_pet=scaled_pet,
        age=age
    )
    print(f"Standard CCA fitting complete. Cached at: {CCA_CACHE_FILE}")

if __name__ == "__main__":
    main()
