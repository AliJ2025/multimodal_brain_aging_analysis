"""
01_dcca_fit.py - Data Preprocessing & dCCA Model Fitting

Loads MRI and PET neuroimaging data, applies standardization, and fits the dCCA model.
Saves intermediate results to a compressed numpy cache (.npz).

Attribution & Implementation Notice:
------------------------------------
This script uses a slightly modified version ('dcca_.py') of the original dCCA 
implementation by Hai Shu et al. (Source: https://github.com/shu-hai/D-CCA).

Modifications in 'dcca_.py':
1. Replaced deprecated 'np.asscalar()' with '.item()' for modern NumPy compatibility.
2. Updated 'dCCA()' return signature to also yield 'Gamma_1' and 'Gamma_2' 
   (canonical weight matrices) required for feature weight analysis and atlas mapping.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    INPUT_FILE_MRI, INPUT_FILE_PET, DCCA_CACHE_FILE,
    R_1, R_2, R_12
)

try:
    import dcca_
except ImportError:
    raise ImportError(
        "Module 'dcca_' not found. Please ensure 'dcca_.py' (modified from "
        "https://github.com/shu-hai/D-CCA) is placed in the project root directory."
    )

def load_and_preprocess_data():
    """Reads raw CSVs, extracts subject demographics, removes subcortical regions, and scales features."""
    print("Loading MRI and PET data...")
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

    print(f"Fitting dCCA model with parameters: r_1={R_1}, r_2={R_2}, r_12={R_12}...")
    X_1_hat, X_2_hat, C_1_hat, C_2_hat, D_1_hat, D_2_hat, r_1_hat, r_2_hat, r_12_hat, ccor_hat, theta_hat, C_base, Gamma_1, Gamma_2 = \
        dcca_.dCCA(scaled_mri.T, scaled_pet.T, r_1=R_1, r_2=R_2, r_12=R_12)

    print("Saving fitted model outputs to cache...")
    np.savez(
        DCCA_CACHE_FILE,
        C_1_hat=C_1_hat,
        C_2_hat=C_2_hat,
        D_1_hat=D_1_hat,
        D_2_hat=D_2_hat,
        C_base=C_base,
        Gamma_1=Gamma_1,
        Gamma_2=Gamma_2,
        scaled_mri=scaled_mri,
        scaled_pet=scaled_pet,
        age=age
    )
    print(f"Model fitting complete. Cached at: {DCCA_CACHE_FILE}")

if __name__ == "__main__":
    main()
