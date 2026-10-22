"""
03_pca_plots.py - Visualizations & Neuroimaging Brain Maps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import nibabel as nib
from nilearn import datasets, plotting

from config import PCA_CACHE_FILE, PLOTS_DIR, ATLAS_ROIS, ATLAS_NETWORKS

def main():
    print("Loading cached PCA results for plotting...")
    cache = np.load(PCA_CACHE_FILE)
    age = cache["age"]
    mri_scores = cache["mri_scores"]
    pet_scores = cache["pet_scores"]
    scaled_mri = cache["scaled_mri"]
    scaled_pet = cache["scaled_pet"]

    # 1. Scatterplots
    sns.set_theme(style="whitegrid")
    df_plot = pd.DataFrame({'Age': age, 'MRI_PCA': mri_scores[:, 0], 'PET_PCA': pet_scores[:, 0]})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].set_facecolor("#E6E6FA")
    sns.scatterplot(ax=axes[0], data=df_plot, x='Age', y='MRI_PCA', s=50, color="blue")
    sns.regplot(ax=axes[0], data=df_plot, x='Age', y='MRI_PCA', scatter=False, ci=None, color="blue")
    axes[0].set_title("PCA MRI (PC1 vs Age)", fontweight='bold')

    axes[1].set_facecolor("#E6E6FA")
    sns.scatterplot(ax=axes[1], data=df_plot, x='Age', y='PET_PCA', s=50, color="red")
    sns.regplot(ax=axes[1], data=df_plot, x='Age', y='PET_PCA', scatter=False, ci=None, color="red")
    axes[1].set_title("PCA PET (PC1 vs Age)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_scores_vs_age.png", dpi=300)
    plt.close()

    # 2. Brain Maps
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=ATLAS_ROIS, yeo_networks=ATLAS_NETWORKS)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()

    mri_loadings = np.array([spearmanr(scaled_mri[:, j], mri_scores[:, 0])[0] for j in range(scaled_mri.shape[1])])
    pet_loadings = np.array([spearmanr(scaled_pet[:, j], pet_scores[:, 0])[0] for j in range(scaled_pet.shape[1])])

    loadings_mri_map = np.zeros_like(atlas_data)
    loadings_pet_map = np.zeros_like(atlas_data)

    for i in range(ATLAS_ROIS):
        loadings_mri_map[atlas_data == (i + 1)] = mri_loadings[i]
        loadings_pet_map[atlas_data == (i + 1)] = pet_loadings[i]

    plotting.plot_stat_map(
        nib.Nifti1Image(loadings_mri_map, atlas_img.affine),
        display_mode="ortho", title="PCA MRI Loadings", colorbar=True
    )
    plt.savefig(PLOTS_DIR / "mri_pca_loadings_ortho.png", dpi=300)
    plt.close()

    plotting.plot_stat_map(
        nib.Nifti1Image(loadings_pet_map, atlas_img.affine),
        display_mode="ortho", title="PCA PET Loadings", colorbar=True
    )
    plt.savefig(PLOTS_DIR / "pet_pca_loadings_ortho.png", dpi=300)
    plt.close()

    print(f"Plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
