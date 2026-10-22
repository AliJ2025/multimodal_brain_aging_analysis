"""
03_cca_plots.py - Visualizations & Neuroimaging Brain Maps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from nilearn import datasets, plotting

from config import CCA_CACHE_FILE, PLOTS_DIR, ATLAS_ROIS, ATLAS_NETWORKS

def main():
    print("Loading cached CCA results for plotting...")
    cache = np.load(CCA_CACHE_FILE)
    age = cache["age"]
    mri_scores = cache["mri_scores"]
    pet_scores = cache["pet_scores"]
    mri_weights = cache["mri_weights"]
    pet_weights = cache["pet_weights"]

    # 1. Scatterplot
    sns.set_theme(style="whitegrid")
    df_plot = pd.DataFrame({'Age': age, 'MRI_CCA': mri_scores[:, 0], 'PET_CCA': pet_scores[:, 0]})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].set_facecolor("#E6E6FA")
    sns.scatterplot(ax=axes[0], data=df_plot, x='Age', y='MRI_CCA', s=50, color="blue")
    sns.regplot(ax=axes[0], data=df_plot, x='Age', y='MRI_CCA', scatter=False, ci=None, color="blue")
    axes[0].set_title("CCA MRI (CC1 vs Age)", fontweight='bold')

    axes[1].set_facecolor("#E6E6FA")
    sns.scatterplot(ax=axes[1], data=df_plot, x='Age', y='PET_CCA', s=50, color="red")
    sns.regplot(ax=axes[1], data=df_plot, x='Age', y='PET_CCA', scatter=False, ci=None, color="red")
    axes[1].set_title("CCA PET (CC1 vs Age)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cca_scores_vs_age.png", dpi=300)
    plt.close()

    # 2. Brain Maps
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=ATLAS_ROIS, yeo_networks=ATLAS_NETWORKS)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()

    weights_mri_map = np.zeros_like(atlas_data)
    weights_pet_map = np.zeros_like(atlas_data)

    for i in range(ATLAS_ROIS):
        weights_mri_map[atlas_data == (i + 1)] = mri_weights[i]
        weights_pet_map[atlas_data == (i + 1)] = pet_weights[i]

    plotting.plot_stat_map(nib.Nifti1Image(weights_mri_map, atlas_img.affine), display_mode="ortho", title="CCA MRI Weights", colorbar=True)
    plt.savefig(PLOTS_DIR / "mri_cca_weights_ortho.png", dpi=300)
    plt.close()

    plotting.plot_stat_map(nib.Nifti1Image(weights_pet_map, atlas_img.affine), display_mode="ortho", title="CCA PET Weights", colorbar=True)
    plt.savefig(PLOTS_DIR / "pet_cca_weights_ortho.png", dpi=300)
    plt.close()

    print(f"Plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
