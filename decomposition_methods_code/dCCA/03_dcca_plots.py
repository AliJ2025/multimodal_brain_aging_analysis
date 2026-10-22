"""
03_dcca_plots.py - Visualizations & Neuroimaging Brain Maps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy.stats import spearmanr
import nibabel as nib
from nilearn import datasets, plotting

from config import DCCA_CACHE_FILE, PLOTS_DIR, ATLAS_ROIS, ATLAS_NETWORKS

def plot_age_scatter(age, C_1_hat, C_2_hat):
    sns.set_theme(style="whitegrid")
    df_plot = pd.DataFrame({'Age': age, 'MRI_Canonical': C_1_hat[0, :], 'PET_Canonical': C_2_hat[0, :]})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("dCCA Canonical Scores vs. Age", fontsize=16, fontweight='bold')

    axes[0].set_facecolor("#E6E6FA")
    sns.scatterplot(ax=axes[0], data=df_plot, x='Age', y='MRI_Canonical', s=50, color="blue")
    sns.regplot(ax=axes[0], data=df_plot, x='Age', y='MRI_Canonical', scatter=False, ci=None, color="blue")
    axes[0].set_title("MRI Canonical Score", fontweight='bold')

    axes[1].set_facecolor("#E6E6FA")
    sns.scatterplot(ax=axes[1], data=df_plot, x='Age', y='PET_Canonical', s=50, color="red")
    sns.regplot(ax=axes[1], data=df_plot, x='Age', y='PET_Canonical', scatter=False, ci=None, color="red")
    axes[1].set_title("PET Canonical Score", fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "canonical_scores_vs_age.png", dpi=300)
    plt.close()

def plot_brain_loadings(scaled_mri, scaled_pet, C_1_hat, C_2_hat):
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=ATLAS_ROIS, yeo_networks=ATLAS_NETWORKS)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    mri_loadings = np.array([spearmanr(scaled_mri[:, j], C_1_hat[0, :])[0] for j in range(scaled_mri.shape[1])])
    pet_loadings = np.array([spearmanr(scaled_pet[:, j], C_2_hat[0, :])[0] for j in range(scaled_pet.shape[1])])

    loadings_mri_map = np.zeros_like(atlas_data)
    loadings_pet_map = np.zeros_like(atlas_data)

    for i in range(ATLAS_ROIS):
        loadings_mri_map[atlas_data == (i + 1)] = mri_loadings[i]
        loadings_pet_map[atlas_data == (i + 1)] = pet_loadings[i]

    mri_load_img = nib.Nifti1Image(loadings_mri_map.astype(float), affine)
    pet_load_img = nib.Nifti1Image(loadings_pet_map.astype(float), affine)

    vmin = 0
    vmax = max(np.max(mri_loadings), np.max(pet_loadings))
    cut_coords = (2, -32, 11)

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1.2], hspace=0.30)

    ax_mri = fig.add_subplot(gs[0, 0])
    plotting.plot_stat_map(mri_load_img, figure=fig, axes=ax_mri, display_mode='ortho',
                           cut_coords=cut_coords, cmap='viridis', black_bg=False, vmin=vmin, vmax=vmax, colorbar=False)
    ax_mri.set_title("MRI Loadings", y=1.10, fontweight='bold')

    ax_pet = fig.add_subplot(gs[1, 0])
    plotting.plot_stat_map(pet_load_img, figure=fig, axes=ax_pet, display_mode='ortho',
                           cut_coords=cut_coords, cmap='viridis', black_bg=False, vmin=vmin, vmax=vmax, colorbar=False)
    ax_pet.set_title("PET Loadings", y=1.10, fontweight='bold')

    ax_cbar = fig.add_subplot(gs[:, 1])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.3f}", f"{vmax:.3f}"])

    plt.savefig(PLOTS_DIR / "brain_loadings_ortho.png", dpi=300)
    plt.close()

def main():
    print("Loading cached dCCA results for plotting...")
    cache = np.load(DCCA_CACHE_FILE)
    plot_age_scatter(cache["age"], cache["C_1_hat"], cache["C_2_hat"])
    plot_brain_loadings(cache["scaled_mri"], cache["scaled_pet"], cache["C_1_hat"], cache["C_2_hat"])
    print(f"All plots successfully generated and saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
