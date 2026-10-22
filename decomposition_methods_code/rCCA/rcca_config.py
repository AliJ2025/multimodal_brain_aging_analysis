"""
Central Configuration File for Regularized CCA (rCCA) Analysis.

This module defines directory paths, file names, and hyperparameters for rCCA.
Path resolution is dynamic via pathlib to ensure multi-platform reproducibility on GitHub.
"""

from pathlib import Path

# Base project directory (dynamically resolved to project root)
BASE_DIR = Path(__file__).resolve().parent

# Input Data Paths (relative to 'data/' folder)
DATA_DIR = BASE_DIR / "data"
INPUT_FILE_MRI = DATA_DIR / "mri_cn.csv"
INPUT_FILE_PET = DATA_DIR / "pet_cn.csv"
INPUT_FILE_MOCA = DATA_DIR / "MOCA.csv"
INPUT_FILE_PSYCH = DATA_DIR / "UWNPSYCHSUM.csv"

# Output Paths (relative to 'output_rcca/' folder)
OUTPUT_DIR = BASE_DIR / "output_rcca"
CACHE_DIR = OUTPUT_DIR / "cache"
TABLES_DIR = OUTPUT_DIR / "weights_loadings"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure all output directories exist
for folder in [CACHE_DIR, TABLES_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Cache file for intermediate numpy results
RCCA_CACHE_FILE = CACHE_DIR / "rcca_results.npz"

# Model Hyperparameters
NUMBER_OF_COMPONENTS = 3
REGULARIZATION_PARAM = 0.98

# Atlas Configuration
ATLAS_ROIS = 200
ATLAS_NETWORKS = 17
