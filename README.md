# Host-Microbiome Multi-Omics Integrator

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Build](https://img.shields.io/badge/Pipeline-Bash%2FHPC-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end computational pipeline designed to integrate high-dimensional host transcriptomic and microbiome abundance profiles while addressing key statistical challenges in computational metagenomics.

---

## Key Methodological Solutions

1. **Compositional Data Bias Mitigation:** Microbiome relative abundances inherently reside in a simplex constrained space ($\sum x_i = 1$). This pipeline implements a custom `scikit-learn` `CLRTransformer` applying Centered Log-Ratio (CLR) transformations to break compositional dependencies before downstream modeling.
2. **High-Dimensional $p \gg n$ Regime:** Addresses overfitting in clinical cohorts with high-dimensional feature spaces via stratified cross-validation and feature selection with penalized Tree Ensembles (Random Forest).
3. **Model Interpretability:** Integrates Game Theory-based SHAP (SHapley Additive exPlanations) values to extract biologically meaningful host-microbiome biomarkers driving phenotype stratification.

---

## Project Architecture

```text
.
├── data/
│   ├── metadata/       # Sample metadata and phenotypes
│   ├── processed/      # CLR-transformed matrices
│   └── raw/            # Raw abundance tables
├── pipeline/
│   └── run_analysis.sh # HPC-ready PBS orchestration script
├── results/
│   └── figures/        # SHAP interpretability outputs
├── src/
│   ├── data_prep/      # CLR Custom Transformer (scikit-learn API)
│   ├── models/         # Machine Learning training modules
│   └── utils/          # SHAP explainers and dummy data generators
└── environment.yml     # Conda environment definition
