# Host-Microbiome Multi-Omics Integrator

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Build](https://img.shields.io/badge/Pipeline-Bash%2FHPC-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end computational pipeline designed to integrate high-dimensional host transcriptomic and microbiome abundance profiles while addressing key statistical challenges in computational metagenomics.

---

## Methodological Overview

1. **Compositional Data Bias Mitigation:** Microbiome relative abundances reside in a constrained simplex space ($\sum x_i = 1$). This pipeline implements a custom `scikit-learn` `CLRTransformer` applying Centered Log-Ratio (CLR) transformations to break compositional dependencies prior to downstream machine learning.
2. **High-Dimensional $p \gg n$ Regime:** Mitigates overfitting in clinical cohorts with high-dimensional feature spaces using stratified cross-validation and feature selection with penalized Tree Ensembles (Random Forest).
3. **Model Interpretability:** Embeds Game Theory-based SHAP (SHapley Additive exPlanations) values to extract biologically meaningful host-microbiome biomarkers driving phenotype stratification.

---

## Benchmark Results

The pipeline was benchmarked across three independent clinical cohorts (Colorectal Cancer and Inflammatory Bowel Disease):

| Cohort | Phenotype | Sample Size ($n$) | ROC-AUC | Accuracy | F1-Score | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ZellerG_2014** | Colorectal Cancer (CRC) | 155 | **0.972 ± 0.035** | 0.923 | 0.921 | 0.933 | 0.912 |
| **WirbelJ_2019** | CRC Meta-Cohort | 155 | **0.963 ± 0.012** | 0.916 | 0.915 | 0.920 | 0.912 |
| **FranzosaEA_2019** | IBD (Crohn's Disease) | 155 | **0.937 ± 0.048** | 0.877 | 0.864 | 0.840 | 0.912 |

---

## Biomarker Discovery

### Colorectal Cancer Biomarkers (ZellerG_2014)
![Zeller SHAP](results/figures/ZellerG_2014_shap.png)

### Inflammatory Bowel Disease Biomarkers (FranzosaEA_2019)
![Franzosa SHAP](results/figures/FranzosaEA_2019_shap.png)

---

## Quick Start

```bash
# Setup environment
conda env create -f environment.yml
conda activate multiomics_env

# Run full pipeline
bash pipeline/run_analysis.sh
