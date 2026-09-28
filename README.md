# HostMicrobiome-ML-Integrator

A computational biology pipeline scaffold for integrative machine-learning analyses of microbiome and transcriptomic datasets.

## Background

Host-associated microbiome data are compositional: sequencing counts represent relative abundances constrained to a simplex, making direct Euclidean modeling inappropriate without transformations such as Centered Log-Ratio (CLR). In parallel, transcriptomics data are typically high-dimensional with **p >> n** (features greatly exceed sample size), requiring careful regularization, feature processing, and robust validation strategies. This repository provides a professional project structure to support reproducible, extensible multi-omics integration workflows.

## Directory Structure

```text
HostMicrobiome-ML-Integrator/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── notebooks/
├── pipeline/
│   └── run_analysis.sh
├── results/
│   ├── figures/
│   └── tables/
└── src/
    ├── data_prep/
    │   └── compositional_transform.py
    ├── models/
    │   └── train_evaluate.py
    └── utils/
```

## Installation

1. Install [Conda](https://docs.conda.io/).
2. Create the project environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate hostmicrobiome-ml-integrator
   ```

## Usage

Run the end-to-end analysis pipeline script:

```bash
bash pipeline/run_analysis.sh
```

The script is PBS-ready for HPC usage and executes data preparation and model training sequentially with `set -e` for fail-fast behavior.
