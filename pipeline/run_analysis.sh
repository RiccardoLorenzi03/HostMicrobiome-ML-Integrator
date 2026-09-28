#!/usr/bin/env bash
#PBS -N multiomics_integration
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$PROJECT_ROOT/src/data_prep/compositional_transform.py"
python "$PROJECT_ROOT/src/models/train_evaluate.py" \
  --features "$PROJECT_ROOT/data/processed/combined_features.csv" \
  --target phenotype
