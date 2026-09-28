#!/usr/bin/env bash
#PBS -N multiomics_integration
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "1/2 Generazione dataset fittizio..."
python "$PROJECT_ROOT/src/utils/generate_dummy_data.py"

echo "2/2 Esecuzione trasformazione composizionale (CLR)..."
python "$PROJECT_ROOT/src/data_prep/compositional_transform.py" \
  --input "$PROJECT_ROOT/data/raw/dummy_microbiome.csv" \
  --output "$PROJECT_ROOT/data/processed/dummy_microbiome_clr.csv"

echo "Pipeline (Fase 1) conclusa con successo."
