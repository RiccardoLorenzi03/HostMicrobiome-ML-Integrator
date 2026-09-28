#!/usr/bin/env bash
#PBS -N multiomics_integration
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== STAGE 1: Generazione Toy Dataset ==="
python "$PROJECT_ROOT/src/utils/generate_dummy_data.py"

echo "=== STAGE 2: Trasformazione Composizionale (CLR) ==="
python "$PROJECT_ROOT/src/data_prep/compositional_transform.py" \
  --input "$PROJECT_ROOT/data/raw/dummy_microbiome.csv" \
  --output "$PROJECT_ROOT/data/processed/dummy_microbiome_clr.csv"

echo "=== STAGE 3: Addestramento Modello ML & Valutazione ==="
python "$PROJECT_ROOT/src/models/train_evaluate.py" \
  --input "$PROJECT_ROOT/data/processed/dummy_microbiome_clr.csv" \
  --output "$PROJECT_ROOT/results/rf_model.pkl"

echo "=== STAGE 4: Estrazione Biomarcatori e Plotting SHAP ==="
python "$PROJECT_ROOT/src/utils/shap_explainer.py" \
  --model "$PROJECT_ROOT/results/rf_model.pkl" \
  --data "$PROJECT_ROOT/data/processed/dummy_microbiome_clr.csv" \
  --output "$PROJECT_ROOT/results/figures/shap_summary.png"

echo "=== PIPELINE COMPLETATA CON SUCCESSO ==="
