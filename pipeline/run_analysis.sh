#!/usr/bin/env bash
#PBS -N multiomics_benchmark
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METRICS_SUMMARY="$PROJECT_ROOT/results/tables/multi_cohort_metrics.csv"

rm -f "$METRICS_SUMMARY"

echo "[STAGE 1] Generating multi-cohort synthetic datasets..."
python "$PROJECT_ROOT/src/data_prep/download_datasets.py"
if [ -n "$1" ]; then
  DATASETS=("$1")
else
  DATASETS=("ZellerG_2014" "FranzosaEA_2019" "WirbelJ_2019")
fi
for DS in "${DATASETS[@]}"; do
  echo "[STAGE 2] Processing cohort: $DS"

  python "$PROJECT_ROOT/src/data_prep/compositional_transform.py" \
    --input "$PROJECT_ROOT/data/raw/${DS}_microbiome.csv" \
    --output "$PROJECT_ROOT/data/processed/${DS}_microbiome_clr.csv"

  python "$PROJECT_ROOT/src/models/train_evaluate.py" \
    --input "$PROJECT_ROOT/data/processed/${DS}_microbiome_clr.csv" \
    --metadata "$PROJECT_ROOT/data/metadata/${DS}_metadata.csv" \
    --model-out "$PROJECT_ROOT/results/${DS}_model.pkl" \
    --metrics-out "$METRICS_SUMMARY" \
    --dataset-name "$DS"

  python "$PROJECT_ROOT/src/utils/shap_explainer.py" \
    --model "$PROJECT_ROOT/results/${DS}_model.pkl" \
    --output "$PROJECT_ROOT/results/figures/${DS}_shap.png" \
    --dataset-name "$DS"
done

echo "[SUCCESS] Multi-cohort pipeline execution completed."
