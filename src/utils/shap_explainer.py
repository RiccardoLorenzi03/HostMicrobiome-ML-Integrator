import pandas as pd
import numpy as np
import argparse
import os
import joblib
import shap
import matplotlib.pyplot as plt

def generate_shap_plot(model_path, data_path, figure_output_path):
    print("Caricamento modello e dati per l'analisi SHAP...")
    rf, feature_names = joblib.load(model_path)
    df = pd.read_csv(data_path, index_col=0)

    # Calcolo dei valori SHAP per il TreeEnsemble
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(df.values)

    # Gestione delle differenze di formato nelle versioni di SHAP per classificatori binari
    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, df, feature_names=feature_names, show=False)
    
    os.makedirs(os.path.dirname(figure_output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figure_output_path, dpi=300)
    plt.close()
    print(f"Grafico SHAP salvato con successo in {figure_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola e salva i valori SHAP per il modello ML.")
    parser.add_argument("--model", type=str, required=True, help="Path del modello addestrato.")
    parser.add_argument("--data", type=str, required=True, help="Path dati processati.")
    parser.add_argument("--output", type=str, required=True, help="Path salvataggio figura PNG.")
    args = parser.parse_args()

    generate_shap_plot(args.model, args.data, args.output)
