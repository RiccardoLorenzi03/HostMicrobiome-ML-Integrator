import argparse
import os
import joblib
import matplotlib.pyplot as plt
import shap

def generate_shap_plot(model_path: str, output_path: str, dataset_name: str) -> None:
    rf, feature_names, X = joblib.load(model_path)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X.values)

    vals = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X, feature_names=feature_names, show=False)
    plt.title(f"SHAP Biomarker Importance - {dataset_name}", fontsize=13, pad=12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] SHAP plot saved to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    generate_shap_plot(args.model, args.output, args.dataset_name)

if __name__ == "__main__":
    main()
