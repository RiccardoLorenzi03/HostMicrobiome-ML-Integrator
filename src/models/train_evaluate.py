import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

def train_and_evaluate(
    input_path: str,
    metadata_path: str,
    model_out: str,
    metrics_out: str,
    dataset_name: str,
    target_col: str = "study_condition",
    control_val: str = "control"
) -> None:
    df_features = pd.read_csv(input_path, index_col=0)
    df_meta = pd.read_csv(metadata_path, index_col=0)

    common_samples = df_features.index.intersection(df_meta.index)
    X = df_features.loc[common_samples]
    y = (df_meta.loc[common_samples, target_col] != control_val).astype(int).values

    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {'auc': [], 'acc': [], 'f1': [], 'sens': [], 'spec': []}

    for train_idx, val_idx in cv.split(X.values, y):
        X_tr, X_va = X.values[train_idx], X.values[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_va)
        probs = rf.predict_proba(X_va)[:, 1]

        metrics['auc'].append(roc_auc_score(y_va, probs))
        metrics['acc'].append(accuracy_score(y_va, preds))
        metrics['f1'].append(f1_score(y_va, preds))
        metrics['sens'].append(recall_score(y_va, preds))

        tn, fp, fn, tp = confusion_matrix(y_va, preds).ravel()
        metrics['spec'].append(tn / (tn + fp))

    print(f"[INFO] Cohort '{dataset_name}' | ROC-AUC: {np.mean(metrics['auc']):.3f} ± {np.std(metrics['auc']):.3f} | F1: {np.mean(metrics['f1']):.3f}")

    res_df = pd.DataFrame([{
        'Dataset': dataset_name,
        'ROC_AUC_Mean': np.mean(metrics['auc']),
        'ROC_AUC_Std': np.std(metrics['auc']),
        'Accuracy': np.mean(metrics['acc']),
        'F1_Score': np.mean(metrics['f1']),
        'Sensitivity': np.mean(metrics['sens']),
        'Specificity': np.mean(metrics['spec'])
    }])

    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    res_df.to_csv(metrics_out, mode='a' if os.path.exists(metrics_out) else 'w', header=not os.path.exists(metrics_out), index=False)

    rf.fit(X.values, y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump((rf, X.columns.tolist(), X), model_out)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ML model on omics data.")
    parser.add_argument("--input", required=True, help="Path to CLR-transformed feature CSV")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--model-out", required=True, help="Path to save trained model .pkl")
    parser.add_argument("--metrics-out", required=True, help="Path to save metrics summary CSV")
    parser.add_argument("--dataset-name", required=True, help="Name of the dataset/cohort")
    parser.add_argument("--target-col", default="study_condition", help="Metadata column name for phenotype")
    parser.add_argument("--control-val", default="control", help="Value representing healthy control group")
    args = parser.parse_args()

    train_and_evaluate(
        args.input,
        args.metadata,
        args.model_out,
        args.metrics_out,
        args.dataset_name,
        args.target_col,
        args.control_val
    )

if __name__ == "__main__":
    main()
