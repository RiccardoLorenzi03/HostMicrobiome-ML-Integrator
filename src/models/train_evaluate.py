import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_model(input_path, model_output_path):
    print(f"Caricamento dati trasformati da {input_path}...")
    df = pd.read_csv(input_path, index_col=0)
    
    # Generazione target fittizio binario (es. 0=Sano, 1=Patologia) per testare la pipeline
    np.random.seed(42)
    y = np.random.choice([0, 1], size=len(df))
    X = df.values

    print(f"Addestramento Random Forest su {X.shape[0]} campioni e {X.shape[1]} feature...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    # Stratified K-Fold per contrastare overfitting in regime p >> n
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
    print(f"ROC-AUC medio in Cross-Validation: {scores.mean():.3f} (+/- {scores.std():.3f})")

    rf.fit(X, y)
    
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump((rf, df.columns.tolist()), model_output_path)
    print(f"Modello salvato con successo in {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestra un modello ML su dati omici.")
    parser.add_argument("--input", type=str, required=True, help="Path CSV dati processati.")
    parser.add_argument("--output", type=str, required=True, help="Path salvataggio modello .pkl.")
    args = parser.parse_args()

    train_model(args.input, args.output)
