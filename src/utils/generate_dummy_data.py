import pandas as pd
import numpy as np
import os

def generate_toy_microbiome(samples=10, taxa=50, output_path="data/raw/dummy_microbiome.csv"):
    np.random.seed(42)
    raw_counts = np.random.lognormal(mean=2, sigma=1.5, size=(samples, taxa))
    rel_abundances = raw_counts / raw_counts.sum(axis=1, keepdims=True)
    
    sample_names = [f"Patient_{i:03d}" for i in range(1, samples+1)]
    taxa_names = [f"Taxon_s_{i}" for i in range(1, taxa+1)]
    
    df = pd.DataFrame(rel_abundances, index=sample_names, columns=taxa_names)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Generato Toy Dataset (Microbioma) in {output_path}")

if __name__ == "__main__":
    generate_toy_microbiome()
