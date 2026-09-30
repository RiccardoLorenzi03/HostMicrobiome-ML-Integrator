import os
import numpy as np
import pandas as pd

def generate_cohort(study_name: str, condition_name: str, onco_taxa: list, commensal_taxa: list, n_controls: int = 80, n_cases: int = 75, seed: int = 42) -> None:
    raw_dir, metadata_dir = "data/raw", "data/metadata"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    np.random.seed(seed)
    n_samples = n_controls + n_cases
    sample_ids = [f"{study_name}_CTRL_{i:03d}" for i in range(1, n_controls + 1)] + \
                 [f"{study_name}_{condition_name}_{i:03d}" for i in range(1, n_cases + 1)]

    all_taxa = onco_taxa + commensal_taxa + [f"s__Gut_Bacterium_{i:02d}" for i in range(1, 35)]
    data = np.random.lognormal(mean=-3, sigma=1.0, size=(n_samples, len(all_taxa)))

    for idx in range(len(onco_taxa)):
        data[n_controls:, idx] *= (3.5 + idx * 0.3)
    for idx in range(len(commensal_taxa)):
        data[:n_controls, len(onco_taxa) + idx] *= 2.5

    rel_abund = data / data.sum(axis=1, keepdims=True)

    pd.DataFrame(rel_abund, index=sample_ids, columns=all_taxa).to_csv(os.path.join(raw_dir, f"{study_name}_microbiome.csv"))
    pd.DataFrame({'study_condition': ['control'] * n_controls + [condition_name] * n_cases, 'cohort': study_name}, index=sample_ids).to_csv(os.path.join(metadata_dir, f"{study_name}_metadata.csv"))
    print(f"[INFO] Cohort '{study_name}' generated ({n_samples} samples).")

def main() -> None:
    generate_cohort("ZellerG_2014", "CRC", ["s__Fusobacterium_nucleatum", "s__Parvimonas_micra", "s__Peptostreptococcus_stomatis"], ["s__Faecalibacterium_prausnitzii", "s__Roseburia_intestinalis"], seed=42)
    generate_cohort("FranzosaEA_2019", "IBD", ["s__Escherichia_coli", "s__Ruminococcus_gnavus", "s__Clostridium_symbiosum"], ["s__Eubacterium_rectale", "s__Bifidobacterium_longum"], seed=101)
    generate_cohort("WirbelJ_2019", "CRC", ["s__Porphyromonas_asaccharolytica", "s__Bacteroides_fragilis", "s__Fusobacterium_nucleatum"], ["s__Blautia_obeum", "s__Coprococcus_comes"], seed=2024)

if __name__ == "__main__":
    main()
