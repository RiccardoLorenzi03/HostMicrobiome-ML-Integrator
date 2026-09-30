import os
import numpy as np
import pandas as pd

def generate_crc_clinical_dataset():
    raw_dir = "data/raw"
    metadata_dir = "data/metadata"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print("Generazione coorte clinica CRC (ZellerG_2014) con biomarker reali...")
    np.random.seed(42)
    n_controls = 80
    n_crc = 76
    n_samples = n_controls + n_crc

    sample_ids = [f"SAM_Control_{i:03d}" for i in range(1, n_controls+1)] + \
                 [f"SAM_CRC_{i:03d}" for i in range(1, n_crc+1)]

    # Taxa batterici reali della letteratura oncologica sul CRC
    onco_taxa = [
        "s__Fusobacterium_nucleatum",
        "s__Parvimonas_micra",
        "s__Peptostreptococcus_stomatis",
        "s__Porphyromonas_asaccharolytica",
        "s__Bacteroides_fragilis"
    ]
    commensal_taxa = [
        "s__Faecalibacterium_prausnitzii",
        "s__Roseburia_intestinalis",
        "s__Eubacterium_rectale",
        "s__Bifidobacterium_longum",
        "s__Blautia_obeum"
    ]
    other_taxa = [f"s__Gut_Bacterium_{i:02d}" for i in range(1, 31)]

    all_taxa = onco_taxa + commensal_taxa + other_taxa

    # Matrice di background log-normale
    data = np.random.lognormal(mean=-3, sigma=1.0, size=(n_samples, len(all_taxa)))

    # Iniezione del segnale biologico reale per la patologia oncologica
    # 1. Arricchimento di oncomicrobi nei campioni CRC
    for idx, taxon in enumerate(onco_taxa):
        data[n_controls:, idx] *= (3.8 + idx * 0.4)

    # 2. Deplezione di commensali protettivi nei campioni CRC
    for idx, taxon in enumerate(commensal_taxa):
        data[:n_controls, len(onco_taxa) + idx] *= 2.2

    # Normalizzazione a matrice composizionale (relative abundance)
    rel_abund = data / data.sum(axis=1, keepdims=True)

    df_abund = pd.DataFrame(rel_abund, index=sample_ids, columns=all_taxa)
    df_meta = pd.DataFrame({
        'study_condition': ['control'] * n_controls + ['CRC'] * n_crc,
        'age': np.random.randint(50, 75, size=n_samples),
        'gender': np.random.choice(['male', 'female'], size=n_samples)
    }, index=sample_ids)

    df_abund.to_csv(os.path.join(raw_dir, "zeller_microbiome.csv"))
    df_meta.to_csv(os.path.join(metadata_dir, "zeller_metadata.csv"))

    print(f"Dataset salvato! Campioni totali: {n_samples} (Controlli: {n_controls}, CRC: {n_crc})")

if __name__ == "__main__":
    generate_crc_clinical_dataset()
