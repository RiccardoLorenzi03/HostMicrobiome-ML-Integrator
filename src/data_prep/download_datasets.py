from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CohortConfig:
    """Configuration schema for clinical multi-omics cohorts."""
    study_name: str
    condition_name: str
    onco_taxa: List[str]
    commensal_taxa: List[str]
    total_samples: int
    seed: int = 42


COHORTS: List[CohortConfig] = [
    CohortConfig(
        study_name="ZellerG_2014",
        condition_name="CRC",
        onco_taxa=[
            "s__Fusobacterium_nucleatum",
            "s__Parvimonas_micra",
            "s__Peptostreptococcus_stomatis",
        ],
        commensal_taxa=[
            "s__Faecalibacterium_prausnitzii",
            "s__Roseburia_intestinalis",
        ],
        total_samples=156,
        seed=42,
    ),
    CohortConfig(
        study_name="FranzosaEA_2019",
        condition_name="IBD",
        onco_taxa=[
            "s__Escherichia_coli",
            "s__Ruminococcus_gnavus",
            "s__Clostridium_symbiosum",
        ],
        commensal_taxa=[
            "s__Eubacterium_rectale",
            "s__Bifidobacterium_longum",
        ],
        total_samples=220,
        seed=101,
    ),
    CohortConfig(
        study_name="WirbelJ_2019",
        condition_name="CRC",
        onco_taxa=[
            "s__Porphyromonas_asaccharolytica",
            "s__Bacteroides_fragilis",
            "s__Fusobacterium_nucleatum",
        ],
        commensal_taxa=[
            "s__Blautia_obeum",
            "s__Coprococcus_comes",
        ],
        total_samples=252,
        seed=2024,
    ),
]


def generate_cohort_data(config: CohortConfig, raw_dir: Path, meta_dir: Path) -> None:
    """Generates synthetic multi-omics abundance matrices and clinical metadata."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)
    n_controls = config.total_samples // 2
    n_cases = config.total_samples - n_controls

    sample_ids = [
        f"{config.study_name}_CTRL_{i:03d}" for i in range(1, n_controls + 1)
    ] + [
        f"{config.study_name}_{config.condition_name}_{i:03d}"
        for i in range(1, n_cases + 1)
    ]

    background_taxa = [f"s__Gut_Bacterium_{i:02d}" for i in range(1, 35)]
    all_taxa = config.onco_taxa + config.commensal_taxa + background_taxa

    # Generate log-normal background relative abundances
    raw_counts = rng.lognormal(mean=-3.0, sigma=1.0, size=(config.total_samples, len(all_taxa)))

    # Inject phenotype-specific biomass shifts
    for idx in range(len(config.onco_taxa)):
        raw_counts[n_controls:, idx] *= 3.5 + idx * 0.3
    for idx in range(len(config.commensal_taxa)):
        raw_counts[:n_controls, len(config.onco_taxa) + idx] *= 2.5

    # Apply simplex closure constraint (relative abundance sum = 1.0)
    rel_abundance = raw_counts / raw_counts.sum(axis=1, keepdims=True)

    # Format DataFrames
    df_abundance = pd.DataFrame(rel_abundance, index=sample_ids, columns=all_taxa)
    df_metadata = pd.DataFrame(
        {
            "study_condition": ["control"] * n_controls + [config.condition_name] * n_cases,
            "cohort": config.study_name,
        },
        index=sample_ids,
    )

    # Persist to CSV
    df_abundance.to_csv(raw_dir / f"{config.study_name}_microbiome.csv")
    df_metadata.to_csv(meta_dir / f"{config.study_name}_metadata.csv")

    print(f"[INFO] Cohort '{config.study_name}' written: {len(df_abundance)} total samples processed.")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    meta_dir = project_root / "data" / "metadata"

    for config in COHORTS:
        generate_cohort_data(config, raw_dir, meta_dir)


if __name__ == "__main__":
    main()
