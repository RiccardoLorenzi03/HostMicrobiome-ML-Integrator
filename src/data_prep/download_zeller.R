if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

# 2. Installa esplicitamente le dipendenze Bioconductor critiche
bioc_pkgs <- c("DirichletMultinomial", "mia", "curatedMetagenomicData")
for (pkg in bioc_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installazione pacchetto Bioconductor:", pkg))
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  }
}

# 3. Installa le dipendenze CRAN
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr", repos = "https://cloud.r-project.org")
}

library(curatedMetagenomicData)
library(dplyr)

message("1/3 Scaricamento metadati e profili di abbondanza...")
tse <- sampleMetadata %>%
  filter(study_name == "ZellerG_2014") %>%
  returnSamples("relative_abundance", counts = FALSE)

message("2/3 Estrazione matrice di abbondanza e metadati clinici...")
abundance_matrix <- assays(tse)$relative_abundance
metadata <- as.data.frame(colData(tse))

message("3/3 Salvataggio file CSV nella struttura del progetto...")
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/metadata", recursive = TRUE, showWarnings = FALSE)

write.csv(abundance_matrix, "data/raw/zeller_microbiome.csv", row.names = TRUE)
write.csv(metadata, "data/metadata/zeller_metadata.csv", row.names = TRUE)

message("Download completato con successo.")
