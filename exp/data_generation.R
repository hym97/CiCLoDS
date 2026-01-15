if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("spatialLIBD")
BiocManager::install("zellkonverter", force=TRUE) # For saving to H5AD

library(spatialLIBD)
# 1. Fetch the specific sample (151673 is the most common benchmark)
sce <- fetch_data(type = "sce", eh = ExperimentHub::ExperimentHub())
# List of representative slices (One from each donor)
sample_ids <- c("151673", "151507", "151669")

for (id in sample_ids) {
  print(paste("Processing:", id))

  # 1. Select the slice
  sce_current <- sce[, sce$sample_name == id]

  # 2. Export Counts (Transposed)
  counts <- t(as.matrix(assays(sce_current)$counts))
  write.csv(counts, file = paste0(id, "_counts.csv"))

  # 3. Export Metadata
  meta <- as.data.frame(colData(sce_current))
  write.csv(meta, file = paste0(id, "_metadata.csv"))
}




