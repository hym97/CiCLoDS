## ==============================================================================
## AUTOMATED BAYESCAFE PIPELINE (3 SAMPLES)
## Samples: 151673 (Donor 1), 151507 (Donor 2), 151669 (Donor 3)
## ==============================================================================

# --- CONFIGURATION ---
SAMPLES_TO_RUN <- c("151673", "151507", "151669")
DO_CROP        <- FALSE   # Set TRUE for fast test (Left 30%), FALSE for full paper run
BAYESCAFE_DIR  <- "/home/rstudio/BayesCafe"

## ------------------------------------------------------------------------------
## 0. INSTALL + LOAD PACKAGES
## ------------------------------------------------------------------------------
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!require("spatialLIBD", quietly = TRUE)) BiocManager::install("spatialLIBD")
if (!require("ExperimentHub", quietly = TRUE)) BiocManager::install("ExperimentHub")
if (!require("mclust", quietly = TRUE)) install.packages("mclust")

suppressPackageStartupMessages({
  library(spatialLIBD)
  library(ExperimentHub)
  library(mclust)
})

# Load BayesCafe functions once
setwd(BAYESCAFE_DIR)
source("R/BayesCafe.R")

## ------------------------------------------------------------------------------
## MAIN LOOP
## ------------------------------------------------------------------------------
for (sample_id in SAMPLES_TO_RUN) {

  cat(sprintf("\n\n################################################################"))
  cat(sprintf("\n### PROCESSING SAMPLE: %s", sample_id))
  cat(sprintf("\n################################################################\n"))

  ## ----------------------------------------------------------------------------
  ## 1. LOAD DATA
  ## ----------------------------------------------------------------------------
  cat("\n--- 1. Loading Data ---\n")
  eh <- ExperimentHub::ExperimentHub()
  sce <- fetch_data(type = "sce", eh = eh)
  sce_sub <- sce[, sce$sample_name == sample_id]

  ## ----------------------------------------------------------------------------
  ## 2. FORMAT DATA + GROUND TRUTH
  ## ----------------------------------------------------------------------------
  cat("--- 2. Formatting ---\n")

  # Counts: Transpose to (Spots x Genes)
  raw_counts   <- as.matrix(assays(sce_sub)$counts)
  count_matrix <- t(raw_counts)

  # Spatial Coordinates
  cols_avail <- colnames(colData(sce_sub))
  if (all(c("array_row", "array_col") %in% cols_avail)) {
    sp_cols <- c("array_row", "array_col")
  } else {
    sp_cols <- c("row", "col")
  }

  loc_matrix <- as.matrix(colData(sce_sub)[, sp_cols, drop = FALSE])
  colnames(loc_matrix) <- c("x", "y")

  # Ground Truth Layers
  layer_candidates <- c("layer_guess_reordered", "layer_guess", "layer")
  layer_col <- layer_candidates[layer_candidates %in% colnames(colData(sce_sub))][1]
  truth_full <- as.character(colData(sce_sub)[[layer_col]])
  names(truth_full) <- colnames(sce_sub)

  # All genes list
  all_genes <- colnames(count_matrix)

  ## ----------------------------------------------------------------------------
  ## 3. SPATIAL CROP OR FULL
  ## ----------------------------------------------------------------------------
  if (DO_CROP) {
    cat("--- 3. Spatial Crop (Left 30%) ---\n")
    # Vertical strip crop preserves layers better than corner crop
    x_cutoff <- quantile(loc_matrix[, "x"], 0.30)
    valid_idx <- which(loc_matrix[, "x"] < x_cutoff)
  } else {
    cat("--- 3. Using FULL Dataset ---\n")
    valid_idx <- seq_len(nrow(loc_matrix))
  }

  count_active <- count_matrix[valid_idx, , drop = FALSE]
  loc_active   <- loc_matrix[valid_idx, , drop = FALSE]

  spot_ids_active <- rownames(count_active)
  truth_active    <- truth_full[spot_ids_active]

  cat(sprintf("Spots to analyze: %d\n", nrow(count_active)))

  # Cleanup big objects for this iteration
  rm(sce, sce_sub, raw_counts, count_matrix, loc_matrix)
  gc()

  ## ----------------------------------------------------------------------------
  ## 4. BAYESCAFE PREPROCESS
  ## ----------------------------------------------------------------------------
  cat("--- 4. BayesCafe Preprocessing ---\n")

  result <- dataPreprocess(
    count = count_active,
    loc = loc_active,
    cutoff_sample = 0,
    cutoff_feature = 0.1,
    cutoff_max = 0,
    size.factor = "tss",
    platform = "Visium",
    findHVG = TRUE,
    n.HVGs = 2000
  )

  count_final <- result$count
  loc_final   <- result$loc
  s_final     <- result$s
  P_final     <- result$P

  # Track Features
  feat_selected <- colnames(count_final)
  feat_removed  <- setdiff(all_genes, feat_selected)

  # Align Truth to final filtered spots
  truth_final <- truth_active[rownames(count_final)]

  rm(result, count_active, loc_active)
  gc()

  ## ----------------------------------------------------------------------------
  ## 5. RUN MCMC
  ## ----------------------------------------------------------------------------
  cat("--- 5. Running BayesCafe MCMC ---\n")

  # Set global loc for compatibility
  loc <<- loc_final

  res <- bayes_cafe(
    count = count_final,
    K = 7,             # Target 7 Layers
    s = s_final,
    P = P_final,
    iter = 2000,
    burn = 1000
  )

  ## ----------------------------------------------------------------------------
  ## 6. EXTRACT & SCORE
  ## ----------------------------------------------------------------------------
  cat("--- 6. Extracting & Scoring ---\n")

  # Extract Clusters
  if ("cluster" %in% colnames(res$cluster_result)) {
    pred <- res$cluster_result[, "cluster"]
  } else {
    pred <- res$cluster_result[, ncol(res$cluster_result)]
  }
  pred <- as.character(pred)

  # Allocation DataFrame
  alloc_df <- data.frame(
    spot_id = rownames(loc_final),
    x = loc_final[, "x"],
    y = loc_final[, "y"],
    cluster = as.integer(as.factor(pred))
  )

  # Compute ARI
  keep <- !(is.na(truth_final) | truth_final %in% c("NA", "", "0", "None", "unknown"))
  ari <- mclust::adjustedRandIndex(pred[keep], truth_final[keep])

  cat(sprintf("Sample %s ARI: %.4f\n", sample_id, ari))

  ## ----------------------------------------------------------------------------
  ## 7. SAVE OUTPUTS
  ## ----------------------------------------------------------------------------
  cat("--- 7. Saving Outputs ---\n")

  prefix <- if(DO_CROP) paste0(sample_id, "_crop") else paste0(sample_id, "_full")

  # Save Text/CSV
  write.csv(alloc_df, paste0(prefix, "_allocation.csv"), row.names = FALSE)
  writeLines(sprintf("ARI\t%.6f", ari), paste0(prefix, "_ARI.txt"))
  writeLines(feat_selected, paste0(prefix, "_feat_selected.txt"))

  # Save RDS Bundle
  saveRDS(list(
    allocation = alloc_df,
    ari = ari,
    count = count_final,
    loc = loc_final,
    bayes_res = res
  ), file = paste0(prefix, "_bayescafe_bundle.rds"))

  # Save Plot
  n_clus <- length(unique(alloc_df$cluster))
  png(paste0(prefix, "_plot.png"), width=600, height=600)
  plot.cluster(
    res$cluster_result,
    x = loc_final[, "x"],
    y = loc_final[, "y"],
    cluster = as.factor(alloc_df$cluster),
    colors = rainbow(n_clus)
  )
  dev.off()

  cat(sprintf("Finished Sample %s!\n", sample_id))
}

cat("\n\n### ALL SAMPLES PROCESSED ###\n")