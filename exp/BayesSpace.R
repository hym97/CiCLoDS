suppressPackageStartupMessages({
  library(BayesSpace)
  library(spatialLIBD)
  library(ExperimentHub)
  library(SingleCellExperiment)
  library(scran)
  library(scater)
  library(ggplot2)
  library(mclust)
})

## ----------------------------
## Helpers
## ----------------------------
ensure_visium_fields <- function(sce_sub) {
  cd <- colData(sce_sub)
  cn <- colnames(cd)
  
  # spot.idx
  if (!("spot.idx" %in% cn)) cd$spot.idx <- seq_len(ncol(sce_sub))
  
  # array_row / array_col
  if (!all(c("array_row", "array_col") %in% cn)) {
    if (all(c("row", "col") %in% cn)) {
      cd$array_row <- cd$row
      cd$array_col <- cd$col
    } else if (all(c("pxl_row_in_fullres", "pxl_col_in_fullres") %in% cn)) {
      cd$array_row <- cd$pxl_row_in_fullres
      cd$array_col <- cd$pxl_col_in_fullres
      message("WARNING: using pxl_*_in_fullres as array coords (fallback)")
    } else {
      stop("Missing array_row/array_col (or row/col). Available colData:\n",
           paste(cn, collapse = ", "))
    }
  }
  
  # ensure row/col exist too (harmless)
  cn <- colnames(cd)
  if (!all(c("row", "col") %in% cn)) {
    cd$row <- cd$array_row
    cd$col <- cd$array_col
  }
  
  # type hygiene
  cd$spot.idx  <- as.integer(cd$spot.idx)
  cd$array_row <- as.numeric(cd$array_row)
  cd$array_col <- as.numeric(cd$array_col)
  cd$row       <- as.numeric(cd$row)
  cd$col       <- as.numeric(cd$col)
  
  colData(sce_sub) <- cd
  sce_sub
}

get_truth <- function(sce_sub) {
  layer_candidates <- c("layer_guess_reordered", "layer_guess", "layer")
  layer_col <- layer_candidates[layer_candidates %in% colnames(colData(sce_sub))][1]
  truth <- NULL
  if (length(layer_col) == 1 && !is.na(layer_col)) {
    truth <- as.character(colData(sce_sub)[[layer_col]])
  }
  list(truth = truth, layer_col = layer_col)
}

run_bayesspace_one <- function(
    sce,
    sample_name,
    out_dir = ".",
    q = 7,
    n_hvg = 2000,
    n_pcs = 15,
    d = 15,
    nrep = 50000,
    gamma = 3,
    seed_base = 100
) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  prefix <- file.path(out_dir, paste0("BayesSpace_", sample_name))
  
  cat("\n==============================\n")
  cat("Running sample:", sample_name, "\n")
  cat("==============================\n")
  
  ## subset sample
  sce_sub <- sce[, sce$sample_name == sample_name]
  if (ncol(sce_sub) == 0) stop("No spots found for sample_name = ", sample_name)
  
  cat(sprintf("Loaded %s: %d genes x %d spots\n", sample_name, nrow(sce_sub), ncol(sce_sub)))
  
  ## ensure required fields
  sce_sub <- ensure_visium_fields(sce_sub)
  
  ## truth
  gt <- get_truth(sce_sub)
  truth <- gt$truth
  layer_col <- gt$layer_col
  if (!is.null(truth)) cat("Truth column:", layer_col, "\n")
  
  ## HVG + PCA
  set.seed(seed_base + 1)
  dec <- scran::modelGeneVar(sce_sub)
  hvg_genes <- scran::getTopHVGs(dec, n = n_hvg)
  
  set.seed(seed_base + 2)
  sce_sub <- scater::runPCA(sce_sub, subset_row = hvg_genes, ncomponents = n_pcs)
  
  ## BayesSpace preprocess
  sce_sub <- BayesSpace::spatialPreprocess(sce_sub, platform = "Visium", skip.PCA = TRUE)
  
  ## re-ensure visium fields (some steps may drop/alter colData)
  sce_sub <- ensure_visium_fields(sce_sub)
  
  ## spatialCluster
  set.seed(seed_base + 4)
  sce_sub <- BayesSpace::spatialCluster(
    sce_sub,
    q = q, d = d, platform = "Visium",
    nrep = nrep, gamma = gamma, save.chain = TRUE
  )
  
  pred <- as.integer(colData(sce_sub)$spatial.cluster)
  cat("Clusters found:", length(unique(pred)), "\n")
  
  ## plot (simple scatter)
  df <- data.frame(
    x = colData(sce_sub)$array_col,
    y = colData(sce_sub)$array_row,
    cluster = factor(colData(sce_sub)$spatial.cluster)
  )
  
  p <- ggplot(df, aes(x = x, y = y, color = cluster)) +
    geom_point(size = 0.4) +
    coord_equal() +
    scale_y_reverse() +
    theme_void() +
    ggtitle(paste0("BayesSpace clusters (DLPFC ", sample_name, ")"))
  
  ggsave(paste0(prefix, "_cluster_scatter.png"), p, width = 6, height = 6, dpi = 200)
  
  ## ARI
  ari <- NA_real_
  if (!is.null(truth)) {
    keep <- !(is.na(truth) | truth %in% c("NA", "", "0", "Unknown", "unknown", "None"))
    if (sum(keep) > 1) {
      ari <- mclust::adjustedRandIndex(pred[keep], truth[keep])
      cat(sprintf("ARI vs %s: %.4f\n", layer_col, ari))
    }
  }
  writeLines(sprintf("ARI\t%0.6f", ari), paste0(prefix, "_ARI.txt"))
  
  ## save allocation + genes + rds
  allocation_df <- data.frame(
    spot = colnames(sce_sub),
    spot.idx = colData(sce_sub)$spot.idx,
    array_row = colData(sce_sub)$array_row,
    array_col = colData(sce_sub)$array_col,
    spatial.cluster = colData(sce_sub)$spatial.cluster,
    stringsAsFactors = FALSE
  )
  if (!is.null(truth)) allocation_df[[layer_col]] <- truth
  
  write.csv(allocation_df, paste0(prefix, "_allocation.csv"), row.names = FALSE)
  writeLines(hvg_genes, paste0(prefix, "_HVGs.txt"))
  saveRDS(sce_sub, paste0(prefix, "_sce_clustered.rds"))
  
  ## return summary row
  data.frame(
    sample_name = sample_name,
    n_spots = ncol(sce_sub),
    n_genes = nrow(sce_sub),
    q = q,
    d = d,
    ari = ari,
    truth_col = if (length(layer_col) == 1) layer_col else NA_character_,
    stringsAsFactors = FALSE
  )
}

## ----------------------------
## Load SCE once
## ----------------------------
cat("\n--- Loading spatialLIBD SCE once ---\n")
eh <- ExperimentHub::ExperimentHub()
sce <- spatialLIBD::fetch_data(type = "sce", eh = eh)

## ----------------------------
## Choose samples to run
## ----------------------------
samples_to_run <- c( "151669","151673","151507")  # <-- edit this list

## ----------------------------
## Run all
## ----------------------------
out_dir <- "BayesSpace_runs"
results <- do.call(rbind, lapply(samples_to_run, function(sid) {
  run_bayesspace_one(
    sce = sce,
    sample_name = sid,
    out_dir = out_dir,
    q = 5,
    n_hvg = 2000,
    n_pcs = 15,
    d = 15,
    nrep = 50000,
    gamma = 3,
    seed_base = 1000 + as.integer(sid)  # different seed per sample
  )
}))

print(results)
write.csv(results, file.path(out_dir, "summary_results_151669.csv"), row.names = FALSE)
cat("\nDONE. Summary written to:", file.path(out_dir, "summary_results.csv"), "\n")
