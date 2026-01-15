## ==============================================================================
## Multi-donor CiCLoDS-seeded BayesSpace runs
## donors: 151673, 151669, 151507
## files:
##   /home/rstudio/<donor>_ciclods_selected_genes.txt
##   /home/rstudio/<donor>_ciclods_cluster_allocations.txt
## outputs saved under out_dir with donor-specific prefixes
## ==============================================================================

suppressPackageStartupMessages({
  library(BayesSpace)
  library(spatialLIBD)
  library(ExperimentHub)
  library(SingleCellExperiment)
  library(scran)
  library(scater)
  library(ggplot2)
  library(mclust)
  library(Matrix)
})

## ----------------------------
## Helpers (same as yours, kept minimal)
## ----------------------------
ensure_visium_cols <- function(sce) {
  cd <- colData(sce)
  
  if (!("spot.idx" %in% colnames(cd))) cd$spot.idx <- seq_len(ncol(sce))
  
  if (!all(c("array_row", "array_col") %in% colnames(cd))) {
    if (all(c("row", "col") %in% colnames(cd))) {
      cd$array_row <- cd$row
      cd$array_col <- cd$col
    } else {
      stop("Missing array_row/array_col (or row/col). Available:\n",
           paste(colnames(cd), collapse = ", "))
    }
  }
  
  if (!all(c("row", "col") %in% colnames(cd))) {
    cd$row <- cd$array_row
    cd$col <- cd$array_col
  }
  
  cd$spot.idx  <- as.integer(cd$spot.idx)
  cd$array_row <- as.numeric(cd$array_row)
  cd$array_col <- as.numeric(cd$array_col)
  cd$row       <- as.numeric(cd$row)
  cd$col       <- as.numeric(cd$col)
  
  colData(sce) <- cd
  sce
}

ensure_pixel_cols_for_plot <- function(sce) {
  cd <- colData(sce)
  if (!("pxl_row_in_fullres" %in% colnames(cd))) cd$pxl_row_in_fullres <- as.numeric(cd$array_row)
  if (!("pxl_col_in_fullres" %in% colnames(cd))) cd$pxl_col_in_fullres <- as.numeric(cd$array_col)
  colData(sce) <- cd
  sce
}

get_truth <- function(sce) {
  layer_candidates <- c("layer_guess_reordered", "layer_guess", "layer")
  layer_col <- layer_candidates[layer_candidates %in% colnames(colData(sce))][1]
  truth <- NULL
  if (length(layer_col) == 1 && !is.na(layer_col)) {
    truth <- as.character(colData(sce)[[layer_col]])
  }
  list(truth = truth, layer_col = layer_col)
}

compute_ari <- function(pred, truth) {
  if (is.null(truth)) return(NA_real_)
  keep <- !(is.na(truth) | truth %in% c("NA", "", "0", "Unknown", "unknown", "None"))
  if (sum(keep) <= 1) return(NA_real_)
  mclust::adjustedRandIndex(pred[keep], truth[keep])
}

read_ciclods_init <- function(sce, alloc_path) {
  alloc <- read.delim(alloc_path, header = TRUE, sep = "\t", stringsAsFactors = FALSE)
  
  if (!all(c("cell_id", "ciclods_cluster") %in% colnames(alloc))) {
    if (ncol(alloc) >= 2) {
      colnames(alloc)[1:2] <- c("cell_id", "ciclods_cluster")
    } else {
      stop("Allocation file not 2-column (cell_id, ciclods_cluster): ", alloc_path)
    }
  }
  
  spot_ids <- colnames(sce)
  m <- match(spot_ids, alloc$cell_id)
  
  init_labels <- rep(NA_integer_, length(spot_ids))
  init_labels[!is.na(m)] <- as.integer(alloc$ciclods_cluster[m[!is.na(m)]])
  
  # shift 0..(q-1) to 1..q if needed
  if (min(init_labels, na.rm = TRUE) == 0L) init_labels <- init_labels + 1L
  
  # fill missing with mode
  if (anyNA(init_labels)) {
    tab <- table(init_labels, useNA = "no")
    fill_val <- as.integer(names(tab)[which.max(tab)])
    init_labels[is.na(init_labels)] <- fill_val
  }
  
  q <- length(unique(init_labels))
  list(init_labels = init_labels, q = q)
}

read_ciclods_genes <- function(sce, genes_path) {
  selected <- readLines(genes_path)
  selected <- unique(trimws(selected))
  selected <- selected[selected != ""]
  
  genes_use <- intersect(selected, rownames(sce))
  
  if (length(genes_use) < 50) {
    rd <- rowData(sce)
    candidate_cols <- intersect(c("gene_id", "ensembl", "ensembl_id", "ENSEMBL", "GeneID", "id"), colnames(rd))
    for (cc in candidate_cols) {
      hits <- which(as.character(rd[[cc]]) %in% selected)
      if (length(hits) >= 50) {
        genes_use <- rownames(sce)[hits]
        break
      }
    }
  }
  
  if (length(genes_use) < 10) stop("Too few selected genes matched for: ", genes_path)
  genes_use
}

save_run_outputs <- function(prefix, sce, truth, layer_col, init_labels) {
  pred <- as.integer(colData(sce)$spatial.cluster)
  
  # plot
  sce_plot <- ensure_pixel_cols_for_plot(sce)
  p <- BayesSpace::clusterPlot(sce_plot, label = "spatial.cluster", size = 0.05) +
    ggtitle(prefix)
  ggsave(sprintf("%s_clusterPlot.png", prefix), plot = p, width = 6, height = 6, dpi = 200)
  
  # ARI
  ari <- compute_ari(pred, truth)
  writeLines(sprintf("ARI\t%0.6f", ari), sprintf("%s_ARI.txt", prefix))
  
  # allocation csv
  cd <- colData(sce)
  allocation_df <- data.frame(
    spot = colnames(sce),
    spot.idx = cd$spot.idx,
    array_row = cd$array_row,
    array_col = cd$array_col,
    ciclods_init = init_labels,
    bayesspace_cluster = cd$spatial.cluster,
    stringsAsFactors = FALSE
  )
  if (!is.null(truth) && length(layer_col) == 1 && !is.na(layer_col)) allocation_df[[layer_col]] <- truth
  write.csv(allocation_df, sprintf("%s_allocation.csv", prefix), row.names = FALSE)
  
  # rds
  saveRDS(sce, sprintf("%s_sce_clustered.rds", prefix))
  
  ari
}

## ----------------------------
## Main: load SCE once
## ----------------------------
cat("\n--- Loading spatialLIBD SCE once ---\n")
eh <- ExperimentHub::ExperimentHub()
sce_all <- spatialLIBD::fetch_data(type = "sce", eh = eh)

## ----------------------------
## Settings
## ----------------------------
donors <- c( "151673")
base_dir <- "/home/rstudio"
out_dir <- file.path(base_dir, "BayesSpace_CiCLoDS_runs")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## BayesSpace parameters
q_use <- 7
ncomponents <- 30
d_use <- 15
nrep_use <- 50000
gamma_use <- 3

## ----------------------------
## Loop donors
## ----------------------------
summary_rows <- list()

for (donor in donors) {
  cat("\n============================================================\n")
  cat("DONOR:", donor, "\n")
  cat("============================================================\n")
  
  genes_path <- file.path(base_dir, paste0(donor, "_ciclods_selected_genes.txt"))
  alloc_path <- file.path(base_dir, paste0(donor, "_ciclods_cluster_allocations.txt"))
  
  if (!file.exists(genes_path)) stop("Missing genes file: ", genes_path)
  if (!file.exists(alloc_path)) stop("Missing alloc file: ", alloc_path)
  
  # subset donor
  sce_sub <- sce_all[, as.character(sce_all$sample_name) == donor]
  if (ncol(sce_sub) == 0) stop("No spots found for donor/sample_name = ", donor)
  cat(sprintf("Loaded %s: %d genes x %d spots\n", donor, nrow(sce_sub), ncol(sce_sub)))
  
  sce_sub <- ensure_visium_cols(sce_sub)
  
  # truth
  gt <- get_truth(sce_sub)
  truth <- gt$truth
  layer_col <- gt$layer_col
  if (!is.null(truth)) cat("Truth column:", layer_col, "\n")
  
  # donor-specific init + genes
  init_obj <- read_ciclods_init(sce_sub, alloc_path)
  init_labels <- init_obj$init_labels
  
  genes_use <- read_ciclods_genes(sce_sub, genes_path)
  writeLines(genes_use, file.path(out_dir, paste0(donor, "_CiCLoDS_selected_genes_used.txt")))
  
  # PCA on CiCLoDS genes
  set.seed(1000 + as.integer(donor))
  sce_sub <- scater::runPCA(sce_sub, subset_row = genes_use, ncomponents = ncomponents)
  
  # preprocess + cluster
  sce_sub <- BayesSpace::spatialPreprocess(sce_sub, platform = "Visium", skip.PCA = TRUE)
  sce_sub <- ensure_visium_cols(sce_sub)
  
  set.seed(2000 + as.integer(donor))
  sce_sub <- BayesSpace::spatialCluster(
    sce_sub,
    q = q_use,
    d = d_use,
    platform = "Visium",
    nrep = nrep_use,
    gamma = gamma_use,
    save.chain = TRUE
  )
  
  # save outputs
  prefix <- file.path(out_dir, paste0("CICLODS_", donor))
  ari <- save_run_outputs(prefix, sce_sub, truth, layer_col, init_labels)
  
  summary_rows[[donor]] <- data.frame(
    donor = donor,
    n_spots = ncol(sce_sub),
    n_genes = nrow(sce_sub),
    q = q_use,
    d = d_use,
    ncomponents = ncomponents,
    nrep = nrep_use,
    gamma = gamma_use,
    ari = ari,
    truth_col = if (length(layer_col) == 1) layer_col else NA_character_,
    stringsAsFactors = FALSE
  )
}

summary_df <- do.call(rbind, summary_rows)
print(summary_df)
write.csv(summary_df, file.path(out_dir, "summary.csv"), row.names = FALSE)

cat("\nDONE. Outputs in:", out_dir, "\n")
