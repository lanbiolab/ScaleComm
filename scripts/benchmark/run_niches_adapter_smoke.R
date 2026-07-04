#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(NICHES)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: run_niches_adapter_smoke.R <input_dir> <output_dir>")
}

input_dir <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expr <- read.csv(file.path(input_dir, "expression.csv"), check.names = FALSE)
cells <- read.csv(file.path(input_dir, "cells.csv"), check.names = FALSE)
lr <- read.csv(file.path(input_dir, "lr_pairs.csv"), check.names = FALSE)
candidates <- read.csv(file.path(input_dir, "candidate_interactions.csv"), check.names = FALSE)

rownames(cells) <- cells$cell_id
mat <- as.matrix(expr[, setdiff(colnames(expr), "cell_id"), drop = FALSE])
rownames(mat) <- expr$cell_id
mat <- t(mat)
mat <- Matrix(mat, sparse = TRUE)

obj <- CreateSeuratObject(counts = mat, assay = "RNA", meta.data = cells[colnames(mat), , drop = FALSE])
obj <- NormalizeData(obj, verbose = FALSE)
Idents(obj) <- obj$cell_type

custom_lr <- unique(lr[, c("ligand", "receptor")])

log_path <- file.path(output_dir, "niches_smoke.log")
sink(log_path, split = TRUE)
msg_con <- file(log_path, open = "at")
sink(msg_con, type = "message")
on.exit({
  sink(type = "message")
  close(msg_con)
  sink()
}, add = TRUE)

cat("NICHES adapter smoke\n")
cat("cells:", ncol(obj), "genes:", nrow(obj), "LR:", nrow(custom_lr), "\n")

status <- data.frame(
  method = "NICHES",
  status = "started",
  input_ok = TRUE,
  run_ok = FALSE,
  output_level = NA_character_,
  comparable_candidate_edge_ranking = FALSE,
  reason = NA_character_,
  output_path = NA_character_,
  log_path = log_path,
  stringsAsFactors = FALSE
)

tryCatch({
  res <- RunNICHES(
    object = obj,
    assay = "RNA",
    LR.database = "custom",
    species = "human",
    custom_LR_database = custom_lr,
    position.x = "x",
    position.y = "y",
    k = min(79, ncol(obj) - 1),
    CellToCell = FALSE,
    CellToCellSpatial = TRUE,
    CellToSystem = FALSE,
    SystemToCell = FALSE,
    output_format = "raw"
  )

  spatial <- NULL
  for (item in res) {
    if (is.list(item) && "CellToCellSpatialMatrix" %in% names(item)) spatial <- item
  }
  if (is.null(spatial)) stop("RunNICHES finished but no CellToCellSpatialMatrix was returned")

  score_mat <- as.matrix(spatial$CellToCellSpatialMatrix)
  meta <- spatial$metadata
  write.csv(score_mat, file.path(output_dir, "niches_cell_to_cell_spatial_matrix.csv"))
  write.csv(meta, file.path(output_dir, "niches_cell_to_cell_spatial_metadata.csv"))

  parts <- strsplit(colnames(score_mat), "\u2014", fixed = TRUE)
  edge_df <- data.frame(
    sender_id = vapply(parts, `[`, character(1), 1),
    receiver_id = vapply(parts, `[`, character(1), 2),
    stringsAsFactors = FALSE
  )
  lr_parts <- strsplit(rownames(score_mat), "\u2014", fixed = TRUE)
  lr_df <- data.frame(
    ligand = vapply(lr_parts, `[`, character(1), 1),
    receptor = vapply(lr_parts, `[`, character(1), 2),
    stringsAsFactors = FALSE
  )

  long_scores <- do.call(rbind, lapply(seq_len(nrow(score_mat)), function(i) {
    data.frame(
      ligand = lr_df$ligand[[i]],
      receptor = lr_df$receptor[[i]],
      sender_id = edge_df$sender_id,
      receiver_id = edge_df$receiver_id,
      niches_score = as.numeric(score_mat[i, ]),
      stringsAsFactors = FALSE
    )
  }))

  mapped <- merge(
    candidates[, c("candidate_id", "sender_id", "receiver_id", "ligand", "receptor", "ground_truth",
                   "mechanism", "wrong_range_false_positive", "niche_rule_satisfied")],
    long_scores,
    by = c("sender_id", "receiver_id", "ligand", "receptor"),
    all.x = TRUE
  )
  mapped <- mapped[match(candidates$candidate_id, mapped$candidate_id), ]
  out_csv <- file.path(output_dir, "niches_candidate_scores.csv")
  write.csv(mapped, out_csv, row.names = FALSE)

  coverage <- mean(!is.na(mapped$niches_score))
  cat("candidate coverage:", coverage, "\n")
  status$status <- ifelse(coverage > 0.8, "ran_edge_scores_high_coverage", "ran_edge_scores_partial_coverage")
  status$run_ok <- TRUE
  status$output_level <- "candidate_edge_scores_projected_from_CellToCellSpatial"
  status$comparable_candidate_edge_ranking <- coverage > 0.8
  status$reason <- paste0("coverage=", signif(coverage, 4), "; official RunNICHES CellToCellSpatial with custom LR")
  status$output_path <- out_csv
}, error = function(e) {
  status$status <- "failed"
  status$reason <- paste(class(e)[1], conditionMessage(e), sep = ": ")
  cat("ERROR:", status$reason, "\n")
})

write.csv(status, file.path(output_dir, "niches_smoke_status.csv"), row.names = FALSE)
