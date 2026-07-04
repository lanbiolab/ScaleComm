#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(nichenetr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: extract_nichenet_targets.R <lr_pairs.csv> <ligand_target_matrix.rds> <output.csv> [top_n]")
}

lr_pairs_path <- args[[1]]
ligand_target_matrix_path <- args[[2]]
output_path <- args[[3]]
top_n <- ifelse(length(args) >= 4, as.integer(args[[4]]), 200L)

lr_pairs <- read.csv(lr_pairs_path, check.names = FALSE, stringsAsFactors = FALSE)
ligand_target_matrix <- readRDS(ligand_target_matrix_path)

rows <- list()
for (i in seq_len(nrow(lr_pairs))) {
  ligand <- as.character(lr_pairs$ligand[i])
  receptor <- as.character(lr_pairs$receptor[i])
  lr_id <- if ("lr_id" %in% colnames(lr_pairs)) as.character(lr_pairs$lr_id[i]) else paste(ligand, receptor, sep = "_")
  mechanism <- if ("mechanism" %in% colnames(lr_pairs)) as.character(lr_pairs$mechanism[i]) else ""

  status <- "targets_found"
  targets <- character()
  weights <- numeric()
  ligand_key <- ligand
  if (!(ligand_key %in% colnames(ligand_target_matrix))) {
    ligand_key <- toupper(ligand)
  }
  if (!(ligand_key %in% colnames(ligand_target_matrix))) {
    ligand_key <- tools::toTitleCase(tolower(ligand))
  }

  if (ligand_key %in% colnames(ligand_target_matrix)) {
    values <- ligand_target_matrix[, ligand_key]
    values <- sort(values[values > 0], decreasing = TRUE)
    if (length(values) > 0) {
      values <- head(values, top_n)
      targets <- names(values)
      weights <- as.numeric(values)
    } else {
      status <- "ligand_has_no_positive_targets"
    }
  } else {
    status <- "ligand_absent_from_nichenet_matrix"
  }

  if (length(targets) == 0) {
    rows[[length(rows) + 1]] <- data.frame(
      lr_id = lr_id,
      ligand = ligand,
      receptor = receptor,
      mechanism = mechanism,
      target_genes = "",
      target_weights = "",
      nichenet_ligand_key = ligand_key,
      n_targets = 0L,
      status = status,
      stringsAsFactors = FALSE
    )
  } else {
    rows[[length(rows) + 1]] <- data.frame(
      lr_id = lr_id,
      ligand = ligand,
      receptor = receptor,
      mechanism = mechanism,
      target_genes = paste(targets, collapse = ";"),
      target_weights = paste(signif(weights, 6), collapse = ";"),
      nichenet_ligand_key = ligand_key,
      n_targets = length(targets),
      status = status,
      stringsAsFactors = FALSE
    )
  }
}

out <- do.call(rbind, rows)
write.csv(out, output_path, row.names = FALSE)
