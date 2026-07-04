#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(cytosignal)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: run_cytosignal_adapter_smoke.R <input_dir> <output_dir>")
}

input_dir <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

expr <- read.csv(file.path(input_dir, "expression.csv"), check.names = FALSE)
cells <- read.csv(file.path(input_dir, "cells.csv"), check.names = FALSE)
lr <- read.csv(file.path(input_dir, "lr_pairs.csv"), check.names = FALSE)

mat <- as.matrix(expr[, setdiff(colnames(expr), "cell_id"), drop = FALSE])
rownames(mat) <- expr$cell_id
mat <- t(mat)
mat <- Matrix(mat, sparse = TRUE)
cells <- cells[match(colnames(mat), cells$cell_id), ]
loc <- as.matrix(cells[, c("x", "y")])
rownames(loc) <- cells$cell_id
clusters <- factor(cells$cell_type)
names(clusters) <- cells$cell_id

log_path <- file.path(output_dir, "cytosignal_smoke.log")
sink(log_path, split = TRUE)
msg_con <- file(log_path, open = "at")
sink(msg_con, type = "message")
on.exit({
  sink(type = "message")
  close(msg_con)
  sink()
}, add = TRUE)

cat("cytosignal adapter smoke\n")
cat("cells:", ncol(mat), "genes:", nrow(mat), "LR:", nrow(lr), "\n")

status <- data.frame(
  method = "CytoSignal",
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
  object <- createCytoSignal(raw.data = mat, cells.loc = loc, clusters = clusters, name = "lymph_smoke")

  genes <- rownames(mat)
  lr_ids <- paste(lr$ligand, lr$receptor, sep = "-")
  lig.fac <- factor(lr_ids, levels = lr_ids)
  names(lig.fac) <- lr$ligand
  recep.fac <- factor(lr_ids, levels = lr_ids)
  names(recep.fac) <- lr$receptor
  combined.fac <- factor(rep(lr_ids, each = 2), levels = lr_ids)
  names(combined.fac) <- as.vector(rbind(lr$ligand, lr$receptor))

  object@intr.valid <- list(
    symbols = list(intr = genes),
    gene_to_uniprot = data.frame(gene_name = genes, uniprot = genes),
    diff_dep = list(ligands = lig.fac, receptors = recep.fac, combined = combined.fac),
    cont_dep = list(ligands = lig.fac, receptors = recep.fac, combined = combined.fac),
    intr.index = data.frame(id_cp_interaction = lr_ids, partner_a = lr$ligand, partner_b = lr$receptor)
  )

  object <- removeLowQuality(object, counts.thresh = 0, gene.thresh = 0)
  object@counts <- object@raw.counts
  object@parameters[["lib.size"]] <- Matrix::colSums(object@counts)
  object <- inferEpsParams(object, scale.factor = 1, r.eps.real = 0.25)
  object <- findNNDT(object, imp.name = "contact")
  object <- findNNRaw(object)
  object <- inferScoreLR(
    object,
    lig.imp = "Raw",
    recep.imp = "Raw",
    norm.method = "none",
    intr.db.use = "diff_dep",
    lrscore.name = "raw_custom_lr"
  )

  scores <- as.matrix(object@lrscore[["raw_custom_lr"]]@score)
  out_csv <- file.path(output_dir, "cytosignal_cell_lr_scores.csv")
  write.csv(scores, out_csv)
  status$status <- "ran_cell_lr_scores"
  status$run_ok <- TRUE
  status$output_level <- "cell_by_lr_score"
  status$comparable_candidate_edge_ranking <- FALSE
  status$reason <- "CytoSignal returned cell-level LR scores; no native sender-receiver candidate-edge ranking in this smoke adapter"
  status$output_path <- out_csv
}, error = function(e) {
  status$status <- "failed"
  status$reason <- paste(class(e)[1], conditionMessage(e), sep = ": ")
  cat("ERROR:", status$reason, "\n")
})

write.csv(status, file.path(output_dir, "cytosignal_smoke_status.csv"), row.names = FALSE)
