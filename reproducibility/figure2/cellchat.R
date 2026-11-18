#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(CellChat)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: run_cellchat_replicate.R <cells.csv> <expression.csv> <lr_pairs.csv> <candidates.csv> <output.csv> [nboot]")
}

cells_path <- args[[1]]
expression_path <- args[[2]]
lr_pairs_path <- args[[3]]
candidates_path <- args[[4]]
output_path <- args[[5]]
nboot <- ifelse(length(args) >= 6, as.integer(args[[6]]), 20L)

cells <- read.csv(cells_path, check.names = FALSE, stringsAsFactors = FALSE)
expr <- read.csv(expression_path, check.names = FALSE, stringsAsFactors = FALSE)
lr_pairs <- read.csv(lr_pairs_path, check.names = FALSE, stringsAsFactors = FALSE)
candidates <- read.csv(candidates_path, check.names = FALSE, stringsAsFactors = FALSE)

cell_ids <- as.character(cells$cell_id)
rownames(cells) <- cell_ids
expr_mat <- as.matrix(expr[, setdiff(colnames(expr), "cell_id"), drop = FALSE])
rownames(expr_mat) <- as.character(expr$cell_id)
expr_mat <- expr_mat[cell_ids, , drop = FALSE]
storage.mode(expr_mat) <- "numeric"
data_input <- t(log1p(expr_mat))

labels <- if ("cell_type" %in% colnames(cells)) cells$cell_type else rep("unknown", nrow(cells))
labels <- gsub("[^A-Za-z0-9_]+", "_", as.character(labels))
meta <- data.frame(labels = labels, samples = "sample1", row.names = cell_ids, stringsAsFactors = FALSE)
coordinates <- as.matrix(cells[, c("x", "y"), drop = FALSE])
rownames(coordinates) <- cell_ids

data(CellChatDB.human)
template <- CellChatDB.human$interaction[0, , drop = FALSE]
db_interaction <- template
for (i in seq_len(nrow(lr_pairs))) {
  ligand <- toupper(as.character(lr_pairs$ligand[i]))
  receptor <- toupper(as.character(lr_pairs$receptor[i]))
  interaction_name <- paste(ligand, receptor, sep = "_")
  row <- template[1, , drop = FALSE]
  row[1, ] <- NA
  row$interaction_name <- interaction_name
  row$pathway_name <- interaction_name
  row$ligand <- ligand
  row$receptor <- receptor
  row$agonist <- ""
  row$antagonist <- ""
  row$co_A_receptor <- ""
  row$co_I_receptor <- ""
  mechanism <- if ("mechanism" %in% colnames(lr_pairs)) as.character(lr_pairs$mechanism[i]) else ""
  row$annotation <- ifelse(grepl("contact|juxtacrine", mechanism), "Cell-Cell Contact", "Secreted Signaling")
  row$interaction_name_2 <- paste(ligand, "-", receptor)
  row$evidence <- "SpatialSelf_Phase1_Benchmark"
  row$is_neurotransmitter <- FALSE
  row$version <- "SpatialSelf_Phase1"
  rownames(row) <- interaction_name
  db_interaction <- rbind(db_interaction, row)
}

CellChatDB.use <- CellChatDB.human
CellChatDB.use$interaction <- db_interaction

spatial_factors <- data.frame(ratio = 1, tol = 0.001)
cellchat <- createCellChat(
  object = data_input,
  meta = meta,
  group.by = "labels",
  datatype = "spatial",
  coordinates = coordinates,
  spatial.factors = spatial_factors
)
cellchat@DB <- CellChatDB.use
cellchat <- subsetData(cellchat)

lr_use <- CellChatDB.use$interaction
common_genes <- rownames(cellchat@data.signaling)
lr_use <- lr_use[lr_use$ligand %in% common_genes & lr_use$receptor %in% common_genes, , drop = FALSE]
if (nrow(lr_use) == 0) {
  out <- data.frame(candidate_id = candidates$candidate_id, method = "cellchat", score = 0)
  write.csv(out, output_path, row.names = FALSE)
  quit(save = "no", status = 0)
}

cellchat <- computeCommunProb(
  cellchat,
  type = "truncatedMean",
  trim = 0.1,
  LR.use = lr_use,
  raw.use = TRUE,
  distance.use = FALSE,
  interaction.range = max(candidates$distance, na.rm = TRUE),
  contact.dependent = TRUE,
  contact.range = quantile(candidates$distance, 0.05, na.rm = TRUE),
  nboot = nboot,
  seed.use = 2025,
  k.min = 1
)

prob <- cellchat@net$prob
if (is.null(prob) || length(prob) == 0) {
  score_lookup <- data.frame()
} else {
  net <- as.data.frame.table(prob, responseName = "prob", stringsAsFactors = FALSE)
  colnames(net)[1:3] <- c("source", "target", "interaction_name")
  net <- net[net$prob > 0, , drop = FALSE]
  lr_map <- lr_use[, c("interaction_name", "ligand", "receptor"), drop = FALSE]
  score_lookup <- merge(net, lr_map, by = "interaction_name", all.x = TRUE, sort = FALSE)
  score_lookup <- score_lookup[, c("source", "target", "ligand", "receptor", "prob"), drop = FALSE]
  score_lookup$ligand <- toupper(score_lookup$ligand)
  score_lookup$receptor <- toupper(score_lookup$receptor)
}

candidates$sender_key <- gsub("[^A-Za-z0-9_]+", "_", as.character(candidates$sender_cell_type))
candidates$receiver_key <- gsub("[^A-Za-z0-9_]+", "_", as.character(candidates$receiver_cell_type))
candidates$ligand_key <- toupper(as.character(candidates$ligand))
candidates$receptor_key <- toupper(as.character(candidates$receptor))

if (nrow(score_lookup) > 0) {
  merged <- merge(
    candidates,
    score_lookup,
    by.x = c("sender_key", "receiver_key", "ligand_key", "receptor_key"),
    by.y = c("source", "target", "ligand", "receptor"),
    all.x = TRUE,
    sort = FALSE
  )
  score <- merged$prob
  score[is.na(score)] <- 0
  out <- data.frame(candidate_id = merged$candidate_id, method = "cellchat", score = score)
} else {
  out <- data.frame(candidate_id = candidates$candidate_id, method = "cellchat", score = 0)
}

write.csv(out, output_path, row.names = FALSE)
