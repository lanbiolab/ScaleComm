#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SpaTalk)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: run_spatalk_replicate.R <cells.csv> <expression.csv> <lr_pairs.csv> <candidates.csv> <output.csv> [per_num] [min_pairs] [use_synthetic_pathways]")
}

cells_path <- args[[1]]
expression_path <- args[[2]]
lr_pairs_path <- args[[3]]
candidates_path <- args[[4]]
output_path <- args[[5]]
per_num <- ifelse(length(args) >= 6, as.integer(args[[6]]), 100L)
min_pairs <- ifelse(length(args) >= 7, as.integer(args[[7]]), 1L)
use_synthetic_pathways <- ifelse(length(args) >= 8, as.logical(args[[8]]), FALSE)

cells <- read.csv(cells_path, check.names = FALSE, stringsAsFactors = FALSE)
expr <- read.csv(expression_path, check.names = FALSE, stringsAsFactors = FALSE)
lr_pairs <- read.csv(lr_pairs_path, check.names = FALSE, stringsAsFactors = FALSE)
candidates <- read.csv(candidates_path, check.names = FALSE, stringsAsFactors = FALSE)

cell_ids <- as.character(cells$cell_id)
expr_mat <- as.matrix(expr[, setdiff(colnames(expr), "cell_id"), drop = FALSE])
rownames(expr_mat) <- as.character(expr$cell_id)
expr_mat <- expr_mat[cell_ids, , drop = FALSE]
storage.mode(expr_mat) <- "numeric"
st_data <- t(expr_mat)
colnames(st_data) <- cell_ids

celltype <- gsub("[^A-Za-z0-9_]+", "_", as.character(cells$cell_type))
st_meta <- data.frame(
  cell = cell_ids,
  x = as.numeric(cells$x),
  y = as.numeric(cells$y),
  stringsAsFactors = FALSE
)

lr_custom <- unique(data.frame(
  ligand = toupper(as.character(lr_pairs$ligand)),
  receptor = toupper(as.character(lr_pairs$receptor)),
  species = "Human",
  stringsAsFactors = FALSE
))

data(list = c("pathways"), package = "SpaTalk")
pathways_use <- pathways[pathways$species == "Human", , drop = FALSE]

if (use_synthetic_pathways) {
  observed_genes <- rownames(st_data)
  fallback_targets <- setdiff(observed_genes, unique(c(lr_custom$ligand, lr_custom$receptor)))
  if (length(fallback_targets) == 0) {
    fallback_targets <- observed_genes
  }
  synthetic <- data.frame()
  for (i in seq_len(nrow(lr_custom))) {
    receptor <- lr_custom$receptor[i]
    if (!(receptor %in% observed_genes)) {
      next
    }
    target <- fallback_targets[((i - 1) %% length(fallback_targets)) + 1]
    synthetic <- rbind(
      synthetic,
      data.frame(
        src = receptor,
        dest = target,
        pathway = paste0("SpatialSelfSynthetic_", receptor),
        source = "SpatialSelf_Phase1_Benchmark",
        type = "Process(synthetic)",
        src_tf = "NO",
        dest_tf = "YES",
        species = "Human",
        stringsAsFactors = FALSE
      )
    )
  }
  pathways_use <- unique(rbind(pathways_use, synthetic))
}

obj <- createSpaTalk(
  st_data = st_data,
  st_meta = st_meta,
  species = "Human",
  if_st_is_sc = TRUE,
  spot_max_cell = 1,
  celltype = celltype
)

empty_out <- function(reason) {
  out <- data.frame(
    candidate_id = candidates$candidate_id,
    method = "spatalk",
    score = 0,
    spatalk_status = reason,
    stringsAsFactors = FALSE
  )
  write.csv(out, output_path, row.names = FALSE)
}

status <- "completed"
tryCatch({
  obj <- find_lr_path(
    object = obj,
    lrpairs = lr_custom,
    pathways = pathways_use,
    if_doParallel = FALSE
  )
  obj <- dec_cci_all(
    obj,
    n_neighbor = 10,
    min_pairs = min_pairs,
    per_num = per_num,
    pvalue = 1,
    co_exp_ratio = 0,
    if_doParallel = FALSE
  )
}, error = function(e) {
  status <<- paste0("no_spatalk_scores: ", conditionMessage(e))
})

if (status != "completed" || is.null(obj@lrpair) || nrow(obj@lrpair) == 0) {
  empty_out(status)
  quit(save = "no", status = 0)
}

score_lookup <- obj@lrpair
score_lookup$source <- gsub("[^A-Za-z0-9_]+", "_", as.character(score_lookup$celltype_sender))
score_lookup$target <- gsub("[^A-Za-z0-9_]+", "_", as.character(score_lookup$celltype_receiver))
score_lookup$ligand_key <- toupper(as.character(score_lookup$ligand))
score_lookup$receptor_key <- toupper(as.character(score_lookup$receptor))
score_lookup <- score_lookup[, c("source", "target", "ligand_key", "receptor_key", "score"), drop = FALSE]
score_lookup <- aggregate(
  score ~ source + target + ligand_key + receptor_key,
  data = score_lookup,
  FUN = max
)

candidates$sender_key <- gsub("[^A-Za-z0-9_]+", "_", as.character(candidates$sender_cell_type))
candidates$receiver_key <- gsub("[^A-Za-z0-9_]+", "_", as.character(candidates$receiver_cell_type))
candidates$ligand_key <- toupper(as.character(candidates$ligand))
candidates$receptor_key <- toupper(as.character(candidates$receptor))

merged <- merge(
  candidates,
  score_lookup,
  by.x = c("sender_key", "receiver_key", "ligand_key", "receptor_key"),
  by.y = c("source", "target", "ligand_key", "receptor_key"),
  all.x = TRUE,
  sort = FALSE
)
score <- merged$score
score[is.na(score)] <- 0
out <- data.frame(
  candidate_id = merged$candidate_id,
  method = "spatalk",
  score = score,
  spatalk_status = status,
  stringsAsFactors = FALSE
)
write.csv(out, output_path, row.names = FALSE)
