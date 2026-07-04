#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: extract_cellchatdb_pathway_genes.R <CellChatDB.human.rda> <output.csv>")
}

rda_path <- args[[1]]
output_path <- args[[2]]

load(rda_path)
db <- CellChatDB.human
interaction <- db$interaction
complex <- db$complex
cofactor <- db$cofactor

expand_name <- function(x) {
  x <- as.character(x)
  if (is.na(x) || x == "") {
    return(character())
  }
  if (x %in% rownames(complex)) {
    vals <- as.character(unlist(complex[x, , drop = FALSE]))
    return(vals[!is.na(vals) & vals != ""])
  }
  if (x %in% rownames(cofactor)) {
    vals <- as.character(unlist(cofactor[x, , drop = FALSE]))
    return(vals[!is.na(vals) & vals != ""])
  }
  parts <- unlist(strsplit(x, "_"))
  parts <- parts[!is.na(parts) & parts != ""]
  return(parts)
}

rows <- list()
for (pathway in sort(unique(interaction$pathway_name))) {
  sub <- interaction[interaction$pathway_name == pathway, , drop = FALSE]
  genes <- character()
  for (i in seq_len(nrow(sub))) {
    genes <- c(genes, expand_name(sub$ligand[i]), expand_name(sub$receptor[i]))
    for (col in intersect(c("agonist", "antagonist", "co_A_receptor", "co_I_receptor"), colnames(sub))) {
      genes <- c(genes, expand_name(sub[[col]][i]))
    }
  }
  genes <- sort(unique(toupper(genes[genes != ""])))
  rows[[length(rows) + 1]] <- data.frame(
    pathway = pathway,
    n_genes = length(genes),
    genes = paste(genes, collapse = ";"),
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
write.csv(out, output_path, row.names = FALSE)
cat("Wrote", nrow(out), "CellChatDB pathway gene sets to", output_path, "\n")
