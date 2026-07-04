#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(values) {
  out <- list()
  i <- 1
  while (i <= length(values)) {
    key <- values[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected positional argument: %s", key))
    }
    name <- sub("^--", "", key)
    if (i == length(values) || startsWith(values[[i + 1]], "--")) {
      out[[name]] <- TRUE
      i <- i + 1
    } else {
      out[[name]] <- values[[i + 1]]
      i <- i + 2
    }
  }
  out
}

cli <- parse_args(args)
required <- c("simulator", "input-expression", "input-cells", "output-expression", "seed")
missing <- required[!vapply(required, function(x) !is.null(cli[[x]]), logical(1))]
if (length(missing) > 0) {
  stop(sprintf("Missing required arguments: %s", paste(missing, collapse = ", ")))
}

simulator <- tolower(as.character(cli[["simulator"]]))
input_expression <- as.character(cli[["input-expression"]])
input_cells <- as.character(cli[["input-cells"]])
output_expression <- as.character(cli[["output-expression"]])
seed <- as.integer(cli[["seed"]])

expression_df <- read.csv(input_expression, check.names = FALSE, stringsAsFactors = FALSE)
cells_df <- read.csv(input_cells, check.names = FALSE, stringsAsFactors = FALSE)

if (!"cell_id" %in% colnames(expression_df) || !"cell_id" %in% colnames(cells_df)) {
  stop("Both input_expression and input_cells must contain a cell_id column")
}

expression_df$cell_id <- as.character(expression_df$cell_id)
cells_df$cell_id <- as.character(cells_df$cell_id)
expression_df <- expression_df[match(cells_df$cell_id, expression_df$cell_id), , drop = FALSE]
if (any(is.na(expression_df$cell_id))) {
  stop("Expression table does not cover all cell_id values from cells table")
}

gene_names <- setdiff(colnames(expression_df), "cell_id")
count_cell_gene <- as.matrix(expression_df[, gene_names, drop = FALSE])
storage.mode(count_cell_gene) <- "double"
count_cell_gene[is.na(count_cell_gene)] <- 0
count_cell_gene <- pmax(round(count_cell_gene), 0)

simulate_with_scdesign3 <- function() {
  if (!requireNamespace("scDesign3", quietly = TRUE)) {
    stop("Package scDesign3 is not installed")
  }
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Package SingleCellExperiment is not installed")
  }
  counts_gene_cell <- t(count_cell_gene)
  rownames(counts_gene_cell) <- gene_names
  colnames(counts_gene_cell) <- cells_df$cell_id
  col_data <- data.frame(
    cell_type = factor(as.character(cells_df$cell_type)),
    region = factor(as.character(cells_df$region)),
    x = as.numeric(cells_df$x),
    y = as.numeric(cells_df$y),
    row.names = cells_df$cell_id,
    check.names = FALSE
  )
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts_gene_cell),
    colData = col_data
  )
  set.seed(seed)
  sim <- scDesign3::scdesign3(
    sce = sce,
    assay_use = "counts",
    celltype = "cell_type",
    pseudotime = NULL,
    spatial = c("x", "y"),
    other_covariates = "region",
    mu_formula = "cell_type + region + s(x, y, bs = 'gp', k = 20)",
    sigma_formula = "1",
    family_use = "nb",
    n_cores = 2,
    usebam = FALSE,
    corr_formula = "cell_type",
    empirical_quantile = FALSE,
    copula = "gaussian",
    fastmvn = FALSE,
    DT = TRUE,
    pseudo_obs = FALSE,
    family_set = c("gauss", "indep"),
    important_feature = "auto",
    nonnegative = TRUE,
    nonzerovar = TRUE,
    return_model = FALSE,
    simplify = FALSE,
    parallelization = "mcmapply",
    BPPARAM = NULL,
    trace = FALSE
  )
  as.matrix(t(sim$new_count))
}

simulate_with_srtsim <- function() {
  if (!requireNamespace("SRTsim", quietly = TRUE)) {
    stop("Package SRTsim is not installed")
  }
  counts_gene_cell <- t(count_cell_gene)
  rownames(counts_gene_cell) <- gene_names
  colnames(counts_gene_cell) <- cells_df$cell_id
  labels <- as.character(cells_df$region)
  if (length(unique(labels)) <= 1) {
    labels <- as.character(cells_df$cell_type)
  }
  loc_df <- data.frame(
    x = as.numeric(cells_df$x),
    y = as.numeric(cells_df$y),
    label = labels,
    row.names = cells_df$cell_id,
    check.names = FALSE
  )
  set.seed(seed)
  sim_obj <- SRTsim::createSRT(count_in = counts_gene_cell, loc_in = loc_df)
  sim_scheme <- if (length(unique(labels)) > 1) "domain" else "tissue"
  sim_obj <- SRTsim::srtsim_fit(sim_obj, marginal = "nb", sim_scheme = sim_scheme)
  sim_obj <- SRTsim::srtsim_count(sim_obj, verbose = FALSE)
  as.matrix(t(SRTsim::simCounts(sim_obj)))
}

message(sprintf("Running simulator: %s", simulator))
simulated_cell_gene <- switch(
  simulator,
  scdesign3 = simulate_with_scdesign3(),
  srtsim = simulate_with_srtsim(),
  stop(sprintf("Unsupported simulator: %s", simulator))
)

if (!is.matrix(simulated_cell_gene)) {
  simulated_cell_gene <- as.matrix(simulated_cell_gene)
}

if (nrow(simulated_cell_gene) != nrow(cells_df)) {
  stop(sprintf(
    "Simulated matrix row count mismatch: got %d, expected %d",
    nrow(simulated_cell_gene), nrow(cells_df)
  ))
}

if (is.null(colnames(simulated_cell_gene))) {
  colnames(simulated_cell_gene) <- gene_names
}
if (is.null(rownames(simulated_cell_gene))) {
  rownames(simulated_cell_gene) <- cells_df$cell_id
}

simulated_cell_gene <- simulated_cell_gene[, gene_names, drop = FALSE]
rownames(simulated_cell_gene) <- cells_df$cell_id
simulated_cell_gene[is.na(simulated_cell_gene)] <- 0
simulated_cell_gene <- pmax(round(simulated_cell_gene), 0)

output_df <- data.frame(
  cell_id = cells_df$cell_id,
  simulated_cell_gene,
  check.names = FALSE,
  row.names = NULL
)
write.csv(output_df, output_expression, row.names = FALSE, quote = FALSE)
message(sprintf("Wrote simulated expression to %s", output_expression))
