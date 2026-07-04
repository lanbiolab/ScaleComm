"""Semi-synthetic CCC benchmark on real ST backgrounds."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SemiSyntheticConfig:
    background_format: str = "h5ad"
    background_simulator: str = "real"
    h5ad_path: str = "Human_Lymph_Node.h5ad"
    simulator_config_json: str = ""
    simulator_r_script: str = ""
    simulator_output_dir: str = ""
    region_csv: str = "visium_lr_benchmark/spot_region_from_c2l.csv"
    matrix_dir: str = ""
    spatial_csv: str = ""
    metadata_csv: str = ""
    cell_id_col: str = "NAME"
    x_col: str = "X"
    y_col: str = "Y"
    cell_type_col: str = "cell_type"
    metadata_cell_type_col: str = "cluster"
    region_col: str = ""
    dataset_name: str = "human_lymph_node_visium"
    injection_set: str = "lymph_node"
    output_dir: str = "outputs/phase1/exp02_semisynthetic_real_background/datasets"
    n_spots: int | None = None
    random_seed: int = 2025
    replicate_id: int = 0
    n_background_genes: int = 2000
    background_gene_selection: str = "mean_variance_random"
    required_gene_csvs: tuple[str, ...] = ()
    max_candidate_range_quantile: float = 0.08
    negative_edge_fraction: float = 0.25
    candidate_generation: str = "full_radius"
    max_candidates_per_lr: int | None = None
    max_candidates_per_sender_lr: int = 64
    expression_boost_strong: float = 8.0
    expression_boost_weak: float = 3.0


@dataclass(frozen=True)
class SemiSyntheticLRSpec:
    lr_id: str
    ligand: str
    receptor: str
    mechanism: str
    sender_classes: tuple[str, ...]
    receiver_classes: tuple[str, ...]
    active_regions: tuple[str, ...]
    range_multiplier: float
    weak_signal: bool = False
    false_positive_high_expression: bool = False
    true_lr: bool = True


RANGE_LYMPH_NODE_INJECTIONS = (
    SemiSyntheticLRSpec(
        "ICAM1_ITGAL_contact_range_benchmark",
        "ICAM1",
        "ITGAL",
        "juxtacrine_contact",
        ("B", "Macrophage", "Myeloid", "Endo", "Stromal"),
        ("T", "B"),
        ("any",),
        0.45,
    ),
    SemiSyntheticLRSpec(
        "CD40LG_CD40_contact_range_benchmark",
        "CD40LG",
        "CD40",
        "juxtacrine_contact",
        ("T",),
        ("B", "FDC"),
        ("B follicle", "Germinal center", "Germinal_center"),
        0.55,
    ),
    SemiSyntheticLRSpec(
        "IL6_IL6R_short_range_benchmark",
        "IL6",
        "IL6R",
        "short_range_paracrine",
        ("Macrophage", "Myeloid", "Stromal", "B"),
        ("T", "B", "Myeloid"),
        ("any",),
        1.15,
    ),
    SemiSyntheticLRSpec(
        "SPP1_CD44_medium_range_benchmark",
        "SPP1",
        "CD44",
        "medium_range_paracrine",
        ("Macrophage", "Myeloid", "B", "T"),
        ("T", "B", "Stromal", "Other"),
        ("Mixed", "Boundary", "any"),
        1.85,
    ),
    SemiSyntheticLRSpec(
        "CCL19_CCR7_long_range_Tzone_benchmark",
        "CCL19",
        "CCR7",
        "long_range_chemokine_niche",
        ("T", "B", "Stromal", "FDC", "Endo"),
        ("T", "DC", "B"),
        ("T cell zone", "T_zone"),
        3.25,
    ),
    SemiSyntheticLRSpec(
        "CXCL13_CXCR5_long_range_GC_benchmark",
        "CXCL13",
        "CXCR5",
        "long_range_chemokine_niche",
        ("FDC", "B", "Stromal"),
        ("B", "T"),
        ("Germinal center", "Germinal_center", "B follicle"),
        3.00,
        weak_signal=True,
    ),
)


RANGE_MELANOMA_INJECTIONS = (
    SemiSyntheticLRSpec(
        "PDCD1_CD274_contact_range_melanoma",
        "PDCD1",
        "CD274",
        "juxtacrine_contact",
        ("T",),
        ("Tumor", "Macrophage", "Myeloid"),
        ("T_CD8", "T_reg", "mono-mac", "tumor", "malignant", "any"),
        0.55,
    ),
    SemiSyntheticLRSpec(
        "CD48_CD244_contact_range_melanoma",
        "CD48",
        "CD244",
        "juxtacrine_contact",
        ("T", "Macrophage", "Myeloid"),
        ("T", "Macrophage", "Myeloid"),
        ("T_CD8", "T_reg", "mono-mac", "any"),
        0.60,
    ),
    SemiSyntheticLRSpec(
        "SPP1_CD44_short_range_melanoma",
        "SPP1",
        "CD44",
        "short_range_paracrine",
        ("Macrophage", "Myeloid"),
        ("T", "Tumor", "Other"),
        ("mono-mac", "T_CD8", "T_reg", "tumor", "malignant"),
        1.20,
    ),
    SemiSyntheticLRSpec(
        "TGFB1_TGFBR2_medium_range_melanoma",
        "TGFB1",
        "TGFBR2",
        "medium_range_paracrine",
        ("Macrophage", "Myeloid", "Tumor"),
        ("T", "Tumor", "Other", "Macrophage", "Myeloid"),
        ("mono-mac", "T_CD8", "T_reg", "tumor", "malignant", "any"),
        1.90,
    ),
    SemiSyntheticLRSpec(
        "CXCL9_CXCR3_long_range_melanoma",
        "CXCL9",
        "CXCR3",
        "long_range_chemokine_niche",
        ("Macrophage", "Myeloid"),
        ("T",),
        ("mono-mac", "T_CD8", "T_reg"),
        3.20,
    ),
    SemiSyntheticLRSpec(
        "CXCL10_CXCR3_long_range_melanoma",
        "CXCL10",
        "CXCR3",
        "long_range_chemokine_niche",
        ("Macrophage", "Myeloid", "Tumor"),
        ("T",),
        ("mono-mac", "T_CD8", "T_reg", "tumor", "malignant"),
        3.00,
        weak_signal=True,
    ),
)


LYMPH_NODE_INJECTIONS = (
    SemiSyntheticLRSpec(
        "ICAM1_ITGAL_Tzone_contact_semisynthetic",
        "ICAM1",
        "ITGAL",
        "juxtacrine_contact",
        ("B", "Macrophage", "Myeloid"),
        ("T", "B"),
        ("any",),
        0.55,
    ),
    SemiSyntheticLRSpec(
        "CCL19_CCR7_Tzone_semisynthetic",
        "CCL19",
        "CCR7",
        "long_range_chemokine_niche",
        ("T", "B", "Stromal", "FDC", "Endo"),
        ("T", "DC"),
        ("T cell zone", "T_zone"),
        3.0,
    ),
    SemiSyntheticLRSpec(
        "CXCL13_CXCR5_GC_semisynthetic_weak",
        "CXCL13",
        "CXCR5",
        "long_range_chemokine_niche",
        ("FDC", "B"),
        ("B", "T"),
        ("Germinal center", "Germinal_center", "B follicle"),
        2.4,
        weak_signal=True,
    ),
    SemiSyntheticLRSpec(
        "SPP1_CD44_boundary_semisynthetic",
        "SPP1",
        "CD44",
        "short_medium_paracrine",
        ("T", "B", "Macrophage", "Myeloid"),
        ("T", "B", "Stromal", "Other"),
        ("Mixed", "Boundary"),
        1.4,
    ),
    SemiSyntheticLRSpec(
        "ACTB_EGFR_false_positive_real_background",
        "ACTB",
        "EGFR",
        "false_positive_high_expression",
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("any",),
        3.0,
        false_positive_high_expression=True,
        true_lr=False,
    ),
    SemiSyntheticLRSpec(
        "GAPDH_ITGB1_false_positive_real_background",
        "GAPDH",
        "ITGB1",
        "false_positive_high_expression",
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("any",),
        3.0,
        false_positive_high_expression=True,
        true_lr=False,
    ),
)


TONSIL_INJECTIONS = (
    SemiSyntheticLRSpec(
        "ICAM1_ITGAL_tonsil_contact",
        "ICAM1",
        "ITGAL",
        "juxtacrine_contact",
        ("FDC", "Endo", "Macrophage", "Myeloid"),
        ("T", "B"),
        ("Tfh", "T_CD4", "B_germinal_center", "FDC"),
        0.55,
    ),
    SemiSyntheticLRSpec(
        "CXCL13_CXCR5_tonsil_GC",
        "CXCL13",
        "CXCR5",
        "long_range_chemokine_niche",
        ("FDC", "B"),
        ("B", "T"),
        ("FDC", "B_germinal_center", "B_GC", "Tfh"),
        2.6,
    ),
    SemiSyntheticLRSpec(
        "CD40LG_CD40_tonsil_Tfh_B",
        "CD40LG",
        "CD40",
        "short_medium_paracrine",
        ("T",),
        ("B", "FDC"),
        ("Tfh", "T_CD4", "B_germinal_center", "FDC"),
        1.5,
    ),
    SemiSyntheticLRSpec(
        "IL21_IL21R_tonsil_Tfh_B_weak",
        "IL21",
        "IL21R",
        "short_medium_paracrine",
        ("T",),
        ("B",),
        ("Tfh", "T_CD4", "B_germinal_center"),
        1.5,
        weak_signal=True,
    ),
    SemiSyntheticLRSpec(
        "ACTB_EGFR_false_positive_tonsil",
        "ACTB",
        "EGFR",
        "false_positive_high_expression",
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("any",),
        2.6,
        false_positive_high_expression=True,
        true_lr=False,
    ),
    SemiSyntheticLRSpec(
        "GAPDH_ITGB1_false_positive_tonsil",
        "GAPDH",
        "ITGB1",
        "false_positive_high_expression",
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("B", "T", "FDC", "Macrophage", "Endo", "Other"),
        ("any",),
        2.6,
        false_positive_high_expression=True,
        true_lr=False,
    ),
)


MELANOMA_INJECTIONS = (
    SemiSyntheticLRSpec(
        "SPP1_CD44_melanoma_macrophage_T_boundary",
        "SPP1",
        "CD44",
        "short_medium_paracrine",
        ("Macrophage", "Myeloid"),
        ("T", "Tumor", "Other"),
        ("mono-mac", "T_CD8", "T_reg", "tumor", "malignant"),
        1.6,
    ),
    SemiSyntheticLRSpec(
        "CXCL9_CXCR3_melanoma_macrophage_T_weak",
        "CXCL9",
        "CXCR3",
        "long_range_chemokine_niche",
        ("Macrophage", "Myeloid"),
        ("T",),
        ("mono-mac", "T_CD8", "T_reg"),
        2.8,
        weak_signal=True,
    ),
    SemiSyntheticLRSpec(
        "PDCD1_CD274_melanoma_checkpoint",
        "PDCD1",
        "CD274",
        "juxtacrine_contact",
        ("T",),
        ("Tumor", "Macrophage", "Myeloid"),
        ("T_CD8", "T_reg", "mono-mac", "tumor", "malignant"),
        0.7,
    ),
    SemiSyntheticLRSpec(
        "ACTB_EGFR_false_positive_melanoma",
        "ACTB",
        "EGFR",
        "false_positive_high_expression",
        ("B", "T", "FDC", "Macrophage", "Endo", "Tumor", "Other"),
        ("B", "T", "FDC", "Macrophage", "Endo", "Tumor", "Other"),
        ("any",),
        2.8,
        false_positive_high_expression=True,
        true_lr=False,
    ),
    SemiSyntheticLRSpec(
        "GAPDH_ITGB1_false_positive_melanoma",
        "GAPDH",
        "ITGB1",
        "false_positive_high_expression",
        ("B", "T", "FDC", "Macrophage", "Endo", "Tumor", "Other"),
        ("B", "T", "FDC", "Macrophage", "Endo", "Tumor", "Other"),
        ("any",),
        2.8,
        false_positive_high_expression=True,
        true_lr=False,
    ),
)


def generate_semisynthetic_real_background(config: SemiSyntheticConfig) -> Path:
    rng = np.random.default_rng(config.random_seed)
    background = _load_real_background(config, rng)
    cells = background["cells"]
    matrix = background["matrix"]
    obs_names = background["obs_names"]
    gene_symbols = background["gene_symbols"]

    lr_pairs = pd.DataFrame(asdict(spec) for spec in _injection_specs(config.injection_set))
    expression = _extract_expression_table(
        obs_names,
        matrix,
        gene_symbols,
        lr_pairs,
        config,
        rng,
    )
    expression = _simulate_expression_background(expression, cells, config)
    expression = _inject_lr_expression(expression, cells, lr_pairs, config, rng)
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    candidate_range = _candidate_range(coords, config.max_candidate_range_quantile)
    candidates = _make_semisynthetic_candidates(cells, expression, lr_pairs, candidate_range, config, rng)
    truth_tables = _make_semisynthetic_truth_tables(candidates, lr_pairs, candidate_range)

    out_root = Path(config.output_dir)
    rep_dir = out_root / f"{config.dataset_name}_replicate_{int(config.replicate_id):02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(rep_dir / "cells.csv", index=False)
    expression.to_csv(rep_dir / "expression.csv", index=False)
    lr_pairs.to_csv(rep_dir / "lr_pairs.csv", index=False)
    candidates.to_csv(rep_dir / "candidate_interactions.csv", index=False)
    truth_tables["ground_truth_table"].to_csv(rep_dir / "ground_truth_table.csv", index=False)
    truth_tables["truth_lr_pairs"].to_csv(rep_dir / "truth_lr_pairs.csv", index=False)
    truth_tables["truth_cell_type_pairs"].to_csv(rep_dir / "truth_cell_type_pairs.csv", index=False)
    truth_tables["truth_spatial_niches"].to_csv(rep_dir / "truth_spatial_niches.csv", index=False)

    simulator = str(config.background_simulator).lower()
    preserved_properties = [
        "spatial_coordinates",
        "cell_type_composition",
        "region_annotation",
        "local_density",
        "tissue_domain_structure",
    ]
    if simulator == "real":
        preserved_properties.extend(
            [
                "library_size",
                "observed_expression_sparsity",
                "broad_expression_background",
            ]
        )
    else:
        preserved_properties.extend(
            [
                "reference-guided_expression_distribution",
                "reference-guided_sparsity",
                "reference-guided_library_size_trend",
            ]
        )

    metadata = {
        "benchmark": (
            "semisynthetic_reference_simulated_st_background"
            if simulator != "real"
            else "semisynthetic_real_st_background"
        ),
        "benchmark_version": (
            "v3_reference_simulated_truth_tables"
            if simulator != "real"
            else "v2_real_background_truth_tables"
        ),
        "dataset_name": config.dataset_name,
        "injection_set": config.injection_set,
        "replicate": int(config.replicate_id),
        "config": asdict(config),
        "background_simulator": str(config.background_simulator or "real"),
        "source_h5ad": config.h5ad_path,
        "source_region_csv": config.region_csv,
        "source_matrix_dir": config.matrix_dir,
        "source_spatial_csv": config.spatial_csv,
        "source_metadata_csv": config.metadata_csv,
        "preserved_real_properties": preserved_properties,
        "schema": {
            "ground_truth_table": "candidate-level truth table with true_edge, LR, cell-type-pair, niche, weak/strong, and mechanism labels",
            "truth_lr_pairs": "LR-pair-level truth summary",
            "truth_cell_type_pairs": "cell-type-pair-by-LR truth summary",
            "truth_spatial_niches": "spatial-niche-by-LR truth summary",
        },
        "expression_genes": int(expression.shape[1] - 1),
        "candidate_range": float(candidate_range),
        "real_background_summary": _real_background_summary(cells),
        "counts": {
            "cells": int(len(cells)),
            "lr_pairs": int(len(lr_pairs)),
            "candidate_interactions": int(len(candidates)),
            "truth_edges": int(truth_tables["ground_truth_table"]["true_edge"].sum()),
            "truth_lr_positive_pairs": int(truth_tables["truth_lr_pairs"]["true_lr_pair"].sum()),
            "truth_cell_type_lr_pairs": int(
                truth_tables["truth_cell_type_pairs"]["true_cell_type_pair"].sum()
            ),
            "truth_spatial_niche_lr_pairs": int(
                truth_tables["truth_spatial_niches"]["true_spatial_niche"].sum()
            ),
            "positive_interactions": int(candidates["ground_truth"].sum()),
            "weak_positive_interactions": int(
                candidates.loc[candidates["weak_signal"], "ground_truth"].sum()
            ),
        },
    }
    (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame({"replicate": [int(config.replicate_id)], "path": [str(rep_dir)]}).to_csv(out_root / "manifest.csv", index=False)
    return rep_dir


def _simulate_expression_background(
    expression: pd.DataFrame,
    cells: pd.DataFrame,
    config: SemiSyntheticConfig,
) -> pd.DataFrame:
    simulator = str(config.background_simulator or "real").lower()
    if simulator in {"", "real", "none"}:
        return expression
    if simulator not in {"srtsim", "scdesign3"}:
        raise ValueError("background_simulator must be one of 'real', 'srtsim', or 'scdesign3'")

    script_path = _resolve_simulator_r_script(config)
    output_root = Path(config.simulator_output_dir) if config.simulator_output_dir else None
    with tempfile.TemporaryDirectory(prefix=f"{simulator}_background_") as tmpdir:
        workdir = Path(tmpdir)
        if output_root is not None:
            workdir = output_root / f"{config.dataset_name}_replicate_{int(config.replicate_id):02d}"
            workdir.mkdir(parents=True, exist_ok=True)
        input_expression = workdir / "input_expression.csv"
        input_cells = workdir / "input_cells.csv"
        output_expression = workdir / "simulated_expression.csv"
        expression.to_csv(input_expression, index=False)
        cells.to_csv(input_cells, index=False)
        cmd = [
            "Rscript",
            str(script_path),
            "--simulator",
            simulator,
            "--input-expression",
            str(input_expression),
            "--input-cells",
            str(input_cells),
            "--output-expression",
            str(output_expression),
            "--seed",
            str(int(config.random_seed)),
        ]
        if config.simulator_config_json:
            cmd.extend(["--config-json", str(config.simulator_config_json)])
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Reference-based simulator {simulator} failed.\nSTDOUT:\n{exc.stdout}\nSTDERR:\n{exc.stderr}"
            ) from exc
        simulated = pd.read_csv(output_expression)
        simulated["cell_id"] = simulated["cell_id"].astype(str)
        if completed.stdout and output_root is not None:
            (workdir / "simulator_stdout.log").write_text(completed.stdout, encoding="utf-8")
        if completed.stderr and output_root is not None:
            (workdir / "simulator_stderr.log").write_text(completed.stderr, encoding="utf-8")
    return _coerce_simulated_expression(simulated, expression)


def _resolve_simulator_r_script(config: SemiSyntheticConfig) -> Path:
    if config.simulator_r_script:
        path = Path(config.simulator_r_script)
        if not path.exists():
            raise FileNotFoundError(f"simulator_r_script does not exist: {path}")
        return path
    project_root = Path(__file__).resolve().parents[3]
    default = project_root / "scripts" / "benchmark" / "reference_simulator_bridge.R"
    if not default.exists():
        raise FileNotFoundError(f"Default simulator bridge does not exist: {default}")
    return default


def _coerce_simulated_expression(simulated: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    ref = reference.copy()
    simulated = simulated.copy()
    missing_columns = [column for column in ref.columns if column not in simulated.columns]
    if missing_columns:
        raise ValueError(f"Simulated expression is missing required columns: {missing_columns}")
    simulated = simulated[ref.columns].copy()
    if not simulated["cell_id"].astype(str).equals(ref["cell_id"].astype(str)):
        simulated = simulated.set_index("cell_id").reindex(ref["cell_id"].astype(str)).reset_index()
        simulated = simulated.rename(columns={"index": "cell_id"})
    if simulated["cell_id"].isna().any():
        raise ValueError("Simulated expression did not preserve all input cell_id values")
    for column in ref.columns:
        if column == "cell_id":
            simulated[column] = simulated[column].astype(str)
            continue
        simulated[column] = pd.to_numeric(simulated[column], errors="coerce").fillna(0.0)
        simulated[column] = np.clip(np.rint(simulated[column].to_numpy(dtype=float)), 0.0, None)
    return simulated


def _injection_specs(injection_set: str) -> tuple[SemiSyntheticLRSpec, ...]:
    key = str(injection_set or "lymph_node").lower()
    if key in {"range_lymph_node_200", "lymph_node_range_200", "range_mechanism_lymph_node_200"}:
        return _large_mechanism_panel("lymph_node")
    if key in {"range_melanoma_200", "melanoma_range_200", "range_mechanism_melanoma_200"}:
        return _large_mechanism_panel("melanoma")
    if key in {"range_lymph_node", "lymph_node_range", "range_mechanism_lymph_node"}:
        return RANGE_LYMPH_NODE_INJECTIONS
    if key in {"range_melanoma", "melanoma_range", "range_mechanism_melanoma"}:
        return RANGE_MELANOMA_INJECTIONS
    if key in {"lymph", "lymph_node", "human_lymph_node_visium"}:
        return LYMPH_NODE_INJECTIONS
    if key in {"tonsil", "human_tonsil", "slide_tags_tonsil"}:
        return TONSIL_INJECTIONS
    if key in {"melanoma", "melanoma_tme", "human_melanoma_multiome"}:
        return MELANOMA_INJECTIONS
    raise ValueError(f"Unknown injection_set={injection_set!r}; expected lymph_node, tonsil, or melanoma")


def _large_mechanism_panel(tissue: str) -> tuple[SemiSyntheticLRSpec, ...]:
    """Build a 200-LR mechanism benchmark panel.

    The panel is deliberately stratified rather than fully random:
    120 LR pairs are injected positives across four spatial mechanisms and
    80 LR pairs are hard negatives with strong expression or niche/range traps.
    Gene symbols are synthetic but mechanism metadata follows biological CCC
    categories. This keeps the benchmark scalable and avoids cherry-picking a
    handful of canonical LR pairs.
    """

    tissue = str(tissue).lower()
    if tissue == "melanoma":
        context = {
            "contact_sender": ("T", "Macrophage", "Myeloid", "Tumor"),
            "contact_receiver": ("T", "Macrophage", "Myeloid", "Tumor"),
            "short_sender": ("Macrophage", "Myeloid", "Tumor"),
            "short_receiver": ("T", "Tumor", "Other", "Macrophage", "Myeloid"),
            "medium_sender": ("Macrophage", "Myeloid", "Tumor"),
            "medium_receiver": ("T", "Tumor", "Other", "Macrophage", "Myeloid"),
            "long_sender": ("Macrophage", "Myeloid", "Tumor"),
            "long_receiver": ("T",),
            "broad_regions": ("mono-mac", "T_CD8", "T_reg", "tumor", "malignant", "any"),
            "long_regions": ("mono-mac", "T_CD8", "T_reg", "tumor", "malignant"),
            "wrong_regions": ("stromal", "endothelial"),
            "all_classes": ("B", "T", "FDC", "Macrophage", "Endo", "Tumor", "Other", "Myeloid"),
            "prefix": "mel",
            "canonical": RANGE_MELANOMA_INJECTIONS,
        }
    else:
        context = {
            "contact_sender": ("B", "Macrophage", "Myeloid", "Endo", "Stromal", "T"),
            "contact_receiver": ("T", "B", "FDC"),
            "short_sender": ("Macrophage", "Myeloid", "Stromal", "B", "T"),
            "short_receiver": ("T", "B", "Myeloid", "Stromal"),
            "medium_sender": ("Macrophage", "Myeloid", "B", "T", "Stromal"),
            "medium_receiver": ("T", "B", "Stromal", "Other", "FDC"),
            "long_sender": ("T", "B", "Stromal", "FDC", "Endo"),
            "long_receiver": ("T", "DC", "B"),
            "broad_regions": ("B follicle", "Germinal center", "Germinal_center", "T cell zone", "T_zone", "Mixed", "Boundary", "any"),
            "long_regions": ("T cell zone", "T_zone", "Germinal center", "Germinal_center", "B follicle"),
            "wrong_regions": ("tumor", "malignant", "mono-mac"),
            "all_classes": ("B", "T", "FDC", "Macrophage", "Endo", "Stromal", "Other", "Myeloid"),
            "prefix": "ln",
            "canonical": RANGE_LYMPH_NODE_INJECTIONS,
        }

    specs: list[SemiSyntheticLRSpec] = list(context["canonical"])
    used = {(spec.ligand, spec.receptor) for spec in specs}

    positive_templates = [
        ("contact", "juxtacrine_contact", 30, 0.52, "contact_sender", "contact_receiver", "broad_regions"),
        ("short", "short_range_paracrine", 30, 1.15, "short_sender", "short_receiver", "broad_regions"),
        ("medium", "medium_range_paracrine", 30, 1.85, "medium_sender", "medium_receiver", "broad_regions"),
        ("long", "long_range_chemokine_niche", 30, 3.10, "long_sender", "long_receiver", "long_regions"),
    ]
    for family, mechanism, target_n, base_range, sender_key, receiver_key, region_key in positive_templates:
        existing = sum(1 for spec in specs if spec.mechanism == mechanism and spec.true_lr)
        for idx in range(max(target_n - existing, 0)):
            number = idx + 1
            ligand = f"{context['prefix'].upper()}_{family.upper()}L{number:03d}"
            receptor = f"{context['prefix'].upper()}_{family.upper()}R{number:03d}"
            if (ligand, receptor) in used:
                continue
            used.add((ligand, receptor))
            specs.append(
                SemiSyntheticLRSpec(
                    f"{context['prefix']}_{family}_true_{number:03d}",
                    ligand,
                    receptor,
                    mechanism,
                    context[sender_key],
                    context[receiver_key],
                    context[region_key],
                    base_range * (0.88 + 0.24 * ((number % 7) / 6.0)),
                    weak_signal=(number % 5 == 0),
                )
            )

    negative_templates = [
        ("hexp", "false_positive_high_expression", 40, 2.70, "all_classes", "all_classes", "broad_regions", True),
        ("wrongniche", "long_range_chemokine_niche", 20, 3.15, "long_sender", "long_receiver", "wrong_regions", False),
        ("wrongtype", "medium_range_paracrine", 20, 1.80, "contact_receiver", "contact_sender", "broad_regions", False),
    ]
    for family, mechanism, count, base_range, sender_key, receiver_key, region_key, high_expr in negative_templates:
        for idx in range(count):
            number = idx + 1
            ligand = f"{context['prefix'].upper()}_{family.upper()}L{number:03d}"
            receptor = f"{context['prefix'].upper()}_{family.upper()}R{number:03d}"
            if (ligand, receptor) in used:
                continue
            used.add((ligand, receptor))
            specs.append(
                SemiSyntheticLRSpec(
                    f"{context['prefix']}_{family}_negative_{number:03d}",
                    ligand,
                    receptor,
                    mechanism,
                    context[sender_key],
                    context[receiver_key],
                    context[region_key],
                    base_range,
                    weak_signal=False,
                    false_positive_high_expression=high_expr,
                    true_lr=False,
                )
            )

    if len(specs) != 200:
        raise RuntimeError(f"Expected 200 LR specs for {tissue}, got {len(specs)}")
    return tuple(specs)


def _load_real_background(config: SemiSyntheticConfig, rng: np.random.Generator) -> dict:
    fmt = str(config.background_format or "h5ad").lower()
    if fmt == "h5ad":
        return _load_h5ad_background(config, rng)
    if fmt in {"matrix_market", "mtx"}:
        return _load_matrix_market_background(config, rng)
    raise ValueError("background_format must be 'h5ad' or 'matrix_market'")


def _load_h5ad_background(config: SemiSyntheticConfig, rng: np.random.Generator) -> dict:
    try:
        import scanpy as sc
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError("scanpy and scipy are required for h5ad semi-synthetic generation") from exc

    adata = sc.read_h5ad(config.h5ad_path)
    if config.n_spots is not None and config.n_spots < adata.n_obs:
        selected = np.sort(rng.choice(adata.n_obs, size=config.n_spots, replace=False))
        adata = adata[selected].copy()

    region = _load_region_series(config.region_csv, adata.obs_names)
    coords = _normalize_coordinates(np.asarray(adata.obsm["spatial"], dtype=float))
    abundance = _dominant_abundance(adata)
    gene_symbols = _gene_symbols(adata)
    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix)

    cells = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "dominant_cell_type": abundance["dominant_cell_type"].to_numpy(),
            "cell_type": abundance["broad_cell_type"].to_numpy(),
            "region": region.to_numpy(),
            "niche": region.to_numpy(),
            "library_size": _row_sums(matrix),
            "observed_sparsity": _row_sparsity(matrix),
        }
    )
    return {
        "cells": cells,
        "matrix": matrix,
        "obs_names": pd.Index(adata.obs_names.astype(str)),
        "gene_symbols": gene_symbols,
    }


def _load_matrix_market_background(config: SemiSyntheticConfig, rng: np.random.Generator) -> dict:
    try:
        from scipy.io import mmread
    except ImportError as exc:
        raise RuntimeError("scipy is required for MatrixMarket semi-synthetic generation") from exc

    matrix_dir = Path(config.matrix_dir)
    if not matrix_dir.exists():
        raise FileNotFoundError(f"matrix_dir does not exist: {matrix_dir}")
    features = pd.read_csv(matrix_dir / "features.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(matrix_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    matrix = mmread(matrix_dir / "matrix.mtx.gz").tocsr()
    if matrix.shape[0] == len(features) and matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsr()
    elif matrix.shape[0] != len(barcodes):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match features={len(features)} and barcodes={len(barcodes)}"
        )

    spatial = _read_spatial_table(config)
    spatial = spatial[spatial[config.cell_id_col].astype(str).isin(set(barcodes))]
    spatial = spatial.set_index(config.cell_id_col, drop=False)
    common = pd.Index([barcode for barcode in barcodes if barcode in spatial.index])
    if len(common) == 0:
        raise ValueError("No overlapping cells between MatrixMarket barcodes and spatial table")
    if config.n_spots is not None and config.n_spots < len(common):
        selected = np.sort(rng.choice(len(common), size=config.n_spots, replace=False))
        common = common[selected]

    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    indices = np.asarray([barcode_to_idx[barcode] for barcode in common], dtype=np.int64)
    matrix = matrix[indices].tocsr()
    spatial = spatial.loc[common]
    raw_cell_type = _matrix_market_cell_type(spatial, config)
    coords = _normalize_coordinates(spatial[[config.x_col, config.y_col]].to_numpy(dtype=float))
    region = _matrix_market_region(spatial, raw_cell_type, config)
    cells = pd.DataFrame(
        {
            "cell_id": common.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "dominant_cell_type": raw_cell_type.astype(str).to_numpy(),
            "cell_type": [_broad_cell_type(value) for value in raw_cell_type],
            "region": region.astype(str).to_numpy(),
            "niche": region.astype(str).to_numpy(),
            "library_size": _row_sums(matrix),
            "observed_sparsity": _row_sparsity(matrix),
        }
    )
    gene_symbols = features[1].astype(str) if features.shape[1] > 1 else features[0].astype(str)
    return {
        "cells": cells,
        "matrix": matrix,
        "obs_names": common.astype(str),
        "gene_symbols": pd.Series(gene_symbols.to_numpy()),
    }


def _read_spatial_table(config: SemiSyntheticConfig) -> pd.DataFrame:
    if not config.spatial_csv:
        raise ValueError("spatial_csv is required for MatrixMarket backgrounds")
    spatial = pd.read_csv(config.spatial_csv)
    spatial = spatial[spatial[config.cell_id_col].astype(str) != "TYPE"].copy()
    for column in (config.x_col, config.y_col):
        spatial[column] = pd.to_numeric(spatial[column], errors="coerce")
    return spatial.dropna(subset=[config.x_col, config.y_col])


def _matrix_market_cell_type(spatial: pd.DataFrame, config: SemiSyntheticConfig) -> pd.Series:
    if config.cell_type_col in spatial.columns:
        return spatial[config.cell_type_col].astype(str)
    if config.metadata_csv:
        metadata = pd.read_csv(config.metadata_csv)
        metadata = metadata[metadata[config.cell_id_col].astype(str) != "TYPE"].copy()
        if config.metadata_cell_type_col in metadata.columns:
            mapped = metadata.set_index(config.cell_id_col)[config.metadata_cell_type_col].astype(str)
            return spatial[config.cell_id_col].astype(str).map(mapped).fillna("Unknown")
    return pd.Series(["Unknown"] * len(spatial), index=spatial.index)


def _matrix_market_region(spatial: pd.DataFrame, raw_cell_type: pd.Series, config: SemiSyntheticConfig) -> pd.Series:
    if config.region_col and config.region_col in spatial.columns:
        return spatial[config.region_col].astype(str)
    return raw_cell_type.astype(str)


def _real_background_summary(cells: pd.DataFrame) -> dict[str, float | int]:
    coords = cells[["x", "y"]].to_numpy(dtype=float)
    local_density = _local_density_summary(coords, k=6)
    library_size = pd.to_numeric(cells["library_size"], errors="coerce")
    sparsity = pd.to_numeric(cells["observed_sparsity"], errors="coerce")
    return {
        "coordinate_span_x": float(cells["x"].max() - cells["x"].min()),
        "coordinate_span_y": float(cells["y"].max() - cells["y"].min()),
        "median_library_size": float(library_size.median()),
        "library_size_iqr": float(library_size.quantile(0.75) - library_size.quantile(0.25)),
        "mean_observed_sparsity": float(sparsity.mean()),
        "median_nn_distance": float(local_density["median_nn_distance"]),
        "median_local_density": float(local_density["median_local_density"]),
        "regions": int(cells["region"].nunique()),
        "cell_types": int(cells["cell_type"].nunique()),
        "dominant_region_fraction": float(cells["region"].value_counts(normalize=True).max()),
        "dominant_cell_type_fraction": float(cells["cell_type"].value_counts(normalize=True).max()),
    }


def _make_semisynthetic_truth_tables(
    candidates: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidate_range: float,
) -> dict[str, pd.DataFrame]:
    lr_meta = lr_pairs[["lr_id", "range_multiplier", "active_regions", "true_lr"]].copy()
    truth = candidates.merge(lr_meta, on="lr_id", how="left", validate="many_to_one")
    truth = truth.rename(
        columns={
            "ground_truth": "true_edge",
            "mechanism": "mechanism_type",
            "sender_niche": "sender_spatial_niche",
            "receiver_niche": "receiver_spatial_niche",
        }
    )
    truth["true_edge"] = truth["true_edge"].astype(int)
    truth["spatial_range"] = float(candidate_range) * truth["range_multiplier"].astype(float)
    truth["active_niche"] = truth["active_regions"].map(_tuple_to_label)
    truth["true_lr_pair"] = truth["true_lr"].astype(bool) & ~truth[
        "false_positive_high_expression"
    ].astype(bool)
    truth["true_cell_type_pair"] = truth["type_rule_satisfied"].astype(bool) & truth[
        "true_lr_pair"
    ].astype(bool)
    truth["true_spatial_niche"] = truth["niche_rule_satisfied"].astype(bool) & truth[
        "true_lr_pair"
    ].astype(bool)
    truth["weak_or_strong"] = np.where(truth["weak_signal"].astype(bool), "weak", "strong")
    truth.loc[truth["false_positive_high_expression"].astype(bool), "weak_or_strong"] = (
        "false_positive_high_expression"
    )
    ground_truth_table = truth[
        [
            "candidate_id",
            "sender_id",
            "receiver_id",
            "lr_id",
            "ligand",
            "receptor",
            "mechanism_type",
            "sender_cell_type",
            "receiver_cell_type",
            "sender_spatial_niche",
            "receiver_spatial_niche",
            "active_niche",
            "distance",
            "spatial_range",
            "weak_signal",
            "weak_or_strong",
            "false_positive_high_expression",
            "true_lr_pair",
            "true_cell_type_pair",
            "true_spatial_niche",
            "type_rule_satisfied",
            "range_rule_satisfied",
            "range_violation_type",
            "wrong_range_false_positive",
            "niche_rule_satisfied",
            "true_edge",
        ]
    ].copy()

    truth_lr_pairs = (
        truth.groupby(
            [
                "lr_id",
                "ligand",
                "receptor",
                "mechanism_type",
                "weak_signal",
                "weak_or_strong",
                "false_positive_high_expression",
                "spatial_range",
                "active_niche",
            ],
            dropna=False,
        )
        .agg(
            candidate_edges=("candidate_id", "count"),
            true_edges=("true_edge", "sum"),
            true_cell_type_candidates=("true_cell_type_pair", "sum"),
            true_spatial_niche_candidates=("true_spatial_niche", "sum"),
            positive_median_distance=("distance", lambda values: _median_where(values, truth.loc[values.index, "true_edge"])),
        )
        .reset_index()
    )
    truth_lr_pairs["true_lr_pair"] = (
        (truth_lr_pairs["true_edges"] > 0)
        & ~truth_lr_pairs["false_positive_high_expression"].astype(bool)
    )

    truth_cell_type_pairs = (
        truth.groupby(
            [
                "lr_id",
                "mechanism_type",
                "weak_or_strong",
                "sender_cell_type",
                "receiver_cell_type",
            ],
            dropna=False,
        )
        .agg(candidate_edges=("candidate_id", "count"), true_edges=("true_edge", "sum"))
        .reset_index()
    )
    truth_cell_type_pairs["true_cell_type_pair"] = truth_cell_type_pairs["true_edges"] > 0

    truth_spatial_niches = (
        truth.groupby(
            [
                "lr_id",
                "mechanism_type",
                "weak_or_strong",
                "sender_spatial_niche",
                "receiver_spatial_niche",
                "active_niche",
            ],
            dropna=False,
        )
        .agg(candidate_edges=("candidate_id", "count"), true_edges=("true_edge", "sum"))
        .reset_index()
    )
    truth_spatial_niches["true_spatial_niche"] = truth_spatial_niches["true_edges"] > 0
    return {
        "ground_truth_table": ground_truth_table,
        "truth_lr_pairs": truth_lr_pairs,
        "truth_cell_type_pairs": truth_cell_type_pairs,
        "truth_spatial_niches": truth_spatial_niches,
    }


def _tuple_to_label(value) -> str:
    if isinstance(value, (tuple, list)):
        return "|".join(str(item) for item in value)
    return str(value)


def _median_where(values: pd.Series, mask: pd.Series) -> float:
    selected = values[mask.astype(bool)]
    if selected.empty:
        return float("nan")
    return float(selected.median())


def _local_density_summary(coords: np.ndarray, *, k: int) -> dict[str, float]:
    if len(coords) < 2:
        return {"median_nn_distance": float("nan"), "median_local_density": float("nan")}
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((delta * delta).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    kth = min(int(k), len(coords) - 1)
    sorted_distances = np.sort(distances, axis=1)
    nn = sorted_distances[:, 0]
    radius = sorted_distances[:, kth - 1]
    density = kth / (np.pi * np.maximum(radius, 1e-12) ** 2)
    return {
        "median_nn_distance": float(np.median(nn)),
        "median_local_density": float(np.median(density)),
    }


def _load_region_series(region_csv: str, obs_names: pd.Index) -> pd.Series:
    path = Path(region_csv)
    if not path.exists():
        return pd.Series(["Unknown"] * len(obs_names), index=obs_names)
    region = pd.read_csv(path)
    spot_col = "spot" if "spot" in region.columns else region.columns[0]
    region_col = "region" if "region" in region.columns else region.columns[-1]
    mapped = region.set_index(spot_col)[region_col].astype(str)
    return pd.Series(obs_names.astype(str), index=obs_names).map(mapped).fillna("Unknown")


def _normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.maximum(high - low, 1e-12)
    return (coords - low) / span


def _dominant_abundance(adata) -> pd.DataFrame:
    if "means_cell_abundance_w_sf" not in adata.obsm:
        dominant = pd.Series(["Unknown"] * adata.n_obs, index=adata.obs_names)
    else:
        abundance = adata.obsm["means_cell_abundance_w_sf"]
        dominant = abundance.idxmax(axis=1).astype(str)
        dominant = dominant.str.replace("meanscell_abundance_w_sf_", "", regex=False)
    return pd.DataFrame(
        {
            "dominant_cell_type": dominant.to_numpy(),
            "broad_cell_type": [_broad_cell_type(value) for value in dominant],
        },
        index=adata.obs_names,
    )


def _broad_cell_type(label: str) -> str:
    text = str(label).lower()
    if text.startswith("b_") or "b_" in text or text.startswith("b"):
        return "B"
    if text.startswith("t_") or "t_" in text or "tcell" in text:
        return "T"
    if "fdc" in text:
        return "FDC"
    if "macrophage" in text or "mono" in text or "myeloid" in text:
        return "Macrophage"
    if "tumor" in text or "malignant" in text or "melanoma" in text or "cancer" in text:
        return "Tumor"
    if "endo" in text:
        return "Endo"
    if "dc" in text:
        return "DC"
    if "fibro" in text or "strom" in text:
        return "Stromal"
    return "Other"


def _gene_symbols(adata) -> pd.Series:
    if "SYMBOL" in adata.var.columns:
        return adata.var["SYMBOL"].astype(str)
    return pd.Series(adata.var_names.astype(str), index=adata.var_names)


def _row_sums(matrix) -> np.ndarray:
    values = np.asarray(matrix.sum(axis=1)).reshape(-1)
    return values.astype(float)


def _row_sparsity(matrix) -> np.ndarray:
    if hasattr(matrix, "getnnz"):
        nonzero = matrix.getnnz(axis=1)
        n_vars = matrix.shape[1]
    else:
        nonzero = (matrix != 0).sum(axis=1)
        n_vars = matrix.shape[1]
    return 1.0 - np.asarray(nonzero).reshape(-1) / float(n_vars)


def _extract_expression_table(
    obs_names: pd.Index,
    matrix,
    gene_symbols: pd.Series,
    lr_pairs: pd.DataFrame,
    config: SemiSyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    lr_genes = sorted(set(lr_pairs["ligand"]) | set(lr_pairs["receptor"]))
    symbol_to_index = _first_symbol_indices(gene_symbols)
    available_symbols = pd.Index(symbol_to_index.keys())
    required_genes = _load_required_genes(config.required_gene_csvs)
    background = _select_background_genes(
        matrix,
        available_symbols,
        symbol_to_index,
        exclude=set(lr_genes) | required_genes,
        n_background_genes=config.n_background_genes,
        strategy=config.background_gene_selection,
        rng=rng,
    )
    genes = _unique_preserve_order(lr_genes + sorted(required_genes) + background)

    data: dict[str, np.ndarray] = {"cell_id": obs_names.astype(str)}
    for gene in genes:
        gene_index = symbol_to_index.get(gene)
        if gene_index is None:
            data[gene] = np.zeros(len(obs_names), dtype=float)
        else:
            values = matrix[:, int(gene_index)]
            if hasattr(values, "toarray"):
                values = values.toarray()
            data[gene] = np.asarray(values).reshape(-1).astype(float)
    return pd.DataFrame(data)


def _load_required_genes(required_gene_csvs: tuple[str, ...]) -> set[str]:
    genes: set[str] = set()
    for item in required_gene_csvs:
        if not item:
            continue
        path = Path(item)
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for column in ("gene", "genes", "target", "targets", "target_genes", "TGs", "TargetGenes"):
            if column not in table.columns:
                continue
            for value in table[column].dropna().tolist():
                genes.update(_split_gene_list(value))
        if {"ligand", "receptor"}.issubset(table.columns):
            genes.update(table["ligand"].dropna().astype(str).tolist())
            genes.update(table["receptor"].dropna().astype(str).tolist())
    return {gene for gene in genes if gene}


def _select_background_genes(
    matrix,
    available_symbols: pd.Index,
    symbol_to_index: dict[str, int],
    *,
    exclude: set[str],
    n_background_genes: int,
    strategy: str,
    rng: np.random.Generator,
) -> list[str]:
    pool = [gene for gene in available_symbols if gene not in exclude]
    if n_background_genes <= 0 or not pool:
        return []

    pool = [gene for gene in pool if gene in symbol_to_index]
    n_background_genes = min(int(n_background_genes), len(pool))
    if n_background_genes == 0:
        return []

    strategy = str(strategy or "random").lower()
    if strategy == "random":
        return sorted(rng.choice(np.array(pool), size=n_background_genes, replace=False).tolist())

    n_mean = n_background_genes // 3
    n_var = n_background_genes // 3
    n_random = n_background_genes - n_mean - n_var
    if strategy in {"mean", "high_mean"}:
        n_mean, n_var, n_random = n_background_genes, 0, 0
    elif strategy in {"variance", "high_variance", "hvg"}:
        n_mean, n_var, n_random = 0, n_background_genes, 0

    selected: list[str] = []
    gene_stats = _gene_stats(matrix, pool, symbol_to_index)
    if n_mean:
        selected.extend(_rank_genes_by_stat(gene_stats, "mean", n_mean))
    if n_var:
        selected.extend(_rank_genes_by_stat(gene_stats, "variance", n_var))

    selected_set = set(selected)
    remaining = [gene for gene in pool if gene not in selected_set]
    if n_random and remaining:
        selected.extend(
            rng.choice(np.array(remaining), size=min(n_random, len(remaining)), replace=False).tolist()
        )

    if len(set(selected)) < n_background_genes:
        selected_set = set(selected)
        remaining = [gene for gene in pool if gene not in selected_set]
        selected.extend(remaining[: n_background_genes - len(selected_set)])
    return sorted(_unique_preserve_order(selected)[:n_background_genes])


def _first_symbol_indices(gene_symbols: pd.Series) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, gene in enumerate(gene_symbols.to_numpy()):
        gene = str(gene)
        if gene not in out:
            out[gene] = idx
    return out


def _gene_stats(
    matrix,
    genes: list[str],
    symbol_to_index: dict[str, int],
) -> pd.DataFrame:
    indices = [int(symbol_to_index[gene]) for gene in genes]
    submatrix = matrix[:, indices]
    if hasattr(submatrix, "toarray"):
        submatrix = submatrix.toarray()
    array = np.asarray(submatrix, dtype=float)
    return pd.DataFrame(
        {
            "gene": genes,
            "mean": array.mean(axis=0),
            "variance": array.var(axis=0),
        }
    )


def _rank_genes_by_stat(gene_stats: pd.DataFrame, stat: str, n: int) -> list[str]:
    ranked = gene_stats.sort_values(stat, ascending=False)
    return ranked["gene"].head(n).astype(str).tolist()


def _split_gene_list(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value)
    for sep in (";", "|"):
        text = text.replace(sep, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _inject_lr_expression(
    expression: pd.DataFrame,
    cells: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    config: SemiSyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    expr = expression.copy()
    for lr in lr_pairs.itertuples(index=False):
        ligand_mask = _eligible_cells(cells, lr.sender_classes, lr.active_regions)
        receptor_mask = _eligible_cells(cells, lr.receiver_classes, lr.active_regions)
        boost = config.expression_boost_weak if lr.weak_signal else config.expression_boost_strong
        if lr.false_positive_high_expression:
            ligand_mask = np.ones(len(cells), dtype=bool)
            receptor_mask = np.ones(len(cells), dtype=bool)
            boost = config.expression_boost_strong * 1.6

        expr.loc[ligand_mask, lr.ligand] = expr.loc[ligand_mask, lr.ligand] + rng.poisson(
            boost, size=int(ligand_mask.sum())
        )
        expr.loc[receptor_mask, lr.receptor] = expr.loc[receptor_mask, lr.receptor] + rng.poisson(
            boost * 0.8, size=int(receptor_mask.sum())
        )
    return expr


def _eligible_cells(cells: pd.DataFrame, classes: tuple[str, ...], regions: tuple[str, ...]) -> np.ndarray:
    class_mask = cells["cell_type"].isin(classes).to_numpy()
    if "any" in regions:
        region_mask = np.ones(len(cells), dtype=bool)
    else:
        region_mask = cells["region"].isin(regions).to_numpy()
    return class_mask & region_mask


def _candidate_range(coords: np.ndarray, quantile: float) -> float:
    sample = coords
    if len(coords) > 1200:
        sample = coords[np.linspace(0, len(coords) - 1, 1200).astype(int)]
    delta = sample[:, None, :] - sample[None, :, :]
    dist = np.sqrt((delta * delta).sum(axis=2))
    dist = dist[np.triu_indices_from(dist, k=1)]
    return float(np.quantile(dist, quantile))


def _make_semisynthetic_candidates(
    cells: pd.DataFrame,
    expression: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    candidate_range: float,
    config: SemiSyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    coords = cells[["x", "y"]].to_numpy()
    if config.candidate_generation not in {"full_radius", "sparse_sampled", "range_stress"}:
        raise ValueError(
            "candidate_generation must be one of 'full_radius', 'sparse_sampled', or 'range_stress'"
        )
    distances = None
    if config.candidate_generation == "full_radius":
        delta = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((delta * delta).sum(axis=2))
        np.fill_diagonal(distances, np.inf)

    cell_ids = cells["cell_id"].astype(str).to_numpy()
    cell_types = cells["cell_type"].astype(str).to_numpy()
    regions = cells["region"].astype(str).to_numpy()
    gene_expr = {
        column: expression[column].to_numpy(dtype=float)
        for column in expression.columns
        if column != "cell_id"
    }

    frames: list[pd.DataFrame] = []
    max_lr_range = max(
        float(candidate_range * lr.range_multiplier)
        for lr in lr_pairs.itertuples(index=False)
    )
    for lr in lr_pairs.itertuples(index=False):
        spatial_range = float(candidate_range * lr.range_multiplier)
        search_range = max_lr_range if config.candidate_generation == "range_stress" else spatial_range
        if config.candidate_generation in {"sparse_sampled", "range_stress"}:
            pairs = _sample_sparse_semisynthetic_pairs(
                cells,
                coords,
                lr,
                spatial_range=spatial_range,
                search_range=search_range,
                config=config,
                rng=rng,
            )
            sender_idx = pairs[:, 0] if len(pairs) else np.asarray([], dtype=np.int64)
            receiver_idx = pairs[:, 1] if len(pairs) else np.asarray([], dtype=np.int64)
            distance = (
                np.linalg.norm(coords[sender_idx] - coords[receiver_idx], axis=1)
                if len(pairs)
                else np.asarray([], dtype=float)
            )
        else:
            assert distances is not None
            sender_idx, receiver_idx = np.nonzero(distances <= search_range)
            distance = distances[sender_idx, receiver_idx]
        if len(sender_idx) == 0:
            continue

        sender_eligible = _eligible_cells(cells, lr.sender_classes, lr.active_regions)
        receiver_eligible = _eligible_cells(cells, lr.receiver_classes, lr.active_regions)

        type_ok = sender_eligible[sender_idx] & receiver_eligible[receiver_idx]
        range_ok = _semisynthetic_range_rule(lr.mechanism, distance, spatial_range)
        range_violation_type = _semisynthetic_range_violation_type(
            lr.mechanism,
            distance,
            spatial_range,
        )
        if "any" in lr.active_regions:
            region_ok = np.ones(len(sender_idx), dtype=bool)
        else:
            active_regions = np.asarray(lr.active_regions, dtype=str)
            region_ok = np.isin(regions[sender_idx], active_regions) & np.isin(
                regions[receiver_idx], active_regions
            )

        eligible_label = bool(lr.true_lr) & type_ok & range_ok & region_ok
        labels = np.zeros(len(sender_idx), dtype=bool)
        if np.any(eligible_label) and not lr.false_positive_high_expression:
            probability = _semisynthetic_signal_probability(
                lr.mechanism,
                distance[eligible_label],
                spatial_range,
                weak_signal=bool(lr.weak_signal),
            )
            labels[eligible_label] = rng.random(int(eligible_label.sum())) < probability

        keep = labels | (rng.random(len(labels)) < config.negative_edge_fraction)
        if not np.any(keep):
            continue

        sender_idx = sender_idx[keep]
        receiver_idx = receiver_idx[keep]
        labels = labels[keep]
        type_ok = type_ok[keep]
        range_ok = range_ok[keep]
        range_violation_type = range_violation_type[keep]
        region_ok = region_ok[keep]
        distance = distance[keep]
        ligand_expr = gene_expr[lr.ligand][sender_idx]
        receptor_expr = gene_expr[lr.receptor][receiver_idx]
        frames.append(
            pd.DataFrame(
                {
                    "sender_id": cell_ids[sender_idx],
                    "receiver_id": cell_ids[receiver_idx],
                    "lr_id": lr.lr_id,
                    "ligand": lr.ligand,
                    "receptor": lr.receptor,
                    "mechanism": lr.mechanism,
                    "distance": distance.astype(float),
                    "sender_cell_type": cell_types[sender_idx],
                    "receiver_cell_type": cell_types[receiver_idx],
                    "sender_niche": regions[sender_idx],
                    "receiver_niche": regions[receiver_idx],
                    "ground_truth": labels.astype(int),
                    "weak_signal": bool(lr.weak_signal),
                    "false_positive_high_expression": bool(lr.false_positive_high_expression),
                    "type_rule_satisfied": type_ok.astype(bool),
                    "range_rule_satisfied": range_ok.astype(bool),
                    "range_violation_type": range_violation_type,
                    "wrong_range_false_positive": (
                        type_ok.astype(bool)
                        & region_ok.astype(bool)
                        & (~range_ok.astype(bool))
                        & bool(lr.true_lr)
                    ),
                    "niche_rule_satisfied": region_ok.astype(bool),
                    "ligand_expr": ligand_expr.astype(float),
                    "receptor_expr": receptor_expr.astype(float),
                    "coexpression_distance_score": (
                        np.log1p(ligand_expr) * np.log1p(receptor_expr) / (1.0 + distance)
                    ).astype(float),
                }
            )
        )

    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    candidates.insert(0, "candidate_id", [f"semi_cand_{i:07d}" for i in range(len(candidates))])
    return candidates


def _sample_sparse_semisynthetic_pairs(
    cells: pd.DataFrame,
    coords: np.ndarray,
    lr,
    *,
    spatial_range: float,
    search_range: float | None = None,
    config: SemiSyntheticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    search_range = float(search_range if search_range is not None else spatial_range)
    sender_eligible = _eligible_cells(cells, lr.sender_classes, lr.active_regions)
    receiver_eligible = _eligible_cells(cells, lr.receiver_classes, lr.active_regions)
    all_idx = np.arange(len(cells), dtype=np.int64)
    max_pairs = config.max_candidates_per_lr
    valid_budget = None if max_pairs is None else max(int(max_pairs * 0.55), 1)
    background_budget = None if max_pairs is None else max(int(max_pairs * 1.35), 1)
    valid_pairs = _sample_radius_pairs(
        coords,
        np.flatnonzero(sender_eligible),
        np.flatnonzero(receiver_eligible),
        radius=spatial_range,
        max_per_sender=config.max_candidates_per_sender_lr,
        max_pairs=valid_budget,
        rng=rng,
    )
    background_pairs = _sample_radius_pairs(
        coords,
        all_idx,
        all_idx,
        radius=search_range,
        max_per_sender=config.max_candidates_per_sender_lr,
        max_pairs=background_budget,
        rng=rng,
    )
    if len(valid_pairs) == 0:
        pairs = background_pairs
    elif len(background_pairs) == 0:
        pairs = valid_pairs
    else:
        pairs = np.vstack([valid_pairs, background_pairs])
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(pairs, axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def _semisynthetic_signal_probability(
    mechanism: str,
    distance: np.ndarray,
    spatial_range: float,
    *,
    weak_signal: bool,
) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    spatial_range = max(float(spatial_range), 1e-6)
    mechanism = str(mechanism)
    if mechanism == "juxtacrine_contact":
        probability = np.full(len(distance), 0.88, dtype=float)
    elif mechanism == "short_range_paracrine":
        probability = 0.18 + 0.70 * np.exp(-distance / (0.55 * spatial_range))
    elif mechanism in {"short_medium_paracrine", "medium_range_paracrine"}:
        peak = 0.35 * spatial_range
        width = 0.38 * spatial_range
        probability = 0.22 + 0.58 * np.exp(-((distance - peak) ** 2) / (2.0 * width**2))
    elif mechanism == "long_range_chemokine_niche":
        # Long-range benchmark truth should not collapse to nearest-neighbor contact.
        peak = 0.62 * spatial_range
        width = 0.30 * spatial_range
        probability = 0.20 + 0.58 * np.exp(-((distance - peak) ** 2) / (2.0 * width**2))
        probability = np.where(distance < 0.18 * spatial_range, probability * 0.35, probability)
    else:
        probability = np.full(len(distance), 0.70, dtype=float)
    if weak_signal:
        probability *= 0.52
    return np.clip(probability, 0.0, 0.95)


def _semisynthetic_range_rule(
    mechanism: str,
    distance: np.ndarray,
    spatial_range: float,
) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    spatial_range = max(float(spatial_range), 1e-6)
    mechanism = str(mechanism)
    if mechanism == "juxtacrine_contact":
        return distance <= spatial_range
    if mechanism == "short_range_paracrine":
        return distance <= spatial_range
    if mechanism in {"short_medium_paracrine", "medium_range_paracrine"}:
        return distance <= spatial_range
    if mechanism == "long_range_chemokine_niche":
        # Long-range chemokine-like signals are not contact edges in this benchmark.
        return (distance >= 0.35 * spatial_range) & (distance <= spatial_range)
    return distance <= spatial_range


def _semisynthetic_range_violation_type(
    mechanism: str,
    distance: np.ndarray,
    spatial_range: float,
) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    spatial_range = max(float(spatial_range), 1e-6)
    mechanism = str(mechanism)
    out = np.full(len(distance), "in_range", dtype=object)
    if mechanism == "long_range_chemokine_niche":
        out[distance < 0.35 * spatial_range] = "too_close_for_long_range_lr"
        out[distance > spatial_range] = "too_far_for_lr"
    else:
        out[distance > spatial_range] = "too_far_for_lr"
    return out


def _sample_radius_pairs(
    coords: np.ndarray,
    sender_indices: np.ndarray,
    receiver_indices: np.ndarray,
    *,
    radius: float,
    max_per_sender: int,
    max_pairs: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(sender_indices) == 0 or len(receiver_indices) == 0 or radius <= 0:
        return np.empty((0, 2), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("sparse_sampled candidate generation requires scipy") from exc

    receiver_indices = np.asarray(receiver_indices, dtype=np.int64)
    tree = cKDTree(coords[receiver_indices])
    rows: list[tuple[int, int]] = []
    sender_order = np.asarray(sender_indices, dtype=np.int64).copy()
    rng.shuffle(sender_order)
    per_sender = max(int(max_per_sender), 1)
    for sender_idx in sender_order:
        local = tree.query_ball_point(coords[int(sender_idx)], r=float(radius))
        if not local:
            continue
        receiver_global = receiver_indices[np.asarray(local, dtype=np.int64)]
        receiver_global = receiver_global[receiver_global != int(sender_idx)]
        if len(receiver_global) == 0:
            continue
        if len(receiver_global) > per_sender:
            receiver_global = rng.choice(receiver_global, size=per_sender, replace=False)
        rows.extend((int(sender_idx), int(receiver_idx)) for receiver_idx in receiver_global)
        if max_pairs is not None and len(rows) >= max_pairs:
            rows = rows[: int(max_pairs)]
            break
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)
