#!/usr/bin/env python
"""Build Phase 7 Goal 5 cross-platform real-data status and validation summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


PRIORITY_PLATFORM_PATH_HINTS = {
    "xenium_human_lymph_node": ["xenium_human_lymph_node", "hLymphNode", "hlymphnode"],
    "visium_hd_crc": ["VisiumHD", "visium_hd", "visium-hd", "CRC"],
    "cosmx_melanoma": ["CosMx", "cosmx"],
    "merfish_or_seqfish_or_slideseqv2": ["MERFISH", "seqFISH", "Slide-seq", "slideseq"],
}


PRIORITY_STANDARDIZATION_IDS = [
    "xenium_human_lymph_node",
    "xenium_human_lymph_node_subset",
    "lung_cancer_visium_subset",
    "post_xenium_human_colon_cancer_subset",
    "spamosaic_tonsil_A1_subset",
    "spamosaic_tonsil_D1_subset",
    "visium_hd_crc_p2crc_subset",
    "visium_hd_crc",
    "cosmx_melanoma",
    "merfish_or_seqfish_or_slideseqv2",
]


ACQUISITION_CANDIDATES = [
    {
        "platform": "xenium_human_lymph_node",
        "dataset_name": "Human Lymph Node Preview Data (Xenium Human Multi-Tissue and Cancer Panel)",
        "source_url": "https://www.10xgenomics.com/cn/datasets/human-lymph-node-preview-data-xenium-human-multi-tissue-and-cancer-panel-1-standard",
        "provider": "10x Genomics",
        "tissue": "human lymph node",
        "technology": "Xenium In Situ Gene Expression",
        "reported_scale": "377985 cells; 18576746 high-quality transcripts",
        "access_status": "public_cc_by_4_0",
        "phase7_use": "T-zone/GC communication and cross-platform lymph-node validation",
        "download_priority": "high",
        "local_status": "not_found_locally",
        "next_action": "download Xenium output bundle or cell-feature matrix; convert cells, expression, coordinates, and cell-type/domain annotations to SpatialSelf schema",
    },
    {
        "platform": "xenium_human_lymph_node",
        "dataset_name": "Preview Data: FFPE Human Lymph Node with 5K Pan Tissue and Pathways Panel",
        "source_url": "https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression",
        "provider": "10x Genomics",
        "tissue": "reactive human lymph node",
        "technology": "Xenium Prime 5K preview",
        "reported_scale": "708983 cells; 202153544 high-quality decoded transcripts",
        "access_status": "public_preview",
        "phase7_use": "higher-plex Xenium lymph-node stress test if file size is manageable",
        "download_priority": "medium",
        "local_status": "not_found_locally",
        "next_action": "prefer subset/ROI first because full preview is very large",
    },
    {
        "platform": "visium_hd_crc",
        "dataset_name": "Visium HD Spatial Gene Expression Library, Human Colorectal Cancer (FFPE)",
        "source_url": "https://www.10xgenomics.com/jp/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc",
        "provider": "10x Genomics",
        "tissue": "human colorectal cancer",
        "technology": "Visium HD Spatial Gene Expression",
        "reported_scale": "Space Ranger 3.0.0 HD output",
        "access_status": "public_10x_dataset",
        "phase7_use": "epithelial-stromal/immune niche and high-resolution spot/bin scalability",
        "download_priority": "high",
        "local_status": "not_found_locally",
        "next_action": "download binned feature matrices and spatial coordinates; start with 16um or ROI before full 2um grid",
    },
    {
        "platform": "visium_hd_crc",
        "dataset_name": "Visium HD Spatial Gene Expression Library, Human Colon Cancer (FFPE), 6.5 mm Capture Area",
        "source_url": "https://www.10xgenomics.com/cn/datasets/visium-hd-cytassist-6p5mm-human-colon-cancer",
        "provider": "10x Genomics",
        "tissue": "human colon cancer",
        "technology": "Visium HD Spatial Gene Expression",
        "reported_scale": "Space Ranger 4.1.0 HD output",
        "access_status": "public_cc_by_4_0",
        "phase7_use": "newer Visium HD colon cancer platform replication",
        "download_priority": "medium",
        "local_status": "not_found_locally",
        "next_action": "use if manuscript needs a newer Space Ranger 4.1.0 platform example",
    },
    {
        "platform": "cosmx_melanoma",
        "dataset_name": "Spatial profiling of benign and malignant melanocytic tumors via RNA-SMI (CosMx)",
        "source_url": "https://datadryad.org/dataset/doi:10.5061/dryad.ksn02v7b1",
        "provider": "Dryad",
        "tissue": "melanocytic tumors including melanoma and metastasis",
        "technology": "CosMx RNA-SMI",
        "reported_scale": "Slide 4 includes 84312 cells across five melanocytic tumors",
        "access_status": "public_dryad",
        "phase7_use": "tumor-immune boundary and melanoma cross-platform validation",
        "download_priority": "high",
        "local_status": "not_found_locally",
        "next_action": "download individual processed matrices/metadata; map tumor, immune, stromal labels; test CXCL9/10-CXCR3, SPP1-CD44, PDCD1-CD274 controls",
    },
    {
        "platform": "merfish_or_seqfish_or_slideseqv2",
        "dataset_name": "Vizgen MERSCOPE FFPE Human Immuno-oncology Data Release",
        "source_url": "https://vizgen.com/human-ffpe-immunooncology-release-roadmap/",
        "provider": "Vizgen",
        "tissue": "human cancer samples",
        "technology": "MERSCOPE / MERFISH",
        "reported_scale": "9 million cells across 16 human cancer samples",
        "access_status": "public_release_program_or_registration",
        "phase7_use": "optional imaging-based high-resolution platform if access is available",
        "download_priority": "medium",
        "local_status": "not_found_locally",
        "next_action": "check data release access and whether processed cell-by-gene plus coordinates can be downloaded without manual approval",
    },
    {
        "platform": "merfish_or_seqfish_or_slideseqv2",
        "dataset_name": "Slide-seqV2 mouse hippocampus puck (Open Problems / Zenodo spatial)",
        "source_url": "https://openproblems.bio/datasets/zenodo_spatial/slideseqv2/mouse_hippocampus_puck",
        "provider": "Open Problems / Zenodo",
        "tissue": "mouse hippocampus",
        "technology": "Slide-seqV2",
        "reported_scale": "near-cellular-resolution Slide-seqV2 puck",
        "access_status": "public_processed_anndata",
        "phase7_use": "supplementary cross-platform technical scalability only; weaker biology fit for human CCC story",
        "download_priority": "low",
        "local_status": "not_found_locally",
        "next_action": "use only if human priority platforms remain unavailable",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase7_functional_cross_platform_validation.yaml",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    goal_cfg = config["goals"]["goal5_cross_platform_real_data"]
    goal_dir = PROJECT_ROOT / goal_cfg["output_dir"]
    goal_dir.mkdir(parents=True, exist_ok=True)

    platform_manifest = _platform_manifest(goal_cfg)
    acquisition_plan = _acquisition_plan()
    standardization_readiness = _standardization_readiness()
    validation_summary = _validation_summary()
    control_panel = _control_panel()
    runtime_summary = _runtime_summary(validation_summary)
    source_index = _source_index()

    platform_manifest.to_csv(goal_dir / "phase7_cross_platform_dataset_manifest.csv", index=False)
    acquisition_plan.to_csv(goal_dir / "phase7_cross_platform_acquisition_plan.csv", index=False)
    standardization_readiness.to_csv(goal_dir / "phase7_cross_platform_standardization_readiness.csv", index=False)
    control_panel.to_csv(goal_dir / "phase7_cross_platform_control_panel.csv", index=False)
    validation_summary.to_csv(goal_dir / "phase7_cross_platform_validation_summary.csv", index=False)
    runtime_summary.to_csv(goal_dir / "phase7_cross_platform_runtime_summary.csv", index=False)
    source_index.to_csv(goal_dir / "phase7_cross_platform_source_tables.csv", index=False)

    manifest = {
        "config": _rel(config_path),
        "goal_dir": _rel(goal_dir),
        "priority_platforms_available": int(
            (
                (platform_manifest["priority_status"] == "phase7_priority")
                & (platform_manifest["availability_status"] == "available")
            ).sum()
        ),
        "priority_download_candidates": int((acquisition_plan["download_priority"].isin(["high", "medium"])).sum()),
        "priority_standardization_checks": int(len(standardization_readiness)),
        "priority_standard_schema_ready": int(
            standardization_readiness.get("standard_schema_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()
        ),
        "non_priority_high_resolution_runs": int((validation_summary["validation_status"] == "initial_supported").sum()),
        "standardized_xenium_subsets": int((validation_summary["dataset"] == "xenium_human_lymph_node_subset").sum()),
        "supplemental_lung_visium_subsets": int((validation_summary["dataset"] == "lung_cancer_visium_subset").sum()),
        "post_xenium_colon_subsets": int(
            (validation_summary["dataset"] == "post_xenium_human_colon_cancer_subset").sum()
        ),
        "visium_hd_crc_p2crc_subsets": int(
            (validation_summary["dataset"] == "visium_hd_crc_p2crc_subset").sum()
        ),
        "claim_boundary": (
            "Current Phase 7 evidence includes a Slide-tags tonsil high-resolution subset with controls, "
            "baseline-matched outputs, modules, and runtime. It includes a standardized 10x Xenium human lymph-node "
            "subset with real coordinates/expression, five baseline methods, within-panel downstream validation, "
            "and 50-permutation prediction-level FDR; this remains mixed evidence because high-expression negative "
            "controls rank and correlate strongly. It also includes a supplemental lung-cancer Visium h5ad subset "
            "with SpatialSelf/COMMOT workflow evidence, currently negative-control-risk evidence rather than a "
            "successful discovery case. A 10x post-Xenium colon cancer Visium CytAssist subset now records an "
            "important calibration stress test: default SpatialSelf and COMMOT rank high-expression negative "
            "controls first, whereas LR-prior/mechanism/niche calibrated SpatialSelf prioritizes CRC/TME positive "
            "controls, suppresses negative top ranks, and has downstream support without negative-control FDR "
            "leakage; prediction-level FDR remains conservative/no-recovery. "
            "True Visium HD, CosMx, MERFISH, seqFISH, and Slide-seqV2 remain acquisition candidates."
        ),
        "outputs": {
            "dataset_manifest": _rel(goal_dir / "phase7_cross_platform_dataset_manifest.csv"),
            "acquisition_plan": _rel(goal_dir / "phase7_cross_platform_acquisition_plan.csv"),
            "standardization_readiness": _rel(goal_dir / "phase7_cross_platform_standardization_readiness.csv"),
            "control_panel": _rel(goal_dir / "phase7_cross_platform_control_panel.csv"),
            "validation_summary": _rel(goal_dir / "phase7_cross_platform_validation_summary.csv"),
            "runtime_summary": _rel(goal_dir / "phase7_cross_platform_runtime_summary.csv"),
            "source_tables": _rel(goal_dir / "phase7_cross_platform_source_tables.csv"),
        },
    }
    (goal_dir / "phase7_goal5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(platform_manifest.to_string(index=False))
    print(acquisition_plan.to_string(index=False))
    print(standardization_readiness.to_string(index=False))
    print(validation_summary.to_string(index=False))


def _platform_manifest(goal_cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for platform in goal_cfg["priority_platforms"]:
        hints = PRIORITY_PLATFORM_PATH_HINTS.get(platform, [platform])
        matches = _find_path_matches(hints)
        has_standard_schema = _has_standard_schema_match(matches)
        rows.append(
            {
                "platform": platform,
                "priority_status": "phase7_priority",
                "availability_status": "available" if matches else "not_found_locally",
                "matched_paths": ";".join(matches[:20]),
                "evaluation_status": (
                    "not_started_requires_data"
                    if not matches
                    else "standardized_ready_for_model_run"
                    if has_standard_schema
                    else "available_requires_standardization"
                ),
                "note": (
                    "No local standardized Phase 7 input found."
                    if not matches
                    else "Standard schema found; run model, controls, downstream, FDR, and runtime panels."
                    if has_standard_schema
                    else "Candidate local paths found; requires parser/manifest standardization."
                ),
            }
        )
    rows.append(
        {
            "platform": "slide_tags_tonsil_subset",
            "priority_status": "available_non_priority_high_resolution",
            "availability_status": "available",
            "matched_paths": "high_spatial/data;outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated",
            "evaluation_status": "initial_subset_evaluated",
            "note": "Used as current cross-platform/high-resolution evidence while priority platforms are pending.",
        }
    )
    rows.append(
        {
            "platform": "spamosaic_tonsil_A1_subset",
            "priority_status": "available_cross_platform_rna_adt",
            "availability_status": "available",
            "matched_paths": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_subset",
            "evaluation_status": "five_method_panel_with_mixed_functional_fdr",
            "note": "SPAMOSAIC tonsil RNA+ADT subset has SpatialSelf/COMMOT/CellChat/SpaTalk/CellNEST predictions, downstream proxy validation, and targeted FDR; labels are ADT marker-rule proxies.",
        }
    )
    rows.append(
        {
            "platform": "spamosaic_tonsil_D1_subset",
            "priority_status": "available_cross_platform_rna_adt",
            "availability_status": "available",
            "matched_paths": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_subset",
            "evaluation_status": "five_method_panel_with_mixed_functional_fdr",
            "note": "Independent SPAMOSAIC tonsil RNA+ADT section D1 has SpatialSelf/COMMOT/CellChat/SpaTalk/CellNEST predictions, downstream proxy validation, receiver-restricted residual downstream validation, and targeted FDR; labels are ADT marker-rule proxies.",
        }
    )
    rows.append(
        {
            "platform": "native_human_melanoma_multiome_subset",
            "priority_status": "available_native_real_sample_diagnostic",
            "availability_status": "available",
            "matched_paths": "HumanMelannomaMultiome/data;outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset",
            "evaluation_status": "native_real_sample_completed_negative_control_leakage",
            "note": "Native HumanMelanomaMultiome workflow completed with real cells, coordinates, cell types, expression, SpatialSelf/COMMOT/CellChat/CellNEST predictions, competitive downstream validation, and targeted FDR; SpatialSelf ranks PDCD1-CD274 first but high-expression negative controls leak, so this is diagnostic rather than closure.",
        }
    )
    return pd.DataFrame(rows)


def _acquisition_plan() -> pd.DataFrame:
    return pd.DataFrame(ACQUISITION_CANDIDATES)


def _standardization_readiness() -> pd.DataFrame:
    rows = []
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    for dataset_id in PRIORITY_STANDARDIZATION_IDS:
        path = goal_dir / f"{dataset_id}__standardization_readiness.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                row = frame.iloc[0].to_dict()
                row["dataset_id"] = dataset_id
                row["readiness_table"] = _rel(path)
                row["contract_table"] = _rel(goal_dir / f"{dataset_id}__standardization_contract.json")
                rows.append(row)
                continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "platform": dataset_id,
                "readiness_status": "readiness_check_not_run",
                "recommended_next_action": "run scripts/phase7/prepare_goal5_cross_platform_dataset.py for this platform",
                "standard_schema_present": False,
                "standard_schema_valid": False,
                "readiness_table": _rel(path),
                "contract_table": _rel(goal_dir / f"{dataset_id}__standardization_contract.json"),
            }
        )
    return pd.DataFrame(rows)


def _validation_summary() -> pd.DataFrame:
    scaling_summary_path = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_summary.csv"
    n1800_summary_path = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/tonsil_fresh_goal4_summary.csv"
    baseline_summary_path = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_summary.csv"
    module_overview_path = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_overview.csv"

    scaling = _read_first(scaling_summary_path)
    n1800 = _read_first(n1800_summary_path)
    baseline = _read_first(baseline_summary_path)
    module = _read_first(module_overview_path)
    tonsil_downstream = _slide_tags_tonsil_downstream_summary()
    tonsil_targeted_fdr = _slide_tags_tonsil_targeted_fdr_summary()
    tonsil_context_model_rerun = _slide_tags_tonsil_context_restricted_model_rerun_summary()

    rows = [
        {
                "dataset": "slide_tags_tonsil_subset",
                "platform": "Slide-tags / high-resolution tonsil",
                "validation_status": (
                    "high_resolution_functional_support_negative_suppressed_but_fdr_no_recovery"
                    if tonsil_downstream and tonsil_targeted_fdr
                    else "initial_supported"
                    if n1800
                    else "missing"
                ),
                "n_cells_max": _value(scaling, "n_cells_max"),
                "n_lr_pairs": _value(n1800, "n_lr_pairs"),
                "n_candidate_edges_max": _value(scaling, "n_candidate_edges_max"),
                "positive_control_ranking": f"best={_value(n1800, 'best_positive_control')};rank={_value(n1800, 'best_positive_control_rank')};top_decile={_value(n1800, 'n_positive_controls_top_decile')}",
                "negative_control_ranking": f"best={_value(n1800, 'best_negative_control')};rank={_value(n1800, 'best_negative_control_rank')};top_decile={_value(n1800, 'n_negative_controls_top_decile')}",
                "region_or_niche_enrichment": _value(n1800, "top_cell_type_pairs"),
                "downstream_target_response": tonsil_downstream.get(
                    "downstream_target_response", "not_yet_available_for_high_resolution_subset"
                ),
                "fdr_or_feasibility_note": tonsil_targeted_fdr.get(
                    "fdr_or_feasibility_note",
                    "control-ranking and module diagnostics available; formal permutation FDR not yet run for high-resolution subset",
                )
                + (
                    f" {tonsil_context_model_rerun.get('context_restricted_model_rerun_note', '')}"
                    if tonsil_context_model_rerun
                    else ""
                ),
                "baseline_status": f"methods_completed={_value(baseline, 'n_methods_completed')};rank_validation_pass={_value(baseline, 'n_methods_rank_validation_pass')};best={_value(baseline, 'best_rank_validation_method')}",
                "module_status": f"modules={_value(module, 'n_modules')};positive_modules={_value(module, 'n_modules_with_positive_controls')};best_module={_value(module, 'best_positive_module')}",
                "runtime_memory": f"largest_wall_clock_seconds={_value(scaling, 'largest_run_wall_clock_seconds')};peak_rss_mb={_value(scaling, 'largest_run_peak_rss_mb')}",
                "claim_boundary": (
                    "Real high-resolution coordinates/expression are preserved. CCL21-CCR7 downstream support and "
                    "negative-control suppression are available, but spatial/receiver FDR remains conservative."
                ),
        }
    ]

    xenium_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "xenium_human_lymph_node_subset/standardization_summary.csv"
    )
    xenium = _read_first(xenium_summary_path)
    if xenium:
        xenium_model = _xenium_model_summary()
        rows.append(
            {
                "dataset": "xenium_human_lymph_node_subset",
                "platform": "10x Xenium human lymph node",
                "validation_status": xenium_model.get("validation_status", "standardized_ready_for_model_run"),
                "n_cells_max": _value(xenium, "n_cells"),
                "n_lr_pairs": _value(xenium, "n_lr_pairs"),
                "n_candidate_edges_max": _value(xenium, "n_candidate_interactions"),
                "positive_control_ranking": xenium_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": xenium_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "marker-rule proxy niches available; not curated region annotation",
                "downstream_target_response": xenium_model.get("downstream_target_response", "not_yet_run"),
                "fdr_or_feasibility_note": xenium_model.get(
                    "fdr_or_feasibility_note",
                    "ready standard schema; model predictions, controls, downstream, and FDR pending",
                ),
                "baseline_status": xenium_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": xenium_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(xenium, "claim_boundary"),
            }
        )
    lung_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "lung_cancer_visium_subset/standardization_summary.csv"
    )
    lung = _read_first(lung_summary_path)
    if lung:
        lung_model = _lung_model_summary()
        rows.append(
            {
                "dataset": "lung_cancer_visium_subset",
                "platform": "10x Visium lung cancer h5ad / supplemental real tissue",
                "validation_status": lung_model.get("validation_status", "standardized_ready_for_model_run"),
                "n_cells_max": _value(lung, "n_cells"),
                "n_lr_pairs": _value(lung, "n_lr_pairs"),
                "n_candidate_edges_max": _value(lung, "n_candidate_interactions"),
                "positive_control_ranking": lung_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": lung_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "cell2location abundance proxy niches available; not manual pathology annotation",
                "downstream_target_response": "not_yet_run",
                "fdr_or_feasibility_note": lung_model.get(
                    "fdr_or_feasibility_note",
                    "ready standard schema; model predictions, controls, downstream, and FDR pending",
                ),
                "baseline_status": lung_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": lung_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(lung, "claim_boundary"),
            }
        )
    colon_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "post_xenium_human_colon_cancer_subset/standardization_summary.csv"
    )
    colon = _read_first(colon_summary_path)
    if colon:
        colon_model = _post_xenium_colon_model_summary()
        rows.append(
            {
                "dataset": "post_xenium_human_colon_cancer_subset",
                "platform": "10x Visium CytAssist post-Xenium human colon cancer FFPE",
                "validation_status": colon_model.get("validation_status", "standardized_ready_for_model_run"),
                "n_cells_max": _value(colon, "n_cells"),
                "n_lr_pairs": _value(colon, "n_lr_pairs"),
                "n_candidate_edges_max": _value(colon, "n_candidate_interactions"),
                "positive_control_ranking": colon_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": colon_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "marker-rule epithelial, stromal, myeloid, immune, and vascular proxy niches available; not curated pathology annotation",
                "downstream_target_response": colon_model.get("downstream_target_response", "not_yet_run"),
                "fdr_or_feasibility_note": colon_model.get(
                    "fdr_or_feasibility_note",
                    "ready standard schema; model predictions, controls, downstream, and FDR pending",
                ),
                "baseline_status": colon_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": colon_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(colon, "claim_boundary"),
            }
        )
    visium_hd_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "visium_hd_crc_p2crc_subset/standardization_summary.csv"
    )
    visium_hd = _read_first(visium_hd_summary_path)
    if visium_hd:
        visium_hd_model = _visium_hd_crc_model_summary()
        rows.append(
            {
                "dataset": "visium_hd_crc_p2crc_subset",
                "platform": "10x Visium HD human colon cancer P2CRC 8um bins",
                "validation_status": visium_hd_model.get("validation_status", "standardized_ready_for_model_run"),
                "n_cells_max": _value(visium_hd, "n_cells"),
                "n_lr_pairs": _value(visium_hd, "n_lr_pairs"),
                "n_candidate_edges_max": _value(visium_hd, "n_candidate_interactions"),
                "positive_control_ranking": visium_hd_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": visium_hd_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "GEO deconvolution labels and Periphery/MacrophageSubtype metadata define tumor, stromal, vascular, immune, and myeloid niches.",
                "downstream_target_response": visium_hd_model.get("downstream_target_response", "not_yet_run"),
                "fdr_or_feasibility_note": visium_hd_model.get(
                    "fdr_or_feasibility_note",
                    "ready standard schema; model predictions, controls, downstream, and FDR pending",
                ),
                "baseline_status": visium_hd_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": visium_hd_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(visium_hd, "claim_boundary"),
            }
        )
    spamosaic_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "spamosaic_tonsil_A1_subset/standardization_summary.csv"
    )
    spamosaic = _read_first(spamosaic_summary_path)
    if spamosaic:
        spamosaic_model = _spamosaic_tonsil_model_summary()
        rows.append(
            {
                "dataset": "spamosaic_tonsil_A1_subset",
                "platform": "SPAMOSAIC tonsil RNA+ADT",
                "validation_status": spamosaic_model.get("validation_status", "standardized_ready_for_model_run"),
                "n_cells_max": _value(spamosaic, "n_cells"),
                "n_lr_pairs": _value(spamosaic, "n_lr_pairs"),
                "n_candidate_edges_max": _value(spamosaic, "n_candidate_interactions"),
                "positive_control_ranking": spamosaic_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": spamosaic_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "ADT marker-rule T, B, Tfh, FDC, myeloid, stromal, and epithelial proxy niches available; not curated histology.",
                "downstream_target_response": spamosaic_model.get("downstream_target_response", "not_yet_run"),
                "fdr_or_feasibility_note": spamosaic_model.get(
                    "fdr_or_feasibility_note",
                    "ready standard schema; model predictions, controls, downstream, and FDR pending",
                ),
                "baseline_status": spamosaic_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": spamosaic_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(spamosaic, "claim_boundary"),
            }
        )
    spamosaic_d1_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "spamosaic_tonsil_D1_subset/standardization_summary.csv"
    )
    spamosaic_d1 = _read_first(spamosaic_d1_summary_path)
    if spamosaic_d1:
        spamosaic_d1_model = _spamosaic_tonsil_section_summary("spamosaic_tonsil_D1")
        rows.append(
            {
                "dataset": "spamosaic_tonsil_D1_subset",
                "platform": "SPAMOSAIC tonsil RNA+ADT independent section D1",
                "validation_status": spamosaic_d1_model.get(
                    "validation_status",
                    "standardized_ready_for_model_run",
                ),
                "n_cells_max": _value(spamosaic_d1, "n_cells"),
                "n_lr_pairs": _value(spamosaic_d1, "n_lr_pairs"),
                "n_candidate_edges_max": _value(spamosaic_d1, "n_candidate_interactions"),
                "positive_control_ranking": spamosaic_d1_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": spamosaic_d1_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "ADT marker-rule proxy niches available; independent SPAMOSAIC section replication, not curated histology.",
                "downstream_target_response": spamosaic_d1_model.get("downstream_target_response", "not_yet_run"),
                "fdr_or_feasibility_note": spamosaic_d1_model.get(
                    "fdr_or_feasibility_note",
                    "ready standard schema; model predictions, controls, downstream, and FDR pending",
                ),
                "baseline_status": spamosaic_d1_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": spamosaic_d1_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(spamosaic_d1, "claim_boundary"),
            }
        )
    native_summary_path = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/"
        / "native_human_melanoma_multiome_subset/standardization_summary.csv"
    )
    native = _read_first(native_summary_path)
    if native:
        native_model = _native_melanoma_multiome_summary()
        rows.append(
            {
                "dataset": "native_human_melanoma_multiome_subset",
                "platform": "Slide-tags / 10x Multiome human melanoma lymph-node metastasis",
                "validation_status": native_model.get(
                    "validation_status",
                    "native_real_sample_completed_negative_control_leakage",
                ),
                "n_cells_max": _value(native, "n_cells"),
                "n_lr_pairs": _value(native, "n_lr_pairs"),
                "n_candidate_edges_max": _value(native, "n_candidate_interactions"),
                "positive_control_ranking": native_model.get("positive_control_ranking", "not_yet_run"),
                "negative_control_ranking": native_model.get("negative_control_ranking", "not_yet_run"),
                "region_or_niche_enrichment": "real cell-type labels plus local tumor/immune/myeloid neighborhood-derived niches; no native CCC ground truth.",
                "downstream_target_response": native_model.get("downstream_target_response", "not_yet_run"),
                "fdr_or_feasibility_note": native_model.get("fdr_or_feasibility_note", "not_yet_run"),
                "baseline_status": native_model.get("baseline_status", "not_yet_run"),
                "module_status": "not_yet_run",
                "runtime_memory": native_model.get("runtime_memory", "not_yet_run"),
                "claim_boundary": _value(native, "claim_boundary"),
            }
        )
    melanoma_targeted = _melanoma_real_background_targeted_summary()
    if melanoma_targeted:
        rows.append(melanoma_targeted)
    cosmx = _cosmx_availability_summary()
    if cosmx:
        rows.append(cosmx)
    return pd.DataFrame(rows)


def _cosmx_availability_summary() -> dict[str, Any]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    readiness_path = goal_dir / "cosmx_melanoma__standardization_readiness.csv"
    if not readiness_path.exists():
        return {}
    readiness = pd.read_csv(readiness_path)
    if readiness.empty:
        return {}
    row = readiness.iloc[0].to_dict()
    status = str(row.get("readiness_status", ""))
    invalid_files = str(row.get("invalid_raw_files", ""))
    missing = str(row.get("missing_standard_files", ""))
    if status == "ready_standard_schema":
        validation_status = "standardized_ready_for_model_run"
        feasibility = "CosMx standard schema is present; model/baseline/downstream/FDR run is pending."
    elif "invalid_download" in status or invalid_files:
        validation_status = "blocked_invalid_local_download_placeholders"
        dryad_note = _cosmx_dryad_access_note(goal_dir)
        feasibility = (
            "CosMx melanoma local files are not usable expression/metadata tables. The detected raw files contain "
            "an Unauthorized/bearer-token error response, so SpatialSelf and baselines cannot be run until the "
            "processed expression matrix, cell metadata, and FOV/coordinate files are downloaded again or exported "
            "to the standard schema. "
            f"{dryad_note}"
        )
    else:
        validation_status = "blocked_requires_raw_bundle_or_standard_schema"
        feasibility = (
            "CosMx melanoma remains a priority acquisition target, but the current local input does not expose the "
            f"required standard files ({missing}) or a recognized raw CosMx bundle."
        )
    return {
        "dataset": "cosmx_melanoma",
        "platform": "CosMx RNA-SMI melanoma",
        "validation_status": validation_status,
        "n_cells_max": "",
        "n_lr_pairs": "",
        "n_candidate_edges_max": "",
        "positive_control_ranking": "not_run_no_valid_expression_matrix",
        "negative_control_ranking": "not_run_no_valid_expression_matrix",
        "region_or_niche_enrichment": "not_run_requires_cell_metadata_and_coordinates",
        "downstream_target_response": "not_run_requires_expression_matrix",
        "fdr_or_feasibility_note": feasibility,
        "baseline_status": "not_run_data_availability_blocker",
        "module_status": "not_run",
        "runtime_memory": "not_run",
        "claim_boundary": (
            "This is a data availability/status audit only. It should be used to document acquisition status, "
            "not as evidence for or against SpatialSelf biological performance."
        ),
    }


def _melanoma_real_background_targeted_summary() -> dict[str, Any]:
    screen_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen"
    )
    audit = _read_optional_table(screen_dir / "phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv")
    candidates = _read_optional_table(screen_dir / "phase7_goal4_expanded_melanoma_spatialself_specific_candidates.csv")
    manifest = _read_json(screen_dir / "phase7_goal4_melanoma_targeted_fdr_audit_200perm_manifest.json")
    rerun_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        "melanoma_real_background_model_rerun_fdr_20perm"
    )
    rerun_control = _read_optional_table(rerun_dir / "post_xenium_colon_model_rerun_control_fdr_summary.csv")
    rerun_audit = _read_optional_table(rerun_dir / "post_xenium_colon_model_rerun_fdr_audit.csv")
    matched_context_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        "melanoma_real_background_matched_context_enrichment_fdr"
    )
    matched_context_control = _read_optional_table(
        matched_context_dir / "melanoma_real_background_matched_context_control_fdr_summary.csv"
    )
    matched_context_audit = _read_optional_table(
        matched_context_dir / "melanoma_real_background_matched_context_fdr_audit.csv"
    )
    hardneg_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        "melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm"
    )
    hardneg_control = _read_optional_table(
        hardneg_dir / "post_xenium_colon_model_rerun_control_fdr_summary.csv"
    )
    hardneg_audit = _read_optional_table(
        hardneg_dir / "post_xenium_colon_model_rerun_fdr_audit.csv"
    )
    if audit.empty:
        return {}
    strict = audit[audit.get("strict_goal4_success", pd.Series(dtype=bool)).fillna(False).astype(bool)].copy()
    if strict.empty:
        return {}
    strict_pairs = strict["canonical_pair"].astype(str).tolist()
    best = strict.sort_values(["raw_spatialself_rank", "downstream_bh_fdr"]).iloc[0]
    candidate_rows = candidates[
        candidates.get("canonical_pair", pd.Series(dtype=str)).astype(str).isin(strict_pairs)
    ] if not candidates.empty else pd.DataFrame()
    baseline_fragments = []
    for row in strict.itertuples(index=False):
        baseline_fragments.append(
            f"{row.canonical_pair}:SpatialSelf_rank{_fmt_int(getattr(row, 'raw_spatialself_rank', ''))},"
            f"COMMOT_rank{_fmt_int(getattr(row, 'commot_rank', ''))},"
            f"CellChat_rank{_fmt_int(getattr(row, 'cellchat_rank', ''))},"
            f"CellNEST_rank{_fmt_int(getattr(row, 'cellnest_rank', ''))}"
        )
    negative_controls = []
    if not candidates.empty:
        neg = candidates[
            candidates.get("false_positive_high_expression", pd.Series(dtype=bool)).fillna(False).astype(bool)
        ]
        for row in neg.itertuples(index=False):
            negative_controls.append(
                f"{getattr(row, 'canonical_pair', '')}:SpatialSelf_rank"
                f"{_fmt_int(getattr(row, 'spatialself_rank', ''))}"
            )
    return {
        "dataset": "melanoma_real_background_targeted_context_screen",
        "platform": "Human melanoma real-background targeted context screen",
        "validation_status": "targeted_real_background_strict_support_not_native_goal5_closure",
        "n_cells_max": "real-background constructed subset",
        "n_lr_pairs": int(audit["lr_id"].nunique()),
        "n_candidate_edges_max": "see constructed_replicate/candidate_interactions.csv",
        "positive_control_ranking": ";".join(baseline_fragments),
        "negative_control_ranking": (
            ";".join(negative_controls[:6]) if negative_controls else "high-expression controls suppressed in targeted audit"
        ),
        "region_or_niche_enrichment": (
            "strict targeted context="
            f"{best.get('target_context', manifest.get('target_context', ''))};"
            f"dominant_context={best.get('calibrated_dominant_context', '')};"
            f"top10_both_rule_rate={_fmt_float(best.get('calibrated_top10_both_rule_rate', ''))}"
        ),
        "downstream_target_response": (
            f"strict_pairs={','.join(strict_pairs)};"
            f"best={best.get('canonical_pair', '')}:rho{_fmt_float(best.get('downstream_spearman_r', ''))},"
            f"FDR{_fmt_float(best.get('downstream_bh_fdr', ''))}"
        ),
        "fdr_or_feasibility_note": (
            f"Targeted 200-permutation FDR supports {len(strict_pairs)} pre-specified CXCR3-axis candidates: "
            f"{','.join(strict_pairs)}. Best targeted FDRs for {best.get('canonical_pair', '')}: "
            f"spatial={_fmt_float(best.get('targeted_spatial_bh_fdr', ''))}, "
            f"receiver={_fmt_float(best.get('targeted_receiver_bh_fdr', ''))}, "
            f"LR-identity={_fmt_float(best.get('targeted_lr_identity_bh_fdr', ''))}. "
            f"Model-rerun diagnostic={_melanoma_rerun_note(rerun_control, rerun_audit)} "
            f"Matched-context rerun diagnostic={_melanoma_matched_context_note(matched_context_control, matched_context_audit)} "
            f"Hard-negative/feasibility-gate smoke={_melanoma_rerun_note(hardneg_control, hardneg_audit)} "
            "This is strong real-background targeted context support, but it is not native real-sample CCC ground truth "
            "and is not counted as strict Goal 5 all-null/model-rerun closure."
        ),
        "baseline_status": (
            "SpatialSelf-specific strict candidates pass rank <=10 while COMMOT/CellChat/CellNEST are >10; "
            f"candidate_rows={len(candidate_rows)}; n_strict_success={manifest.get('n_strict_success', len(strict_pairs))}"
        ),
        "module_status": "not_yet_run",
        "runtime_memory": (
            f"time_log={_rel(screen_dir.parent / 'goal4_expanded_melanoma_subset_screen_time.log')};"
            f"fdr_time_log={_rel(screen_dir.parent / 'goal4_expanded_melanoma_subset_screen_time_rerun.log')}"
        ),
        "claim_boundary": (
            "Use as targeted real-background support for weak/context-specific melanoma CCC. Do not present as native "
            "real-sample ground truth or as strict Goal 5 cross-platform closure."
        ),
    }


def _melanoma_rerun_note(control: pd.DataFrame, audit: pd.DataFrame) -> str:
    if control.empty or audit.empty:
        return "not_run."
    status = str(audit.iloc[0].get("model_rerun_all_null_status", ""))
    fragments = [
        f"{row.null_model}:{row.control_status}"
        for row in control.itertuples(index=False)
        if hasattr(row, "null_model") and hasattr(row, "control_status")
    ]
    return f"{status} ({';'.join(fragments)})."


def _melanoma_matched_context_note(control: pd.DataFrame, audit: pd.DataFrame) -> str:
    if control.empty or audit.empty:
        return "not_run."
    status = str(audit.iloc[0].get("matched_context_fdr_status", ""))
    fragments = [
        f"{row.null_model}:{row.control_status}"
        for row in control.itertuples(index=False)
        if hasattr(row, "null_model") and hasattr(row, "control_status")
    ]
    return f"{status} ({';'.join(fragments)})."


def _native_melanoma_multiome_summary() -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    native_dir = goal_dir / "native_human_melanoma_multiome"
    ranking = _read_optional_table(native_dir / "native_melanoma_control_ranking.csv")
    downstream = _read_optional_table(native_dir / "native_melanoma_competitive_downstream_summary.csv")
    residual = _read_optional_table(
        native_dir
        / "residual_downstream_200perm/native_melanoma_residual_downstream_method_summary.csv"
    )
    residual_claims = _read_optional_table(
        native_dir
        / "residual_downstream_200perm/native_melanoma_residual_downstream_claims.csv"
    )
    claim = _read_optional_table(native_dir / "native_melanoma_claim_audit.csv")
    status = _read_optional_table(native_dir / "native_melanoma_baseline_status.csv")
    manifest = _read_json(native_dir / "native_melanoma_validation_manifest.json")
    spatialself = _find_row(ranking, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(ranking, "method", "commot")
    claim_row = _find_row(claim, "method", "spatialself_wavelet_edge_prior")
    downstream_row = _find_row(downstream, "method", "spatialself_wavelet_edge_prior")
    residual_row = _find_row(residual, "method", "spatialself_wavelet_edge_prior")
    residual_claim = _find_row(residual_claims, "method", "spatialself_wavelet_edge_prior")
    pathway_note = _pathway_specific_downstream_note(goal_dir, "native_human_melanoma_multiome")
    methods_completed = ";".join(
        status.loc[status.get("status", pd.Series(dtype=str)).astype(str).eq("completed"), "method"].astype(str).tolist()
    ) if not status.empty and "method" in status else ""
    return {
        "validation_status": "native_real_sample_completed_negative_control_leakage",
        "positive_control_ranking": (
            "SpatialSelf="
            f"{spatialself.get('best_positive', '')}:rank{_fmt_int(spatialself.get('best_positive_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_positive', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))}"
        ),
        "negative_control_ranking": (
            "SpatialSelf="
            f"{spatialself.get('best_negative', '')}:rank{_fmt_int(spatialself.get('best_negative_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_negative', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))}"
        ),
        "downstream_target_response": (
            f"SpatialSelf best expected={downstream_row.get('best_expected_lr', '')};"
            f"FDR={downstream_row.get('best_expected_bh_fdr', '')};"
            f"negative_downstream_fdr_0_1={downstream_row.get('n_negative_fdr_0_1', '')};"
            f"status={downstream_row.get('competitive_downstream_status', '')}"
            f"; residual best expected={residual_row.get('best_expected_lr', '')};"
            f"residual_FDR={residual_row.get('best_expected_bh_fdr', '')};"
            f"residual_negative_fdr_0_1={residual_row.get('n_negative_fdr_0_1', '')};"
            f"residual_claim={residual_claim.get('claim_status', '')}"
        ),
        "fdr_or_feasibility_note": (
            "Native melanoma pipeline completed, but strict claim failed: "
            f"{claim_row.get('allowed_claim', 'negative-control leakage or no FDR recovery')}"
            f" {pathway_note}"
        ),
        "baseline_status": f"methods_completed={methods_completed};claim_status={claim_row.get('claim_status', '')}",
        "runtime_memory": (
            f"manifest_runtime_seconds={manifest.get('runtime_seconds', '')};"
            f"peak_rss_mb={manifest.get('peak_rss_mb', '')};"
            f"manifest={_rel(native_dir / 'native_melanoma_validation_manifest.json')}"
        ),
    }


def _cosmx_dryad_access_note(goal_dir: Path) -> str:
    manifest_path = goal_dir / "cosmx_melanoma_dryad_access_manifest.json"
    file_access_path = goal_dir / "cosmx_melanoma_dryad_file_access.csv"
    if not manifest_path.exists() or not file_access_path.exists():
        return "Dryad access audit has not been run."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        manifest = {}
    files = pd.read_csv(file_access_path)
    n_files = int(manifest.get("n_files_listed", len(files)))
    n_accessible = int(manifest.get("n_files_download_accessible", 0))
    statuses = []
    if not files.empty:
        api_status = sorted(set(files.get("api_download_status", pd.Series(dtype=str)).astype(str)))
        stream_status = sorted(set(files.get("file_stream_status", pd.Series(dtype=str)).astype(str)))
        statuses.append(f"api_download_status={','.join(api_status)}")
        statuses.append(f"file_stream_status={','.join(stream_status)}")
    return (
        f"Dryad API lists {n_files} files for version {manifest.get('version_id', '')}, but "
        f"{n_accessible}/{n_files} file downloads are currently accessible"
        + (f" ({';'.join(statuses)})." if statuses else ".")
    )


def _xenium_model_summary() -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    control_summary_path = goal_dir / "xenium_human_lymph_node_method_control_summary.csv"
    if not control_summary_path.exists():
        return {}
    control_summary = pd.read_csv(control_summary_path)
    spatialself = _find_row(control_summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(control_summary, "method", "commot")
    downstream = _xenium_downstream_summary(goal_dir)
    runtime = _read_optional_table(goal_dir / "xenium_human_lymph_node_method_runtime_summary.csv")
    status = _read_optional_table(goal_dir / "xenium_human_lymph_node_method_status_summary.csv")
    prediction_fdr = _xenium_prediction_fdr_summary(goal_dir)
    zarr_groups = _xenium_analysis_zarr_group_summary(goal_dir)
    graphclust_validation = _xenium_graphclust_validation_summary(goal_dir)
    graphclust_downstream = _xenium_graphclust_residual_downstream_summary(goal_dir)
    rich_target_note = _xenium_rich_target_downstream_note(goal_dir)
    negative_top = [
        f"{row.method}:{_fmt_int(row.best_negative_rank)}"
        for row in control_summary.itertuples(index=False)
        if _safe_int(getattr(row, "best_negative_rank", 999)) <= 3
    ]
    methods_completed = ";".join(str(method) for method in control_summary["method"].dropna().astype(str))
    noninformative = [
        f"{row.method}:{row.rank_validation_status}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")).startswith("noninformative")
    ]
    validation_status = (
        "five_method_panel_with_graphclust_fdr_support_but_mixed_downstream"
        if graphclust_validation and graphclust_downstream
        else
        "five_method_panel_with_mixed_downstream_and_conservative_prediction_fdr"
        if downstream and prediction_fdr
        else "five_method_panel_with_mixed_downstream_control_risk"
        if downstream
        else "model_run_with_control_risk"
    )
    fdr_note = (
        downstream.get("fdr_or_feasibility_note")
        if downstream
        else (
            "SpatialSelf edge-prior and COMMOT model runs completed on the Xenium subset; no native ground truth. "
            f"High-expression negative controls rank near the top for {', '.join(negative_top) if negative_top else 'no completed method'}, "
            "so this is platform/baseline wiring plus risk evidence, not final false-positive suppression evidence. "
            "Downstream target response and permutation/FDR are pending."
        )
    )
    return {
        "validation_status": validation_status,
        "positive_control_ranking": (
            "SpatialSelf="
            f"{spatialself.get('best_positive_control', '')}:rank{_fmt_int(spatialself.get('best_positive_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_positive_control', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))}"
        ),
        "negative_control_ranking": (
            "SpatialSelf="
            f"{spatialself.get('best_negative_control', '')}:rank{_fmt_int(spatialself.get('best_negative_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_negative_control', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))}"
        ),
        "downstream_target_response": downstream.get("downstream_target_response", "not_yet_run"),
        "fdr_or_feasibility_note": (
            f"{fdr_note} Prediction-level FDR: {prediction_fdr.get('prediction_fdr_note', 'not_yet_run')}"
            + (f" {zarr_groups.get('zarr_group_note', '')}" if zarr_groups else "")
            + (f" {graphclust_validation.get('graphclust_validation_note', '')}" if graphclust_validation else "")
            + (f" {graphclust_downstream.get('graphclust_residual_downstream_note', '')}" if graphclust_downstream else "")
            + (f" {rich_target_note}" if rich_target_note else "")
        ),
        "baseline_status": (
            f"methods_completed={methods_completed};"
            f"negative_top3_methods={', '.join(negative_top) if negative_top else 'none'};"
            f"noninformative_methods={', '.join(noninformative) if noninformative else 'none'};"
            f"status_summary={_status_compact(status)}"
        ),
        "runtime_memory": _runtime_compact(runtime)
        if not runtime.empty
        else (
            "SpatialSelf="
            f"{_runtime_from_time_log(goal_dir / 'xenium_human_lymph_node_baseline_panel_time.log')};"
            "COMMOT="
            f"{_runtime_from_time_log(goal_dir / 'xenium_human_lymph_node_commot_time.log')}"
        ),
    }


def _lung_model_summary() -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    control_summary_path = goal_dir / "lung_cancer_visium_method_control_summary.csv"
    if not control_summary_path.exists():
        return {}
    control_summary = pd.read_csv(control_summary_path)
    spatialself = _find_row(control_summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(control_summary, "method", "commot")
    negative_top = [
        f"{row.method}:{_fmt_int(row.best_negative_rank)}"
        for row in control_summary.itertuples(index=False)
        if _safe_int(getattr(row, "best_negative_rank", 999)) <= 3
    ]
    methods_completed = ";".join(str(method) for method in control_summary["method"].dropna().astype(str))
    return {
        "validation_status": "supplemental_visium_lung_mixed_or_negative_control_risk",
        "positive_control_ranking": (
            "SpatialSelf="
            f"{spatialself.get('best_positive_control', '')}:rank{_fmt_int(spatialself.get('best_positive_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_positive_control', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))}"
        ),
        "negative_control_ranking": (
            "SpatialSelf="
            f"{spatialself.get('best_negative_control', '')}:rank{_fmt_int(spatialself.get('best_negative_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_negative_control', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))}"
        ),
        "fdr_or_feasibility_note": (
            "Supplemental lung Visium workflow is standardized and SpatialSelf/COMMOT completed, but high-expression "
            f"negative controls rank near the top ({', '.join(negative_top) if negative_top else 'none'}). "
            "This is useful cross-tissue risk evidence and workflow coverage, not a successful false-positive-control "
            "or weak/context-specific discovery case. Downstream and prediction-level FDR have not yet been run."
        ),
        "baseline_status": f"methods_completed={methods_completed};CellChat/SpaTalk/CellNEST not_yet_run_on_lung_supplemental",
        "runtime_memory": _runtime_from_time_log(goal_dir / "lung_cancer_visium_baseline_panel_time.log"),
    }


def _post_xenium_colon_model_summary() -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    control_summary_path = goal_dir / "post_xenium_human_colon_cancer_method_control_summary.csv"
    if not control_summary_path.exists():
        return {}
    control_summary = pd.read_csv(control_summary_path)
    default = _find_row(control_summary, "method", "spatialself_default")
    calibrated = _find_row(control_summary, "method", "spatialself_calibrated")
    commot = _find_row(control_summary, "method", "commot")
    cellchat = _find_row(control_summary, "method", "cellchat")
    spatalk = _find_row(control_summary, "method", "spatalk")
    cellnest = _find_row(control_summary, "method", "cellnest")
    downstream = _post_xenium_colon_downstream_summary(goal_dir)
    prediction_fdr = _post_xenium_colon_prediction_fdr_summary(goal_dir)
    targeted_region_fdr = _post_xenium_colon_targeted_region_fdr_summary(goal_dir)
    region_aware_fdr = _post_xenium_colon_region_aware_fdr_summary(goal_dir)
    context_model_rerun = _post_xenium_context_restricted_model_rerun_summary(goal_dir)
    pathway_note = _pathway_specific_downstream_note(goal_dir, "post_xenium_colon")
    methods_completed = ";".join(str(method) for method in control_summary["method"].dropna().astype(str))
    default_neg = f"{default.get('best_negative_control', '')}:rank{_fmt_int(default.get('best_negative_rank', ''))}"
    commot_neg = f"{commot.get('best_negative_control', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))}"
    cellchat_neg = f"{cellchat.get('best_negative_control', '')}:rank{_fmt_int(cellchat.get('best_negative_rank', ''))}"
    cellnest_neg = f"{cellnest.get('best_negative_control', '')}:rank{_fmt_int(cellnest.get('best_negative_rank', ''))}"
    calibrated_neg = f"{calibrated.get('best_negative_control', '')}:rank{_fmt_int(calibrated.get('best_negative_rank', ''))}"
    baseline_risk = [
        f"{row.method}:{_fmt_int(row.best_negative_rank)}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")) == "negative_control_ranks_as_high_or_higher"
    ]
    noninformative = [
        f"{row.method}:{row.rank_validation_status}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")).startswith("noninformative")
    ]
    return {
        "validation_status": "functional_support_negative_suppressed_but_fdr_no_recovery",
        "positive_control_ranking": (
            "SpatialSelf_default="
            f"{default.get('best_positive_control', '')}:rank{_fmt_int(default.get('best_positive_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_positive_control', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))};"
            "CellChat="
            f"{cellchat.get('best_positive_control', '')}:rank{_fmt_int(cellchat.get('best_positive_rank', ''))};"
            "CellNEST="
            f"{cellnest.get('best_positive_control', '')}:rank{_fmt_int(cellnest.get('best_positive_rank', ''))};"
            "SpatialSelf_calibrated="
            f"{calibrated.get('best_positive_control', '')}:rank{_fmt_int(calibrated.get('best_positive_rank', ''))}"
        ),
        "negative_control_ranking": (
            f"SpatialSelf_default={default_neg};COMMOT={commot_neg};CellChat={cellchat_neg};"
            f"CellNEST={cellnest_neg};SpatialSelf_calibrated={calibrated_neg}"
        ),
        "downstream_target_response": downstream.get("downstream_target_response", "not_yet_run"),
        "fdr_or_feasibility_note": (
            "Post-Xenium colon cancer is a real 10x Visium CytAssist background stress test. Default SpatialSelf and "
            f"COMMOT both rank high-expression negative controls first ({default_neg}; {commot_neg}), and their top100 "
            "edges are dominated by negative controls. CellChat and CellNEST show the same high-expression negative-control "
            f"risk ({cellchat_neg}; {cellnest_neg}); SpaTalk is noninformative/status-only on this panel. "
            "In contrast, LR-prior/mechanism/niche calibrated SpatialSelf "
            f"prioritizes {calibrated.get('best_positive_control', '')} at rank "
            f"{_fmt_int(calibrated.get('best_positive_rank', ''))}, has best negative {calibrated_neg}, "
            f"positive_top3={_fmt_int(calibrated.get('positive_top3', ''))}, "
            f"negative_top3={_fmt_int(calibrated.get('negative_top3', ''))}, "
            f"positive_top100_edges={_fmt_int(calibrated.get('positive_top100_edges', ''))}, "
            f"negative_top100_edges={_fmt_int(calibrated.get('negative_top100_edges', ''))}. "
            "This supports the calibration mechanism as a candidate false-positive suppressor. "
            f"{downstream.get('downstream_note', 'Downstream target validation has not yet been run.')} "
            f"{prediction_fdr.get('prediction_fdr_note', 'Prediction-level FDR has not yet been run.')} "
            f"{targeted_region_fdr.get('targeted_region_note', '')} "
            f"{region_aware_fdr.get('region_aware_note', '')} "
            f"{context_model_rerun.get('context_restricted_model_rerun_note', '')} "
            f"{pathway_note}"
        ),
        "baseline_status": (
            f"methods_completed={methods_completed};"
            f"negative_control_risk_methods={', '.join(baseline_risk) if baseline_risk else 'none'};"
            f"noninformative_methods={', '.join(noninformative) if noninformative else 'none'};"
            f"default_status={default.get('rank_validation_status', '')};"
            f"commot_status={commot.get('rank_validation_status', '')};"
            f"cellchat_status={cellchat.get('rank_validation_status', '')};"
            f"spatalk_status={spatalk.get('rank_validation_status', '')};"
            f"cellnest_status={cellnest.get('rank_validation_status', '')};"
            f"calibrated_status={calibrated.get('rank_validation_status', '')}"
        ),
        "runtime_memory": (
            "default_panel="
            f"{_runtime_from_time_log(goal_dir / 'post_xenium_human_colon_cancer_baseline_panel_time.log')};"
            "extended_baselines="
            f"{_runtime_from_time_log(goal_dir / 'post_xenium_human_colon_cancer_extended_baselines_time.log')};"
            "calibrated="
            f"{_runtime_from_time_log(goal_dir / 'post_xenium_human_colon_cancer_spatialself_calibrated_time.log')}"
        ),
    }


def _post_xenium_context_restricted_model_rerun_summary(goal_dir: Path) -> dict[str, str]:
    raw_dir = goal_dir / "post_xenium_colon_context_restricted_model_rerun_fdr_raw"
    diagnostic_dir = goal_dir / "post_xenium_colon_context_restricted_model_rerun_fdr_fp_suppressed_diagnostic"
    raw_control = _read_optional_table(raw_dir / "post_xenium_colon_context_restricted_control_fdr_summary.csv")
    raw_audit = _read_optional_table(raw_dir / "post_xenium_colon_context_restricted_model_rerun_fdr_audit.csv")
    diagnostic_control = _read_optional_table(
        diagnostic_dir / "post_xenium_colon_context_restricted_control_fdr_summary.csv"
    )
    if raw_control.empty and diagnostic_control.empty:
        return {}
    raw_status = (
        raw_audit.iloc[0].get("context_restricted_model_rerun_status", "")
        if not raw_audit.empty
        else ""
    )
    raw_notes = [
        f"{row.null_model}:pos={_fmt_float(getattr(row, 'best_positive_group_bh_fdr', ''))},"
        f"neg={_fmt_float(getattr(row, 'best_negative_group_bh_fdr', ''))},"
        f"status={getattr(row, 'control_status', '')}"
        for row in raw_control.itertuples(index=False)
    ]
    diagnostic_notes = [
        f"{row.null_model}:pos={_fmt_float(getattr(row, 'best_positive_group_bh_fdr', ''))},"
        f"neg={_fmt_float(getattr(row, 'best_negative_group_bh_fdr', ''))},"
        f"status={getattr(row, 'control_status', '')}"
        for row in diagnostic_control.itertuples(index=False)
    ]
    return {
        "context_restricted_model_rerun_note": (
            "Context-restricted model-rerun diagnostic reused the existing 50-per-null retrained predictions on "
            "pre-specified targeted candidate edges. "
            f"Raw status={raw_status}; raw nulls [{'; '.join(raw_notes)}]. "
            f"False-positive-suppressed diagnostic also failed to recover positives [{'; '.join(diagnostic_notes)}]. "
            "Thus post-Xenium is not a strict Goal 5 closure candidate under the current model-rerun null."
        )
    }


def _post_xenium_colon_downstream_summary(goal_dir: Path) -> dict[str, str]:
    downstream_dir = goal_dir / "post_xenium_colon_downstream_target_validation_50perm"
    summary_path = downstream_dir / "post_xenium_colon_downstream_method_summary.csv"
    claims_path = downstream_dir / "post_xenium_colon_downstream_claims.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    default = _find_row(summary, "method", "spatialself_default")
    calibrated = _find_row(summary, "method", "spatialself_calibrated")
    commot = _find_row(summary, "method", "commot")
    calibrated_claim = _find_row(claims, "method", "spatialself_calibrated")
    return {
        "downstream_target_response": (
            "SpatialSelf_calibrated="
            f"{_short_lr_id(calibrated.get('best_positive_lr', ''))}:rho{_fmt_float(calibrated.get('best_positive_spearman', ''))},"
            f"FDR{_fmt_float(calibrated.get('best_positive_bh_fdr', ''))},"
            f"neg={_short_lr_id(calibrated.get('best_negative_lr', ''))}:rho{_fmt_float(calibrated.get('best_negative_spearman', ''))},"
            f"FDR{_fmt_float(calibrated.get('best_negative_bh_fdr', ''))};"
            "SpatialSelf_default="
            f"{_short_lr_id(default.get('best_positive_lr', ''))}:rho{_fmt_float(default.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(default.get('best_negative_lr', ''))}:rho{_fmt_float(default.get('best_negative_spearman', ''))};"
            "COMMOT="
            f"{_short_lr_id(commot.get('best_positive_lr', ''))}:rho{_fmt_float(commot.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(commot.get('best_negative_lr', ''))}:rho{_fmt_float(commot.get('best_negative_spearman', ''))}"
        ),
        "downstream_note": (
            "Post-Xenium colon selected-gene downstream validation with 50 permutations shows calibrated SpatialSelf "
            f"best positive {_short_lr_id(calibrated.get('best_positive_lr', ''))} rho="
            f"{_fmt_float(calibrated.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(calibrated.get('best_positive_bh_fdr', ''))}; its best negative "
            f"{_short_lr_id(calibrated.get('best_negative_lr', ''))} has rho="
            f"{_fmt_float(calibrated.get('best_negative_spearman', ''))} and no positive negative-control FDR signal "
            f"(positive_negative_delta={_fmt_float(calibrated.get('positive_negative_delta', ''))}; "
            f"claim={calibrated_claim.get('claim_status', 'not_available')}). In contrast, COMMOT and default SpatialSelf "
            "show strong positive downstream correlations for high-expression negative controls, so they are mixed/risk "
            "evidence on this subset."
        ),
    }


def _post_xenium_colon_prediction_fdr_summary(goal_dir: Path) -> dict[str, str]:
    fdr_dir = goal_dir / "post_xenium_colon_prediction_fdr_50perm"
    control_path = fdr_dir / "post_xenium_colon_prediction_control_fdr_summary.csv"
    status_path = fdr_dir / "post_xenium_colon_prediction_fdr_status.csv"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    n_pos_sig = int(pd.to_numeric(control["n_positive_significant"], errors="coerce").fillna(0).sum())
    n_neg_sig = int(pd.to_numeric(control["n_negative_significant"], errors="coerce").fillna(0).sum())
    methods_completed = int((status.get("status", pd.Series(dtype=str)).astype(str) == "completed").sum())
    min_positive = float(pd.to_numeric(control["best_positive_group_bh_fdr"], errors="coerce").min())
    min_negative = float(pd.to_numeric(control["best_negative_group_bh_fdr"], errors="coerce").min())
    status_counts = control["control_fdr_status"].value_counts().to_dict()
    return {
        "prediction_fdr_note": (
            "Post-Xenium colon 50-permutation prediction-level FDR completed for "
            f"{methods_completed} method/config rows across spatial, receiver, and LR-identity nulls. "
            "No positive controls and no high-expression negative controls are significant at alpha=0.05 "
            f"(positive significant={n_pos_sig}; negative significant={n_neg_sig}; "
            f"min positive group BH-FDR={_fmt_float(min_positive)}; min negative group BH-FDR={_fmt_float(min_negative)}). "
            "This is conservative/no-recovery FDR evidence, not positive-control recovery. "
            f"Control status counts: {status_counts}."
        )
    }


def _post_xenium_colon_targeted_region_fdr_summary(goal_dir: Path) -> dict[str, str]:
    target_dir = goal_dir / "post_xenium_colon_targeted_region_fdr_100perm"
    audit_path = target_dir / "post_xenium_colon_targeted_region_fdr_audit.csv"
    control_path = target_dir / "post_xenium_colon_targeted_control_fdr_summary.csv"
    if not audit_path.exists() or not control_path.exists():
        return {}
    audit = pd.read_csv(audit_path)
    control = pd.read_csv(control_path)
    strict_success = bool(
        (audit.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str)
        == "targeted_positive_significant_negative_suppressed").any()
    )
    lr_identity = _find_row(control, "null_model", "lr_identity_permutation")
    spatial = _find_row(control, "null_model", "spatial_permutation")
    receiver = _find_row(control, "null_model", "receiver_permutation")
    return {
        "targeted_region_note": (
            "Supplemental targeted/region-aware FDR has been run on rule-satisfied positive-prior edges while "
            "retaining all high-expression negative-control edges. It does not replace the main all-candidate "
            f"FDR panel. Strict targeted three-null success={strict_success}. LR-identity null shows "
            f"{lr_identity.get('control_status', 'not_available')} "
            f"(best positive FDR={_fmt_float(lr_identity.get('best_positive_group_bh_fdr', ''))}; "
            f"best negative FDR={_fmt_float(lr_identity.get('best_negative_group_bh_fdr', ''))}), while "
            f"spatial/receiver nulls remain {spatial.get('control_status', 'not_available')}/"
            f"{receiver.get('control_status', 'not_available')}."
        )
    }


def _post_xenium_colon_region_aware_fdr_summary(goal_dir: Path) -> dict[str, str]:
    target_dir = goal_dir / "post_xenium_colon_region_aware_targeted_fdr_200perm"
    control_path = target_dir / "post_xenium_colon_region_aware_targeted_control_fdr_summary.csv"
    manifest_path = target_dir / "post_xenium_colon_region_aware_targeted_fdr_manifest.json"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    strict = False
    if manifest_path.exists():
        try:
            strict = bool(json.loads(manifest_path.read_text(encoding="utf-8")).get("strict_region_aware_success", False))
        except json.JSONDecodeError:
            strict = False
    within = _find_row(control, "null_model", "within_region_permutation")
    cell_type = _find_row(control, "null_model", "cell_type_preserving_permutation")
    identity = _find_row(control, "null_model", "lr_identity_permutation")
    return {
        "region_aware_note": (
            "A supplemental region/cell-type-preserving targeted FDR audit was run to test a more tissue-structure-aware "
            f"null. Strict region-aware success={strict}. Within-region and cell-type-preserving nulls remain "
            f"{within.get('control_status', 'not_available')}/{cell_type.get('control_status', 'not_available')} "
            f"(best positive FDR={_fmt_float(within.get('best_positive_group_bh_fdr', ''))}/"
            f"{_fmt_float(cell_type.get('best_positive_group_bh_fdr', ''))}), while LR-identity remains "
            f"{identity.get('control_status', 'not_available')} with best positive FDR="
            f"{_fmt_float(identity.get('best_positive_group_bh_fdr', ''))}. This strengthens the conservative-FDR boundary "
            "rather than closing strict Goal 5."
        )
    }


def _visium_hd_crc_model_summary() -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    control_summary_path = goal_dir / "visium_hd_crc_p2crc_method_control_summary.csv"
    if not control_summary_path.exists():
        return {}
    control_summary = pd.read_csv(control_summary_path)
    default = _find_row(control_summary, "method", "spatialself_wavelet_edge_prior")
    calibrated = _find_row(control_summary, "method", "spatialself_calibrated")
    commot = _find_row(control_summary, "method", "commot")
    cellchat = _find_row(control_summary, "method", "cellchat")
    cellnest = _find_row(control_summary, "method", "cellnest")
    downstream = _visium_hd_crc_downstream_summary(goal_dir)
    rich_downstream = _visium_hd_crc_rich_target_downstream_summary(goal_dir)
    prediction_fdr = _visium_hd_crc_prediction_fdr_summary(goal_dir)
    region_aware_fdr = _visium_hd_crc_region_aware_fdr_summary(goal_dir)
    pathway_note = _pathway_specific_downstream_note(goal_dir, "visium_hd_crc_p2crc")
    methods_completed = ";".join(str(method) for method in control_summary["method"].dropna().astype(str))
    default_neg = f"{default.get('best_negative_control', '')}:rank{_fmt_int(default.get('best_negative_rank', ''))}"
    calibrated_neg = f"{calibrated.get('best_negative_control', '')}:rank{_fmt_int(calibrated.get('best_negative_rank', ''))}"
    commot_neg = f"{commot.get('best_negative_control', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))}"
    cellchat_neg = f"{cellchat.get('best_negative_control', '')}:rank{_fmt_int(cellchat.get('best_negative_rank', ''))}"
    cellnest_neg = f"{cellnest.get('best_negative_control', '')}:rank{_fmt_int(cellnest.get('best_negative_rank', ''))}"
    baseline_risk = [
        f"{row.method}:{_fmt_int(row.best_negative_rank)}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")) == "negative_control_ranks_as_high_or_higher"
    ]
    return {
        "validation_status": "visium_hd_mixed_ranking_improved_but_downstream_fdr_negative_leakage",
        "positive_control_ranking": (
            "SpatialSelf_default="
            f"{default.get('best_positive_control', '')}:rank{_fmt_int(default.get('best_positive_rank', ''))};"
            "SpatialSelf_calibrated="
            f"{calibrated.get('best_positive_control', '')}:rank{_fmt_int(calibrated.get('best_positive_rank', ''))};"
            "COMMOT="
            f"{commot.get('best_positive_control', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))};"
            "CellChat="
            f"{cellchat.get('best_positive_control', '')}:rank{_fmt_int(cellchat.get('best_positive_rank', ''))};"
            "CellNEST="
            f"{cellnest.get('best_positive_control', '')}:rank{_fmt_int(cellnest.get('best_positive_rank', ''))}"
        ),
        "negative_control_ranking": (
            f"SpatialSelf_default={default_neg};SpatialSelf_calibrated={calibrated_neg};"
            f"COMMOT={commot_neg};CellChat={cellchat_neg};CellNEST={cellnest_neg}"
        ),
        "downstream_target_response": downstream.get("downstream_target_response", "not_yet_run"),
        "fdr_or_feasibility_note": (
            "Visium HD CRC P2CRC is a real 8um high-resolution platform stress test with GEO deconvolution labels. "
            f"Default SpatialSelf and all baseline adapters rank a high-expression negative control first or near first "
            f"({default_neg}; {commot_neg}; {cellchat_neg}; {cellnest_neg}). LR-prior/mechanism/niche calibrated "
            f"SpatialSelf improves ranking ({calibrated.get('best_positive_control', '')} rank "
            f"{_fmt_int(calibrated.get('best_positive_rank', ''))}; best negative {calibrated_neg}; "
            f"negative_top3={_fmt_int(calibrated.get('negative_top3', ''))}; "
            f"negative_top100_edges={_fmt_int(calibrated.get('negative_top100_edges', ''))}). "
            f"{downstream.get('downstream_note', 'Downstream target validation has not yet been run.')} "
            f"{rich_downstream.get('rich_target_downstream_note', '')} "
            f"{prediction_fdr.get('prediction_fdr_note', 'Prediction-level FDR has not yet been run.')} "
            f"{region_aware_fdr.get('region_aware_note', 'Region-aware targeted FDR has not yet been run.')} "
            f"{pathway_note}"
        ),
        "baseline_status": (
            f"methods_completed={methods_completed};"
            f"negative_control_risk_methods={', '.join(baseline_risk) if baseline_risk else 'none'};"
            f"default_status={default.get('rank_validation_status', '')};"
            f"calibrated_status={calibrated.get('rank_validation_status', '')};"
            f"commot_status={commot.get('rank_validation_status', '')};"
            f"cellchat_status={cellchat.get('rank_validation_status', '')};"
            f"cellnest_status={cellnest.get('rank_validation_status', '')}"
        ),
        "runtime_memory": (
            "default_panel="
            f"{_runtime_from_time_log(goal_dir / 'visium_hd_crc_p2crc_baseline_panel_time.log')};"
            "calibrated="
            f"{_runtime_from_time_log(goal_dir / 'visium_hd_crc_p2crc_spatialself_calibrated_time.log')}"
        ),
    }


def _visium_hd_crc_downstream_summary(goal_dir: Path) -> dict[str, str]:
    downstream_dir = goal_dir / "visium_hd_crc_p2crc_downstream_target_validation"
    summary_path = downstream_dir / "visium_hd_crc_p2crc_downstream_method_summary.csv"
    claims_path = downstream_dir / "visium_hd_crc_p2crc_downstream_claims.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    default = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    calibrated = _find_row(summary, "method", "spatialself_calibrated")
    commot = _find_row(summary, "method", "commot")
    cellchat = _find_row(summary, "method", "cellchat")
    cellnest = _find_row(summary, "method", "cellnest")
    calibrated_claim = _find_row(claims, "method", "spatialself_calibrated")
    return {
        "downstream_target_response": (
            "SpatialSelf_calibrated="
            f"{_short_lr_id(calibrated.get('best_positive_lr', ''))}:rho{_fmt_float(calibrated.get('best_positive_spearman', ''))},"
            f"FDR{_fmt_float(calibrated.get('best_positive_bh_fdr', ''))},"
            f"neg={_short_lr_id(calibrated.get('best_negative_lr', ''))}:rho{_fmt_float(calibrated.get('best_negative_spearman', ''))},"
            f"FDR{_fmt_float(calibrated.get('best_negative_bh_fdr', ''))};"
            "COMMOT="
            f"{_short_lr_id(commot.get('best_positive_lr', ''))}:rho{_fmt_float(commot.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(commot.get('best_negative_lr', ''))}:rho{_fmt_float(commot.get('best_negative_spearman', ''))};"
            "CellChat="
            f"{_short_lr_id(cellchat.get('best_positive_lr', ''))}:rho{_fmt_float(cellchat.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(cellchat.get('best_negative_lr', ''))}:rho{_fmt_float(cellchat.get('best_negative_spearman', ''))};"
            "CellNEST="
            f"{_short_lr_id(cellnest.get('best_positive_lr', ''))}:rho{_fmt_float(cellnest.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(cellnest.get('best_negative_lr', ''))}:rho{_fmt_float(cellnest.get('best_negative_spearman', ''))};"
            "SpatialSelf_default="
            f"{_short_lr_id(default.get('best_positive_lr', ''))}:rho{_fmt_float(default.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(default.get('best_negative_lr', ''))}:rho{_fmt_float(default.get('best_negative_spearman', ''))}"
        ),
        "downstream_note": (
            "Visium HD CRC curated within-panel downstream validation shows calibrated SpatialSelf has a positive "
            f"target-response association ({_short_lr_id(calibrated.get('best_positive_lr', ''))}, rho="
            f"{_fmt_float(calibrated.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(calibrated.get('best_positive_bh_fdr', ''))}), but high-expression negative controls also "
            f"correlate (best negative rho={_fmt_float(calibrated.get('best_negative_spearman', ''))}, FDR="
            f"{_fmt_float(calibrated.get('best_negative_bh_fdr', ''))}; claim="
            f"{calibrated_claim.get('claim_status', 'not_available')}). COMMOT/CellChat/CellNEST show stronger "
            "negative-control downstream correlations, so this is relative-improvement but still mixed evidence."
        ),
    }


def _visium_hd_crc_rich_target_downstream_summary(goal_dir: Path) -> dict[str, str]:
    summary_path = (
        goal_dir
        / "visium_hd_crc_p2crc_rich_targets_downstream_validation/"
        "visium_hd_crc_p2crc_downstream_method_summary.csv"
    )
    competitive_path = (
        goal_dir
        / "visium_hd_crc_p2crc_rich_targets_competitive_downstream_200perm/"
        "visium_hd_crc_competitive_downstream_method_summary.csv"
    )
    claims_path = (
        goal_dir
        / "visium_hd_crc_p2crc_rich_targets_competitive_downstream_200perm/"
        "visium_hd_crc_competitive_downstream_claims.csv"
    )
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    spatialself = _find_row(summary, "method", "spatialself_calibrated")
    competitive = pd.read_csv(competitive_path) if competitive_path.exists() else pd.DataFrame()
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    comp_ss = _find_row(claims, "method", "spatialself_calibrated")
    commot = _find_row(claims, "method", "commot")
    return {
        "rich_target_downstream_note": (
            "A richer Visium HD target profile was generated from the original 18,085-gene matrix "
            "(72 retained LR/marker/target genes versus 49 in the compact panel). Rich-target downstream still "
            "does not close Goal 5: calibrated SpatialSelf best positive "
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))} rho="
            f"{_fmt_float(spatialself.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}, but best negative "
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))} also has rho="
            f"{_fmt_float(spatialself.get('best_negative_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}. "
            f"Rich competitive downstream claim for calibrated SpatialSelf={comp_ss.get('claim_status', '')}; "
            f"COMMOT={commot.get('claim_status', '')}. This is boundary evidence, not strict functional success."
        )
    }


def _visium_hd_crc_prediction_fdr_summary(goal_dir: Path) -> dict[str, str]:
    fdr_dir = goal_dir / "visium_hd_crc_p2crc_prediction_fdr_50perm"
    control_path = fdr_dir / "visium_hd_crc_p2crc_prediction_control_fdr_summary.csv"
    status_path = fdr_dir / "visium_hd_crc_p2crc_prediction_fdr_status.csv"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    n_pos_sig = int(pd.to_numeric(control["n_positive_significant"], errors="coerce").fillna(0).sum())
    n_neg_sig = int(pd.to_numeric(control["n_negative_significant"], errors="coerce").fillna(0).sum())
    methods_completed = int((status.get("status", pd.Series(dtype=str)).astype(str) == "completed").sum())
    return {
        "prediction_fdr_note": (
            "Visium HD CRC 50-permutation prediction-level FDR completed for "
            f"{methods_completed} method/config rows across spatial, receiver, and LR-identity nulls. "
            f"No positive or negative controls are significant (positive significant={n_pos_sig}; "
            f"negative significant={n_neg_sig}), so this is conservative/no-recovery FDR evidence."
        )
    }


def _visium_hd_crc_region_aware_fdr_summary(goal_dir: Path) -> dict[str, str]:
    target_dir = goal_dir / "visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm"
    control_path = target_dir / "visium_hd_crc_p2crc_region_aware_targeted_control_fdr_summary.csv"
    audit_path = target_dir / "visium_hd_crc_p2crc_region_aware_targeted_fdr_audit.csv"
    manifest_path = target_dir / "visium_hd_crc_p2crc_region_aware_targeted_fdr_manifest.json"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    strict = False
    if manifest_path.exists():
        try:
            strict = bool(json.loads(manifest_path.read_text(encoding="utf-8")).get("strict_region_aware_success", False))
        except json.JSONDecodeError:
            strict = False
    within = _find_row(control, "null_model", "within_region_permutation")
    cell_type = _find_row(control, "null_model", "cell_type_preserving_permutation")
    identity = _find_row(control, "null_model", "lr_identity_permutation")
    leaked = sorted(
        set(
            audit.loc[
                audit.get("region_aware_fdr_status", pd.Series(dtype=str)).astype(str).eq("negative_control_leakage"),
                "control_name",
            ].astype(str)
        )
    ) if not audit.empty and "control_name" in audit else []
    positives = sorted(
        set(
            audit.loc[
                audit.get("region_aware_fdr_status", pd.Series(dtype=str)).astype(str).eq(
                    "region_aware_cell_type_positive_significant"
                ),
                "control_name",
            ].astype(str)
        )
    ) if not audit.empty and "control_name" in audit else []
    return {
        "region_aware_note": (
            "Visium HD CRC region/cell-type-preserving targeted FDR finds cell-type-level positive signals for "
            f"{','.join(positives) if positives else 'none'}, but strict success={strict} because negative-control "
            f"leakage remains ({','.join(leaked) if leaked else 'none'}). Null statuses are within-region="
            f"{within.get('control_status', 'not_available')}, cell-type-preserving="
            f"{cell_type.get('control_status', 'not_available')}, LR-identity="
            f"{identity.get('control_status', 'not_available')}."
        )
    }


def _spamosaic_tonsil_model_summary() -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    control_summary_path = goal_dir / "spamosaic_tonsil_A1_method_control_summary.csv"
    if not control_summary_path.exists():
        return {}
    control_summary = pd.read_csv(control_summary_path)
    spatialself = _find_row(control_summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(control_summary, "method", "commot")
    cellchat = _find_row(control_summary, "method", "cellchat")
    spatalk = _find_row(control_summary, "method", "spatalk")
    cellnest = _find_row(control_summary, "method", "cellnest")
    downstream = _spamosaic_tonsil_downstream_summary(goal_dir)
    restricted_downstream = _spamosaic_tonsil_restricted_downstream_summary(goal_dir)
    rich_target_note = _spamosaic_tonsil_rich_target_note(goal_dir, "spamosaic_tonsil_A1")
    pathway_note = _pathway_specific_downstream_note(goal_dir, "spamosaic_tonsil_A1")
    targeted = _spamosaic_tonsil_targeted_fdr_summary(goal_dir)
    runtime = _read_optional_table(goal_dir / "spamosaic_tonsil_A1_method_runtime_summary.csv")
    methods_completed = ";".join(str(method) for method in control_summary["method"].dropna().astype(str))
    negative_risk = [
        f"{row.method}:{_fmt_int(row.best_negative_rank)}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")) == "negative_control_ranks_as_high_or_higher"
    ]
    noninformative = [
        f"{row.method}:{row.rank_validation_status}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")).startswith("noninformative")
    ]
    spatialself_pos = f"{spatialself.get('best_positive_control', '')}:rank{_fmt_int(spatialself.get('best_positive_rank', ''))}"
    spatialself_neg = f"{spatialself.get('best_negative_control', '')}:rank{_fmt_int(spatialself.get('best_negative_rank', ''))}"
    return {
        "validation_status": "five_method_panel_with_spatialself_ranking_advantage_but_mixed_functional_fdr",
        "positive_control_ranking": (
            f"SpatialSelf={spatialself_pos},IL21-IL21R:rank{_fmt_int(spatialself.get('weak_context_lr_rank', ''))};"
            f"COMMOT={commot.get('best_positive_control', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(commot.get('weak_context_lr_rank', ''))};"
            f"CellChat={cellchat.get('best_positive_control', '')}:rank{_fmt_int(cellchat.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(cellchat.get('weak_context_lr_rank', ''))};"
            f"CellNEST={cellnest.get('best_positive_control', '')}:rank{_fmt_int(cellnest.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(cellnest.get('weak_context_lr_rank', ''))}"
        ),
        "negative_control_ranking": (
            f"SpatialSelf={spatialself_neg};"
            f"COMMOT={commot.get('best_negative_control', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))};"
            f"CellChat={cellchat.get('best_negative_control', '')}:rank{_fmt_int(cellchat.get('best_negative_rank', ''))};"
            f"CellNEST={cellnest.get('best_negative_control', '')}:rank{_fmt_int(cellnest.get('best_negative_rank', ''))};"
            f"SpaTalk={spatalk.get('rank_validation_status', '')}"
        ),
        "downstream_target_response": downstream.get("downstream_target_response", "not_yet_run"),
        "fdr_or_feasibility_note": (
            "SPAMOSAIC tonsil RNA+ADT is useful as a cross-platform ranking stress test: SpatialSelf ranks all five "
            "positive controls above both high-expression negatives and prioritizes weak/context IL21-IL21R at rank "
            f"{_fmt_int(spatialself.get('weak_context_lr_rank', ''))}; COMMOT, CellChat, and CellNEST rank a high-expression "
            f"negative control first or near-first ({', '.join(negative_risk) if negative_risk else 'none'}). "
            f"{downstream.get('downstream_note', '')} {restricted_downstream.get('restricted_downstream_note', '')} "
            f"{rich_target_note} {targeted.get('targeted_region_note', '')} {pathway_note}"
        ),
        "baseline_status": (
            f"methods_completed={methods_completed};"
            f"spatialself_status={spatialself.get('rank_validation_status', '')};"
            f"negative_control_risk_methods={', '.join(negative_risk) if negative_risk else 'none'};"
            f"noninformative_methods={', '.join(noninformative) if noninformative else 'none'}"
        ),
        "runtime_memory": _runtime_compact(runtime),
    }


def _spamosaic_tonsil_section_summary(section_id: str) -> dict[str, str]:
    goal_dir = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
    control_summary_path = goal_dir / f"{section_id}_method_control_summary.csv"
    if not control_summary_path.exists():
        return {}
    control_summary = pd.read_csv(control_summary_path)
    spatialself = _find_row(control_summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(control_summary, "method", "commot")
    cellchat = _find_row(control_summary, "method", "cellchat")
    spatalk = _find_row(control_summary, "method", "spatalk")
    cellnest = _find_row(control_summary, "method", "cellnest")
    downstream = _spamosaic_tonsil_section_downstream_summary(goal_dir, section_id)
    restricted_downstream = _spamosaic_tonsil_section_restricted_downstream_summary(goal_dir, section_id)
    rich_target_note = _spamosaic_tonsil_rich_target_note(goal_dir, section_id)
    pathway_note = _pathway_specific_downstream_note(goal_dir, section_id)
    targeted = _spamosaic_tonsil_section_targeted_fdr_summary(goal_dir, section_id)
    runtime = _read_optional_table(goal_dir / f"{section_id}_method_runtime_summary.csv")
    methods_completed = ";".join(str(method) for method in control_summary["method"].dropna().astype(str))
    negative_risk = [
        f"{row.method}:{_fmt_int(row.best_negative_rank)}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")) == "negative_control_ranks_as_high_or_higher"
    ]
    noninformative = [
        f"{row.method}:{row.rank_validation_status}"
        for row in control_summary.itertuples(index=False)
        if str(getattr(row, "rank_validation_status", "")).startswith("noninformative")
    ]
    return {
        "validation_status": "independent_section_ranking_replication_but_mixed_functional_fdr",
        "positive_control_ranking": (
            f"SpatialSelf={spatialself.get('best_positive_control', '')}:rank{_fmt_int(spatialself.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(spatialself.get('weak_context_lr_rank', ''))};"
            f"COMMOT={commot.get('best_positive_control', '')}:rank{_fmt_int(commot.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(commot.get('weak_context_lr_rank', ''))};"
            f"CellChat={cellchat.get('best_positive_control', '')}:rank{_fmt_int(cellchat.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(cellchat.get('weak_context_lr_rank', ''))};"
            f"CellNEST={cellnest.get('best_positive_control', '')}:rank{_fmt_int(cellnest.get('best_positive_rank', ''))},"
            f"IL21-IL21R:rank{_fmt_int(cellnest.get('weak_context_lr_rank', ''))}"
        ),
        "negative_control_ranking": (
            f"SpatialSelf={spatialself.get('best_negative_control', '')}:rank{_fmt_int(spatialself.get('best_negative_rank', ''))};"
            f"COMMOT={commot.get('best_negative_control', '')}:rank{_fmt_int(commot.get('best_negative_rank', ''))};"
            f"CellChat={cellchat.get('best_negative_control', '')}:rank{_fmt_int(cellchat.get('best_negative_rank', ''))};"
            f"CellNEST={cellnest.get('best_negative_control', '')}:rank{_fmt_int(cellnest.get('best_negative_rank', ''))};"
            f"SpaTalk={spatalk.get('rank_validation_status', '')}"
        ),
        "downstream_target_response": downstream.get("downstream_target_response", "not_yet_run"),
        "fdr_or_feasibility_note": (
            f"{section_id} independently replicates the SPAMOSAIC ranking pattern: SpatialSelf prioritizes positives "
            f"and suppresses high-expression negatives (status={spatialself.get('rank_validation_status', '')}), while "
            f"COMMOT/CellChat/CellNEST show negative-control top-rank risk ({', '.join(negative_risk) if negative_risk else 'none'}). "
            f"{downstream.get('downstream_note', '')} {restricted_downstream.get('restricted_downstream_note', '')} "
            f"{rich_target_note} {targeted.get('targeted_region_note', '')} {pathway_note}"
        ),
        "baseline_status": (
            f"methods_completed={methods_completed};"
            f"negative_control_risk_methods={', '.join(negative_risk) if negative_risk else 'none'};"
            f"noninformative_methods={', '.join(noninformative) if noninformative else 'none'}"
        ),
        "runtime_memory": _runtime_compact(runtime),
    }


def _spamosaic_tonsil_section_downstream_summary(goal_dir: Path, section_id: str) -> dict[str, str]:
    summary_path = goal_dir / f"{section_id}_downstream_target_validation/{section_id}_downstream_method_summary.csv"
    claims_path = goal_dir / f"{section_id}_downstream_target_validation/{section_id}_downstream_claims.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    spatialself = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(summary, "method", "commot")
    spatialself_claim = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    return {
        "downstream_target_response": (
            f"SpatialSelf={_short_lr_id(spatialself.get('best_positive_lr', ''))}:rho"
            f"{_fmt_float(spatialself.get('best_positive_spearman', ''))},FDR"
            f"{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))},neg="
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))}:rho"
            f"{_fmt_float(spatialself.get('best_negative_spearman', ''))};COMMOT="
            f"{_short_lr_id(commot.get('best_positive_lr', ''))}:rho"
            f"{_fmt_float(commot.get('best_positive_spearman', ''))},neg="
            f"{_short_lr_id(commot.get('best_negative_lr', ''))}:rho"
            f"{_fmt_float(commot.get('best_negative_spearman', ''))}"
        ),
        "downstream_note": (
            f"Downstream proxy validation remains mixed for {section_id}: SpatialSelf best positive "
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))} rho="
            f"{_fmt_float(spatialself.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}, but best negative "
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))} also correlates with FDR="
            f"{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}; claim="
            f"{spatialself_claim.get('claim_status', 'not_available')}."
        ),
    }


def _spamosaic_tonsil_section_restricted_downstream_summary(goal_dir: Path, section_id: str) -> dict[str, str]:
    summary_path = (
        goal_dir
        / f"{section_id}_restricted_downstream_validation/{section_id}_restricted_downstream_method_summary.csv"
    )
    claims_path = (
        goal_dir / f"{section_id}_restricted_downstream_validation/{section_id}_restricted_downstream_claims.csv"
    )
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    spatialself = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    spatialself_claim = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    return {
        "restricted_downstream_note": (
            f"A stricter receiver-restricted residual downstream audit was also run for {section_id}. It remains mixed: "
            f"SpatialSelf best positive {_short_lr_id(spatialself.get('best_positive_lr', ''))} has rho="
            f"{_fmt_float(spatialself.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}, but best negative "
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))} also has rho="
            f"{_fmt_float(spatialself.get('best_negative_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}; restricted claim="
            f"{spatialself_claim.get('claim_status', 'not_available')}."
        )
    }


def _spamosaic_tonsil_section_targeted_fdr_summary(goal_dir: Path, section_id: str) -> dict[str, str]:
    target_dir = goal_dir / f"{section_id}_targeted_region_fdr_200perm"
    control_path = target_dir / f"{section_id}_targeted_control_fdr_summary.csv"
    audit_path = target_dir / f"{section_id}_targeted_region_fdr_audit.csv"
    if not control_path.exists() or not audit_path.exists():
        return {}
    control = pd.read_csv(control_path)
    audit = pd.read_csv(audit_path)
    lr_identity = _find_row(control, "null_model", "lr_identity_permutation")
    spatial = _find_row(control, "null_model", "spatial_permutation")
    receiver = _find_row(control, "null_model", "receiver_permutation")
    leakage = bool(
        (audit.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any()
    )
    return {
        "targeted_region_note": (
            f"Targeted FDR for {section_id} again supports only LR-identity recovery at LR-level "
            f"(status={lr_identity.get('control_status', '')}; best positive FDR="
            f"{_fmt_float(lr_identity.get('best_positive_group_bh_fdr', ''))}; best negative FDR="
            f"{_fmt_float(lr_identity.get('best_negative_group_bh_fdr', ''))}); spatial/receiver statuses="
            f"{spatial.get('control_status', '')}/{receiver.get('control_status', '')}; "
            f"cell-type negative leakage={leakage}. This is replication/supportive evidence, not strict Goal 5 success."
        )
    }


def _spamosaic_tonsil_rich_target_note(goal_dir: Path, section_id: str) -> str:
    downstream_path = (
        goal_dir
        / f"{section_id}_rich_targets_downstream_validation/{section_id}_downstream_method_summary.csv"
    )
    restricted_path = (
        goal_dir
        / f"{section_id}_rich_targets_restricted_downstream_validation/{section_id}_restricted_downstream_method_summary.csv"
    )
    pathway_claims_path = (
        goal_dir
        / f"{section_id}_rich_targets_pathway_specific_downstream/{section_id}_pathway_specific_downstream_claims.csv"
    )
    if not downstream_path.exists():
        return ""
    downstream = pd.read_csv(downstream_path)
    restricted = pd.read_csv(restricted_path) if restricted_path.exists() else pd.DataFrame()
    pathway_claims = pd.read_csv(pathway_claims_path) if pathway_claims_path.exists() else pd.DataFrame()
    ss_downstream = _find_row(downstream, "method", "spatialself_wavelet_edge_prior")
    ss_restricted = _find_row(restricted, "method", "spatialself_wavelet_edge_prior")
    ss_pathway = _find_row(pathway_claims, "method", "spatialself_wavelet_edge_prior")
    return (
        f"Rich-target SPAMOSAIC audit for {section_id} used the original full-transcriptome h5ad to expand "
        "the retained RNA target panel while preserving the exact same cells/candidate_ids as existing predictions. "
        "It does not close Goal 5: broad downstream SpatialSelf best positive "
        f"{_short_lr_id(ss_downstream.get('best_positive_lr', ''))} rho="
        f"{_fmt_float(ss_downstream.get('best_positive_spearman', ''))}, FDR="
        f"{_fmt_float(ss_downstream.get('best_positive_bh_fdr', ''))}, but best negative "
        f"{_short_lr_id(ss_downstream.get('best_negative_lr', ''))} also has rho="
        f"{_fmt_float(ss_downstream.get('best_negative_spearman', ''))}, FDR="
        f"{_fmt_float(ss_downstream.get('best_negative_bh_fdr', ''))}. Restricted residual audit best positive "
        f"{_short_lr_id(ss_restricted.get('best_positive_lr', ''))} FDR="
        f"{_fmt_float(ss_restricted.get('best_positive_bh_fdr', ''))}, best negative "
        f"{_short_lr_id(ss_restricted.get('best_negative_lr', ''))} FDR="
        f"{_fmt_float(ss_restricted.get('best_negative_bh_fdr', ''))}; pathway-specific claim="
        f"{ss_pathway.get('claim_status', 'not_available')}. This is boundary evidence that the tonsil failure is "
        "not simply caused by too few target genes."
    )


def _spamosaic_tonsil_downstream_summary(goal_dir: Path) -> dict[str, str]:
    summary_path = (
        goal_dir
        / "spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_method_summary.csv"
    )
    claims_path = goal_dir / "spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_claims.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    spatialself = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(summary, "method", "commot")
    cellchat = _find_row(summary, "method", "cellchat")
    cellnest = _find_row(summary, "method", "cellnest")
    spatialself_claim = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    return {
        "downstream_target_response": (
            "SpatialSelf="
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))}:rho{_fmt_float(spatialself.get('best_positive_spearman', ''))},"
            f"FDR{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))},"
            f"neg={_short_lr_id(spatialself.get('best_negative_lr', ''))}:rho{_fmt_float(spatialself.get('best_negative_spearman', ''))},"
            f"FDR{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))};"
            "COMMOT="
            f"{_short_lr_id(commot.get('best_positive_lr', ''))}:rho{_fmt_float(commot.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(commot.get('best_negative_lr', ''))}:rho{_fmt_float(commot.get('best_negative_spearman', ''))};"
            "CellChat="
            f"{_short_lr_id(cellchat.get('best_positive_lr', ''))}:rho{_fmt_float(cellchat.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(cellchat.get('best_negative_lr', ''))}:rho{_fmt_float(cellchat.get('best_negative_spearman', ''))};"
            "CellNEST="
            f"{_short_lr_id(cellnest.get('best_positive_lr', ''))}:rho{_fmt_float(cellnest.get('best_positive_spearman', ''))},"
            f"neg={_short_lr_id(cellnest.get('best_negative_lr', ''))}:rho{_fmt_float(cellnest.get('best_negative_spearman', ''))}"
        ),
        "downstream_note": (
            "Downstream proxy validation remains mixed: SpatialSelf's best positive is "
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))} "
            f"(rho={_fmt_float(spatialself.get('best_positive_spearman', ''))}, "
            f"FDR={_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}), but "
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))} also correlates "
            f"(rho={_fmt_float(spatialself.get('best_negative_spearman', ''))}, "
            f"FDR={_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}; "
            f"claim={spatialself_claim.get('claim_status', 'not_available')}). Therefore SPAMOSAIC is not counted as strict functional success."
        ),
    }


def _spamosaic_tonsil_restricted_downstream_summary(goal_dir: Path) -> dict[str, str]:
    summary_path = (
        goal_dir
        / "spamosaic_tonsil_A1_restricted_downstream_validation/"
        "spamosaic_tonsil_restricted_downstream_method_summary.csv"
    )
    claims_path = (
        goal_dir
        / "spamosaic_tonsil_A1_restricted_downstream_validation/"
        "spamosaic_tonsil_restricted_downstream_claims.csv"
    )
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    spatialself = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    spatialself_claim = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    return {
        "restricted_downstream_note": (
            "A stricter receiver-restricted residual downstream audit was also run, controlling for library size, "
            "receptor abundance, and receiver cell-type proxies. It remains mixed: SpatialSelf best positive "
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))} has rho="
            f"{_fmt_float(spatialself.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}, but best negative "
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))} also has rho="
            f"{_fmt_float(spatialself.get('best_negative_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}; "
            f"restricted claim={spatialself_claim.get('claim_status', 'not_available')}."
        )
    }


def _spamosaic_tonsil_targeted_fdr_summary(goal_dir: Path) -> dict[str, str]:
    target_dir = goal_dir / "spamosaic_tonsil_A1_targeted_region_fdr_200perm"
    control_path = target_dir / "spamosaic_tonsil_A1_targeted_control_fdr_summary.csv"
    audit_path = target_dir / "spamosaic_tonsil_A1_targeted_region_fdr_audit.csv"
    if not control_path.exists() or not audit_path.exists():
        target_dir = goal_dir / "spamosaic_tonsil_A1_targeted_region_fdr_100perm"
        control_path = target_dir / "spamosaic_tonsil_A1_targeted_control_fdr_summary.csv"
        audit_path = target_dir / "spamosaic_tonsil_A1_targeted_region_fdr_audit.csv"
    if not control_path.exists() or not audit_path.exists():
        return {}
    control = pd.read_csv(control_path)
    audit = pd.read_csv(audit_path)
    lr_identity = _find_row(control, "null_model", "lr_identity_permutation")
    spatial = _find_row(control, "null_model", "spatial_permutation")
    receiver = _find_row(control, "null_model", "receiver_permutation")
    leakage = bool(
        (audit.get("targeted_region_fdr_status", pd.Series(dtype=str)).astype(str) == "negative_control_leakage").any()
    )
    cxcl13 = _find_row(audit, "control_name", "CXCL13-CXCR5")
    return {
        "targeted_region_note": (
            f"Targeted FDR ({target_dir.name}) shows LR-identity positive recovery with negative suppression at LR-level "
            f"(status={lr_identity.get('control_status', '')}; best positive FDR="
            f"{_fmt_float(lr_identity.get('best_positive_group_bh_fdr', ''))}; best negative FDR="
            f"{_fmt_float(lr_identity.get('best_negative_group_bh_fdr', ''))}), but spatial/receiver LR-level nulls remain "
            f"{spatial.get('control_status', '')}/{receiver.get('control_status', '')}. "
            f"Cell-type targeted FDR has negative-control leakage={leakage}; CXCL13-CXCR5 cell-type all-three="
            f"{cxcl13.get('targeted_cell_type_any_all_three_fdr05', '')}. This is supportive, not strict Goal 5 FDR success."
        )
    }


def _pathway_specific_downstream_note(goal_dir: Path, dataset_id: str) -> str:
    output_dir = goal_dir / f"{dataset_id}_pathway_specific_downstream"
    summary_path = output_dir / f"{dataset_id}_pathway_specific_downstream_summary.csv"
    claims_path = output_dir / f"{dataset_id}_pathway_specific_downstream_claims.csv"
    if not summary_path.exists() or not claims_path.exists():
        return ""
    summary = _read_optional_table(summary_path)
    claims = _read_optional_table(claims_path)
    if summary.empty or claims.empty:
        return ""
    spatialself = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    if not spatialself:
        spatialself = _find_row(claims, "method", "spatialself_calibrated")
    if not spatialself:
        return ""
    target_pair = spatialself.get("best_target_pair", "")
    target_rows = summary[
        summary.get("method", pd.Series(dtype=str)).astype(str).eq(str(spatialself.get("method", "")))
        & summary.get("target_pair", pd.Series(dtype=str)).astype(str).eq(str(target_pair))
    ]
    target_row = target_rows.iloc[0].to_dict() if not target_rows.empty else {}
    residual_note = _fully_residual_pathway_note(goal_dir, dataset_id)
    clean_target_note = _clean_target_pathway_note(goal_dir, dataset_id)
    return (
        "A pathway-specific downstream competition diagnostic was also run. "
        f"Best SpatialSelf-family target={target_pair}, claim={spatialself.get('claim_status', '')}, "
        f"target rho/FDR={_fmt_float(target_row.get('target_lr_spearman', ''))}/"
        f"{_fmt_float(target_row.get('target_lr_bh_fdr', ''))}, "
        f"best negative={target_row.get('best_negative_pair', '')}, negative FDR="
        f"{_fmt_float(target_row.get('best_negative_bh_fdr', ''))}. "
        "This diagnostic is retained as functional-specificity evidence and does not replace strict all-null FDR."
        f" {residual_note} {clean_target_note}"
    )


def _fully_residual_pathway_note(goal_dir: Path, dataset_id: str) -> str:
    output_dirs = [
        goal_dir / f"{dataset_id}_fully_residual_pathway_downstream",
        goal_dir / f"{dataset_id}_rich_targets_fully_residual_pathway_downstream",
    ]
    for output_dir in output_dirs:
        summary_paths = sorted(output_dir.glob("*_pathway_specific_downstream_summary.csv"))
        claims_paths = sorted(output_dir.glob("*_pathway_specific_downstream_claims.csv"))
        if not summary_paths or not claims_paths:
            continue
        summary = _read_optional_table(summary_paths[0])
        claims = _read_optional_table(claims_paths[0])
        if summary.empty or claims.empty:
            continue
        spatialself = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
        if not spatialself:
            spatialself = _find_row(claims, "method", "spatialself_calibrated")
        if not spatialself:
            continue
        target_pair = spatialself.get("best_target_pair", "")
        target_rows = summary[
            summary.get("method", pd.Series(dtype=str)).astype(str).eq(str(spatialself.get("method", "")))
            & summary.get("target_pair", pd.Series(dtype=str)).astype(str).eq(str(target_pair))
        ]
        target_row = target_rows.iloc[0].to_dict() if not target_rows.empty else {}
        return (
            "A fully residual pathway diagnostic was added with both incoming communication and target signatures "
            "residualized against receiver context. It still did not close Goal 5: "
            f"target={target_pair}, claim={spatialself.get('claim_status', '')}, target rho/FDR="
            f"{_fmt_float(target_row.get('target_lr_spearman', ''))}/"
            f"{_fmt_float(target_row.get('target_lr_bh_fdr', ''))}, best negative="
            f"{target_row.get('best_negative_pair', '')}, negative FDR="
            f"{_fmt_float(target_row.get('best_negative_bh_fdr', ''))}."
        )
    return ""


def _clean_target_pathway_note(goal_dir: Path, dataset_id: str) -> str:
    output_dir = goal_dir / f"{dataset_id}_clean_targets_fully_residual_pathway_downstream"
    summary_paths = sorted(output_dir.glob("*_pathway_specific_downstream_summary.csv"))
    claims_paths = sorted(output_dir.glob("*_pathway_specific_downstream_claims.csv"))
    if not summary_paths or not claims_paths:
        return ""
    summary = _read_optional_table(summary_paths[0])
    claims = _read_optional_table(claims_paths[0])
    if summary.empty or claims.empty:
        return ""
    spatialself = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    if not spatialself:
        spatialself = _find_row(claims, "method", "spatialself_calibrated")
    if not spatialself:
        return ""
    target_pair = spatialself.get("best_target_pair", "")
    target_rows = summary[
        summary.get("method", pd.Series(dtype=str)).astype(str).eq(str(spatialself.get("method", "")))
        & summary.get("target_pair", pd.Series(dtype=str)).astype(str).eq(str(target_pair))
    ]
    target_row = target_rows.iloc[0].to_dict() if not target_rows.empty else {}
    n_tested = int(pd.to_numeric(summary.get("n_tested", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    return (
        "A stricter clean-target diagnostic removed ligand/receptor genes, high-expression negative-control genes, "
        "common lineage markers, and housekeeping genes before fully residual downstream testing. It did not close "
        f"Goal 5: claim={spatialself.get('claim_status', '')}, total tested target rows={n_tested}, "
        f"best target={target_pair}, target rho/FDR={_fmt_float(target_row.get('target_lr_spearman', ''))}/"
        f"{_fmt_float(target_row.get('target_lr_bh_fdr', ''))}, best negative="
        f"{target_row.get('best_negative_pair', '')}, negative FDR="
        f"{_fmt_float(target_row.get('best_negative_bh_fdr', ''))}."
    )


def _slide_tags_tonsil_downstream_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "outputs/phase2/goal5_interpretability/tonsil_fresh_subset/tonsil_downstream_target_correlation.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    positives = table[table.get("control_class", pd.Series(dtype=str)).astype(str).eq("tonsil_positive_prior")]
    negatives = table[table.get("control_class", pd.Series(dtype=str)).astype(str).eq("housekeeping_negative_control")]
    positive_supported = positives[
        (pd.to_numeric(positives.get("spearman_r", pd.Series(dtype=float)), errors="coerce") > 0)
        & (pd.to_numeric(positives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1)
    ].copy()
    if positive_supported.empty:
        return {}
    best = positive_supported.sort_values(["bh_fdr", "spearman_r"], ascending=[True, False]).iloc[0]
    n_negative = int((pd.to_numeric(negatives.get("bh_fdr", pd.Series(dtype=float)), errors="coerce") <= 0.1).sum())
    return {
        "downstream_target_response": (
            f"SpatialSelf={best.get('canonical_pair')}:rho{float(best.get('spearman_r')):.4f},"
            f"FDR{float(best.get('bh_fdr')):.4f};negative_downstream_fdr_0_1={n_negative}"
        ),
        "downstream_note": (
            "High-resolution tonsil downstream validation supports CCL21-CCR7 while housekeeping negative controls "
            "do not show downstream FDR leakage."
        ),
    }


def _slide_tags_tonsil_targeted_fdr_summary() -> dict[str, str]:
    target_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        "slide_tags_tonsil_targeted_region_fdr_100perm"
    )
    control_path = target_dir / "slide_tags_tonsil_targeted_control_fdr_summary.csv"
    manifest_path = target_dir / "slide_tags_tonsil_targeted_region_fdr_manifest.json"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    strict = False
    if manifest_path.exists():
        try:
            strict = bool(json.loads(manifest_path.read_text(encoding="utf-8")).get("strict_success", False))
        except json.JSONDecodeError:
            strict = False
    lr_identity = _find_row(control, "null_model", "lr_identity_permutation")
    spatial = _find_row(control, "null_model", "spatial_permutation")
    receiver = _find_row(control, "null_model", "receiver_permutation")
    return {
        "fdr_or_feasibility_note": (
            "Targeted high-resolution tonsil FDR has strict three-null success="
            f"{strict}. LR-identity null status={lr_identity.get('control_status', '')}; "
            f"spatial/receiver statuses={spatial.get('control_status', '')}/{receiver.get('control_status', '')}. "
            "Negatives are suppressed, but spatial/receiver nulls have no positive recovery."
        ),
        "targeted_region_note": (
            f"LR-identity best positive FDR={lr_identity.get('best_positive_group_bh_fdr', '')}; "
            f"best negative FDR={lr_identity.get('best_negative_group_bh_fdr', '')}."
        ),
    }


def _slide_tags_tonsil_context_restricted_model_rerun_summary() -> dict[str, str]:
    output_dir = (
        PROJECT_ROOT
        / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/"
        "slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_raw"
    )
    control_path = output_dir / "slide_tags_tonsil_n1200_context_restricted_control_fdr_summary.csv"
    audit_path = output_dir / "slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_audit.csv"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    status = (
        audit.iloc[0].get("context_restricted_model_rerun_status", "")
        if not audit.empty
        else ""
    )
    notes = [
        f"{row.null_model}:pos={_fmt_float(getattr(row, 'best_positive_group_bh_fdr', ''))},"
        f"neg={_fmt_float(getattr(row, 'best_negative_group_bh_fdr', ''))},"
        f"status={getattr(row, 'control_status', '')}"
        for row in control.itertuples(index=False)
    ]
    return {
        "context_restricted_model_rerun_note": (
            "A Slide-tags n1200 context-restricted model-rerun diagnostic reused existing 20-per-null retrained "
            f"predictions and remained non-strict: status={status}; nulls [{'; '.join(notes)}]. "
            "This records why Slide-tags remains supportive but not strict all-null closure."
        )
    }


def _xenium_downstream_summary(goal_dir: Path) -> dict[str, str]:
    downstream_dir = goal_dir / "xenium_downstream_target_validation"
    summary_path = downstream_dir / "xenium_downstream_method_summary.csv"
    claims_path = downstream_dir / "xenium_downstream_claims.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    spatialself = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    commot = _find_row(summary, "method", "commot")
    spatialself_claim = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    commot_claim = _find_row(claims, "method", "commot")
    claim_statuses = sorted(
        {
            str(value)
            for value in claims.get("claim_status", pd.Series(dtype=str)).dropna().astype(str)
        }
    )
    return {
        "downstream_target_response": (
            "SpatialSelf="
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))}:rho{_fmt_float(spatialself.get('best_positive_spearman', ''))},"
            f"FDR{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))};"
            "SpatialSelf_negative="
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))}:rho{_fmt_float(spatialself.get('best_negative_spearman', ''))},"
            f"FDR{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))};"
            "COMMOT="
            f"{_short_lr_id(commot.get('best_positive_lr', ''))}:rho{_fmt_float(commot.get('best_positive_spearman', ''))},"
            f"FDR{_fmt_float(commot.get('best_positive_bh_fdr', ''))};"
            "COMMOT_negative="
            f"{_short_lr_id(commot.get('best_negative_lr', ''))}:rho{_fmt_float(commot.get('best_negative_spearman', ''))},"
            f"FDR{_fmt_float(commot.get('best_negative_bh_fdr', ''))}"
        ),
        "fdr_or_feasibility_note": (
            "Xenium within-panel downstream target validation has been run for SpatialSelf and COMMOT using empirical "
            "permutation p-values plus BH-FDR. CCL19-CCR7 shows significant target-response correlation, stronger for "
            f"SpatialSelf (rho={_fmt_float(spatialself.get('best_positive_spearman', ''))}, "
            f"FDR={_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}) than COMMOT "
            f"(rho={_fmt_float(commot.get('best_positive_spearman', ''))}, "
            f"FDR={_fmt_float(commot.get('best_positive_bh_fdr', ''))}). However, high-expression negative controls also "
            "show significant target correlations for both methods "
            f"(SpatialSelf { _short_lr_id(spatialself.get('best_negative_lr', '')) } "
            f"rho={_fmt_float(spatialself.get('best_negative_spearman', ''))}, "
            f"FDR={_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}; COMMOT "
            f"{ _short_lr_id(commot.get('best_negative_lr', '')) } "
            f"rho={_fmt_float(commot.get('best_negative_spearman', ''))}, "
            f"FDR={_fmt_float(commot.get('best_negative_bh_fdr', ''))}). "
            "Therefore the Xenium result is mixed platform/functional evidence, not yet successful false-positive-control evidence. "
            f"Claim statuses: {', '.join(claim_statuses) if claim_statuses else 'not_available'}; "
            f"SpatialSelf allowed claim: {spatialself_claim.get('allowed_claim', 'not_available')}; "
            f"COMMOT allowed claim: {commot_claim.get('allowed_claim', 'not_available')}."
        ),
    }


def _xenium_prediction_fdr_summary(goal_dir: Path) -> dict[str, str]:
    fdr_dir = goal_dir / "xenium_prediction_fdr_50perm"
    control_path = fdr_dir / "xenium_prediction_control_fdr_summary.csv"
    status_path = fdr_dir / "xenium_prediction_fdr_status.csv"
    if not control_path.exists():
        return {}
    control = pd.read_csv(control_path)
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    if control.empty:
        return {"prediction_fdr_note": "FDR table exists but is empty."}
    n_pos_sig = int(pd.to_numeric(control["n_positive_significant"], errors="coerce").fillna(0).sum())
    n_neg_sig = int(pd.to_numeric(control["n_negative_significant"], errors="coerce").fillna(0).sum())
    methods_completed = int((status.get("status", pd.Series(dtype=str)).astype(str) == "completed").sum())
    min_positive = float(pd.to_numeric(control["best_positive_group_bh_fdr"], errors="coerce").min())
    min_negative = float(pd.to_numeric(control["best_negative_group_bh_fdr"], errors="coerce").min())
    status_counts = control["control_fdr_status"].value_counts().to_dict()
    return {
        "prediction_fdr_note": (
            "50-permutation prediction-level FDR completed for "
            f"{methods_completed} methods across spatial, receiver, and LR-identity nulls. "
            "No positive controls and no high-expression negative controls are significant at alpha=0.05 "
            f"(positive significant={n_pos_sig}; negative significant={n_neg_sig}; "
            f"min positive group BH-FDR={_fmt_float(min_positive)}; min negative group BH-FDR={_fmt_float(min_negative)}). "
            "This supports conservative significance calling on Xenium, but not positive-control recovery. "
            f"Control status counts: {status_counts}."
        )
    }


def _xenium_analysis_zarr_group_summary(goal_dir: Path) -> dict[str, str]:
    summary_path = goal_dir / "xenium_analysis_zarr_cell_group_summary.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    graph = _find_row(summary, "grouping_name", "gene_expression_graphclust")
    return {
        "zarr_group_note": (
            "10x analysis.zarr cell groups were audited as a possible stronger stratification source: graphclust maps "
            f"{_fmt_int(graph.get('n_subset_assigned', ''))}/{_fmt_int(graph.get('n_subset_cells', ''))} subset cells "
            f"across {_fmt_int(graph.get('n_subset_clusters', ''))} clusters. These are unsupervised 10x clusters, not "
            "curated cell types, and are recorded for future within-cluster nulls/annotation refinement."
        )
    }


def _xenium_graphclust_validation_summary(goal_dir: Path) -> dict[str, str]:
    validation_dir = goal_dir / "xenium_graphclust_validation_200perm"
    audit_path = validation_dir / "xenium_graphclust_validation_audit.csv"
    control_path = validation_dir / "xenium_graphclust_control_summary.csv"
    manifest_path = validation_dir / "xenium_graphclust_validation_manifest.json"
    if not audit_path.exists() or not control_path.exists():
        return {}
    audit = pd.read_csv(audit_path)
    control = pd.read_csv(control_path)
    spatialself = _find_row(audit, "method", "spatialself_wavelet_edge_prior")
    strict = bool(spatialself.get("graphclust_validation_status") == "strict_graphclust_success")
    leaking = [
        f"{row.method}:{row.graphclust_validation_status}"
        for row in audit.itertuples(index=False)
        if str(getattr(row, "graphclust_validation_status", "")) == "graphclust_negative_control_leakage"
    ]
    ss_pair = _find_xenium_graphclust_control(
        control,
        method="spatialself_wavelet_edge_prior",
        null_model="graphclust_pair_permutation",
    )
    ss_edge = _find_xenium_graphclust_control(
        control,
        method="spatialself_wavelet_edge_prior",
        null_model="edge_lr_identity_permutation",
    )
    claim_boundary = ""
    if manifest_path.exists():
        try:
            claim_boundary = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("claim_boundary", ""))
        except json.JSONDecodeError:
            claim_boundary = ""
    return {
        "graphclust_validation_note": (
            "Supplemental Xenium graphclust-aware validation was run using 10x gene-expression graph clusters. "
            f"SpatialSelf strict_graphclust_success={strict}: best positive "
            f"{spatialself.get('best_graphclust_positive', '')} rank="
            f"{_fmt_int(spatialself.get('best_graphclust_positive_rank', ''))}, best negative "
            f"{spatialself.get('best_graphclust_negative', '')} rank="
            f"{_fmt_int(spatialself.get('best_graphclust_negative_rank', ''))}; graphclust-pair FDR "
            f"positive/negative={_fmt_float(ss_pair.get('best_positive_group_bh_fdr', ''))}/"
            f"{_fmt_float(ss_pair.get('best_negative_group_bh_fdr', ''))}; edge-LR-identity FDR "
            f"positive/negative={_fmt_float(ss_edge.get('best_positive_group_bh_fdr', ''))}/"
            f"{_fmt_float(ss_edge.get('best_negative_group_bh_fdr', ''))}. "
            f"Baseline graphclust leakage methods: {', '.join(leaking) if leaking else 'none'}. "
            f"{claim_boundary}"
        )
    }


def _xenium_graphclust_residual_downstream_summary(goal_dir: Path) -> dict[str, str]:
    downstream_dir = goal_dir / "xenium_graphclust_residual_downstream_200perm"
    summary_path = downstream_dir / "xenium_graphclust_residual_downstream_method_summary.csv"
    claims_path = downstream_dir / "xenium_graphclust_residual_downstream_claims.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    claims = pd.read_csv(claims_path) if claims_path.exists() else pd.DataFrame()
    spatialself = _find_row(summary, "method", "spatialself_wavelet_edge_prior")
    claim = _find_row(claims, "method", "spatialself_wavelet_edge_prior")
    return {
        "graphclust_residual_downstream_note": (
            "Receiver graphclust-residual downstream validation was also run. It remains mixed for SpatialSelf "
            f"(claim={claim.get('claim_status', '')}): best positive "
            f"{_short_lr_id(spatialself.get('best_positive_lr', ''))} rho="
            f"{_fmt_float(spatialself.get('best_positive_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_positive_bh_fdr', ''))}; best negative "
            f"{_short_lr_id(spatialself.get('best_negative_lr', ''))} rho="
            f"{_fmt_float(spatialself.get('best_negative_spearman', ''))}, FDR="
            f"{_fmt_float(spatialself.get('best_negative_bh_fdr', ''))}. Therefore Xenium is graphclust/FDR support, "
            "not clean functional downstream closure."
        )
    }


def _xenium_rich_target_downstream_note(goal_dir: Path) -> str:
    downstream_path = (
        goal_dir
        / "xenium_rich_targets_downstream_validation/xenium_downstream_method_summary.csv"
    )
    graphclust_path = (
        goal_dir
        / "xenium_rich_targets_graphclust_residual_downstream_200perm/"
        "xenium_graphclust_residual_downstream_method_summary.csv"
    )
    if not downstream_path.exists():
        return ""
    downstream = pd.read_csv(downstream_path)
    graphclust = pd.read_csv(graphclust_path) if graphclust_path.exists() else pd.DataFrame()
    ss_downstream = _find_row(downstream, "method", "spatialself_wavelet_edge_prior")
    ss_graphclust = _find_row(graphclust, "method", "spatialself_wavelet_edge_prior")
    return (
        "Rich-target Xenium audit expanded within-panel target signatures while preserving the exact same "
        "cells/candidate_ids as existing predictions. It remains mixed: broad downstream SpatialSelf best positive "
        f"{_short_lr_id(ss_downstream.get('best_positive_lr', ''))} rho="
        f"{_fmt_float(ss_downstream.get('best_positive_spearman', ''))}, FDR="
        f"{_fmt_float(ss_downstream.get('best_positive_bh_fdr', ''))}, but best negative "
        f"{_short_lr_id(ss_downstream.get('best_negative_lr', ''))} also has rho="
        f"{_fmt_float(ss_downstream.get('best_negative_spearman', ''))}, FDR="
        f"{_fmt_float(ss_downstream.get('best_negative_bh_fdr', ''))}. Graphclust-residual rich-target audit best "
        f"positive {_short_lr_id(ss_graphclust.get('best_positive_lr', ''))} FDR="
        f"{_fmt_float(ss_graphclust.get('best_positive_bh_fdr', ''))}, best negative "
        f"{_short_lr_id(ss_graphclust.get('best_negative_lr', ''))} FDR="
        f"{_fmt_float(ss_graphclust.get('best_negative_bh_fdr', ''))}. This is boundary evidence that Xenium "
        "downstream leakage is not simply caused by too few target genes."
    )


def _find_xenium_graphclust_control(control: pd.DataFrame, *, method: str, null_model: str) -> dict[str, Any]:
    if control.empty:
        return {}
    subset = control[
        control.get("method", pd.Series(dtype=str)).astype(str).eq(method)
        & control.get("summary_type", pd.Series(dtype=str)).astype(str).eq("group_fdr")
        & control.get("null_model", pd.Series(dtype=str)).astype(str).eq(null_model)
    ]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _short_lr_id(value: Any) -> str:
    text = str(value)
    if not text:
        return ""
    parts = text.split("_")
    if len(parts) >= 2:
        suffix = "_".join(parts[-2:])
        if suffix.endswith("_negative"):
            suffix = "_".join(parts[-3:])
        return suffix.replace("_", "-")
    return text


def _runtime_from_time_log(path: Path) -> str:
    if not path.exists():
        return "not_yet_run"
    wall = ""
    rss = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Elapsed (wall clock) time" in line:
            wall = line.split("):", 1)[-1].strip() if "):" in line else line.rsplit(":", 1)[-1].strip()
        if "Maximum resident set size" in line:
            rss = line.split(":", 1)[-1].strip()
    return f"wall_clock={wall};max_rss_kb={rss}"


def _read_optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _runtime_compact(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "not_yet_run"
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(f"{row.method}={row.runtime}")
    return ";".join(rows)


def _status_compact(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "not_available"
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(f"{row.method}:{row.status}")
    return ",".join(rows)


def _fmt_int(value: Any) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _find_row(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if column not in frame:
        return {}
    subset = frame[frame[column].astype(str) == value]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 999


def _control_panel() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/tonsil_fresh_control_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "dataset", "slide_tags_tonsil_subset")
    frame.insert(1, "platform", "Slide-tags / high-resolution tonsil")
    return frame


def _runtime_summary(validation_summary: pd.DataFrame) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_run_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "dataset", "slide_tags_tonsil_subset")
    frame.insert(1, "platform", "Slide-tags / high-resolution tonsil")
    return frame


def _source_index() -> pd.DataFrame:
    rows = [
        (
            "dataset_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_cross_platform_dataset_manifest.csv",
        ),
        (
            "phase7_requirement_audit",
            "outputs/phase7/functional_cross_platform_validation/phase7_requirement_audit.csv",
        ),
        (
            "phase7_requirement_audit_summary",
            "outputs/phase7/functional_cross_platform_validation/phase7_requirement_audit_summary.json",
        ),
        (
            "acquisition_plan",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_cross_platform_acquisition_plan.csv",
        ),
        (
            "control_panel",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_cross_platform_control_panel.csv",
        ),
        (
            "validation_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_cross_platform_validation_summary.csv",
        ),
        (
            "runtime_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_cross_platform_runtime_summary.csv",
        ),
        (
            "standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_cross_platform_standardization_readiness.csv",
        ),
        (
            "xenium_human_lymph_node_subset_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_subset__standardization_readiness.csv",
        ),
        (
            "xenium_human_lymph_node_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_subset/standardization_summary.csv",
        ),
        (
            "xenium_human_lymph_node_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_subset/metadata.json",
        ),
        (
            "xenium_human_lymph_node_rich_targets_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/standardization_summary.csv",
        ),
        (
            "xenium_human_lymph_node_rich_targets_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/metadata.json",
        ),
        (
            "xenium_human_lymph_node_rich_targets_target_signatures",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/target_signatures.csv",
        ),
        (
            "xenium_human_lymph_node_method_lr_rank_table",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_method_lr_rank_table.csv",
        ),
        (
            "xenium_human_lymph_node_method_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_method_control_summary.csv",
        ),
        (
            "xenium_human_lymph_node_method_runtime_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_method_runtime_summary.csv",
        ),
        (
            "xenium_human_lymph_node_method_status_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_method_status_summary.csv",
        ),
        (
            "xenium_human_lymph_node_method_summary_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_method_summary_manifest.json",
        ),
        (
            "xenium_human_lymph_node_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_baseline_panel/spatialself_wavelet_edge_prior/xenium_human_lymph_node_subset__spatialself_wavelet_predictions.csv",
        ),
        (
            "xenium_human_lymph_node_commot_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_commot/xenium_human_lymph_node_replicate_00__commot_predictions.csv",
        ),
        (
            "xenium_human_lymph_node_cellchat_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_cellchat/cellchat/xenium_human_lymph_node_subset__cellchat_predictions.csv",
        ),
        (
            "xenium_human_lymph_node_spatalk_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_spatalk/spatalk/xenium_human_lymph_node_subset__spatalk_predictions.csv",
        ),
        (
            "xenium_human_lymph_node_cellnest_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node_cellnest/cellnest/xenium_human_lymph_node_subset__cellnest_predictions.csv",
        ),
        (
            "xenium_downstream_target_correlation",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_downstream_target_validation/xenium_downstream_target_correlation.csv",
        ),
        (
            "xenium_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_downstream_target_validation/xenium_downstream_method_summary.csv",
        ),
        (
            "xenium_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_downstream_target_validation/xenium_downstream_claims.csv",
        ),
        (
            "xenium_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_downstream_target_validation/xenium_downstream_manifest.json",
        ),
        (
            "xenium_prediction_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_prediction_fdr_50perm/xenium_prediction_fdr_summary.csv",
        ),
        (
            "xenium_prediction_lr_pair_fdr_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_prediction_fdr_50perm/xenium_prediction_lr_pair_fdr.csv",
        ),
        (
            "xenium_prediction_control_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_prediction_fdr_50perm/xenium_prediction_control_fdr_summary.csv",
        ),
        (
            "xenium_prediction_fdr_status_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_prediction_fdr_50perm/xenium_prediction_fdr_status.csv",
        ),
        (
            "xenium_prediction_fdr_manifest_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_prediction_fdr_50perm/xenium_prediction_fdr_manifest.json",
        ),
        (
            "xenium_analysis_zarr_cell_group_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_analysis_zarr_cell_group_summary.csv",
        ),
        (
            "xenium_analysis_zarr_subset_cell_group_assignments",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_analysis_zarr_subset_cell_group_assignments.csv",
        ),
        (
            "xenium_analysis_zarr_cell_group_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_analysis_zarr_cell_group_manifest.json",
        ),
        (
            "xenium_graphclust_validation_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_validation_200perm/xenium_graphclust_validation_audit.csv",
        ),
        (
            "xenium_graphclust_validation_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_validation_200perm/xenium_graphclust_control_summary.csv",
        ),
        (
            "xenium_graphclust_validation_lr_pair_fdr_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_validation_200perm/xenium_graphclust_lr_pair_fdr_all.csv",
        ),
        (
            "xenium_graphclust_validation_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_validation_200perm/xenium_graphclust_validation_manifest.json",
        ),
        (
            "xenium_graphclust_residual_downstream_correlation_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_correlation.csv",
        ),
        (
            "xenium_graphclust_residual_downstream_method_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_method_summary.csv",
        ),
        (
            "xenium_graphclust_residual_downstream_claims_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_claims.csv",
        ),
        (
            "xenium_graphclust_residual_downstream_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_manifest.json",
        ),
        (
            "xenium_rich_targets_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_rich_targets_downstream_validation/xenium_downstream_method_summary.csv",
        ),
        (
            "xenium_rich_targets_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_rich_targets_downstream_validation/xenium_downstream_claims.csv",
        ),
        (
            "xenium_rich_targets_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_rich_targets_downstream_validation/xenium_downstream_manifest.json",
        ),
        (
            "xenium_rich_targets_graphclust_residual_downstream_method_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_rich_targets_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_method_summary.csv",
        ),
        (
            "xenium_rich_targets_graphclust_residual_downstream_claims_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_rich_targets_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_claims.csv",
        ),
        (
            "xenium_rich_targets_graphclust_residual_downstream_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_rich_targets_graphclust_residual_downstream_200perm/xenium_graphclust_residual_downstream_manifest.json",
        ),
        (
            "lung_cancer_visium_subset_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/lung_cancer_visium_subset__standardization_readiness.csv",
        ),
        (
            "lung_cancer_visium_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/lung_cancer_visium_subset/standardization_summary.csv",
        ),
        (
            "lung_cancer_visium_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/lung_cancer_visium_subset/metadata.json",
        ),
        (
            "lung_cancer_visium_method_lr_rank_table",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/lung_cancer_visium_method_lr_rank_table.csv",
        ),
        (
            "lung_cancer_visium_method_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/lung_cancer_visium_method_control_summary.csv",
        ),
        (
            "lung_cancer_visium_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/lung_cancer_visium_baseline_panel/spatialself_wavelet_edge_prior/lung_cancer_visium_subset__spatialself_wavelet_predictions.csv",
        ),
        (
            "lung_cancer_visium_commot_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/lung_cancer_visium_baseline_panel/commot/lung_cancer_visium_subset__commot_predictions.csv",
        ),
        (
            "lung_cancer_visium_baseline_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/lung_cancer_visium_baseline_panel_time.log",
        ),
        (
            "slide_tags_tonsil_n1200_context_restricted_model_rerun_control",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_raw/slide_tags_tonsil_n1200_context_restricted_control_fdr_summary.csv",
        ),
        (
            "slide_tags_tonsil_n1200_context_restricted_model_rerun_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_raw/slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_audit.csv",
        ),
        (
            "slide_tags_tonsil_n1200_context_restricted_model_rerun_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_context_restricted_model_rerun_fdr_raw/slide_tags_tonsil_n1200_context_restricted_model_rerun_manifest.json",
        ),
        (
            "slide_tags_tonsil_n1200_matched_context_target_masks",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_matched_context_enrichment_fdr/slide_tags_tonsil_n1200_matched_context_target_masks.csv",
        ),
        (
            "slide_tags_tonsil_n1200_matched_context_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_matched_context_enrichment_fdr/slide_tags_tonsil_n1200_matched_context_control_fdr_summary.csv",
        ),
        (
            "slide_tags_tonsil_n1200_matched_context_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_matched_context_enrichment_fdr/slide_tags_tonsil_n1200_matched_context_fdr_audit.csv",
        ),
        (
            "slide_tags_tonsil_n1200_matched_context_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_n1200_matched_context_enrichment_fdr/slide_tags_tonsil_n1200_matched_context_manifest.json",
        ),
        (
            "post_xenium_human_colon_cancer_subset_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_subset__standardization_readiness.csv",
        ),
        (
            "post_xenium_human_colon_cancer_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/post_xenium_human_colon_cancer_subset/standardization_summary.csv",
        ),
        (
            "post_xenium_human_colon_cancer_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/post_xenium_human_colon_cancer_subset/metadata.json",
        ),
        (
            "post_xenium_human_colon_cancer_method_lr_rank_table",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_method_lr_rank_table.csv",
        ),
        (
            "post_xenium_human_colon_cancer_method_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_method_control_summary.csv",
        ),
        (
            "post_xenium_human_colon_cancer_method_runtime_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_method_runtime_summary.csv",
        ),
        (
            "post_xenium_human_colon_cancer_method_status_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_method_status_summary.csv",
        ),
        (
            "post_xenium_human_colon_cancer_default_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_baseline_panel/spatialself_wavelet_edge_prior/post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        ),
        (
            "post_xenium_human_colon_cancer_commot_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_baseline_panel/commot/post_xenium_human_colon_cancer_subset__commot_predictions.csv",
        ),
        (
            "post_xenium_human_colon_cancer_cellchat_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_extended_baselines/cellchat/post_xenium_human_colon_cancer_subset__cellchat_predictions.csv",
        ),
        (
            "post_xenium_human_colon_cancer_spatalk_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_extended_baselines/spatalk/post_xenium_human_colon_cancer_subset__spatalk_predictions.csv",
        ),
        (
            "post_xenium_human_colon_cancer_cellnest_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_extended_baselines/cellnest/post_xenium_human_colon_cancer_subset__cellnest_predictions.csv",
        ),
        (
            "post_xenium_human_colon_cancer_calibrated_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_spatialself_calibrated/spatialself_wavelet_edge_prior/post_xenium_human_colon_cancer_subset__spatialself_wavelet_predictions.csv",
        ),
        (
            "post_xenium_human_colon_cancer_default_panel_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_baseline_panel_time.log",
        ),
        (
            "post_xenium_human_colon_cancer_extended_baselines_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_extended_baselines_time.log",
        ),
        (
            "post_xenium_human_colon_cancer_calibrated_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_human_colon_cancer_spatialself_calibrated_time.log",
        ),
        (
            "post_xenium_colon_downstream_target_correlation_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_downstream_target_validation_50perm/post_xenium_colon_downstream_target_correlation.csv",
        ),
        (
            "post_xenium_colon_downstream_method_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_downstream_target_validation_50perm/post_xenium_colon_downstream_method_summary.csv",
        ),
        (
            "post_xenium_colon_downstream_claims_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_downstream_target_validation_50perm/post_xenium_colon_downstream_claims.csv",
        ),
        (
            "post_xenium_colon_prediction_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_prediction_fdr_50perm/post_xenium_colon_prediction_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_prediction_lr_pair_fdr_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_prediction_fdr_50perm/post_xenium_colon_prediction_lr_pair_fdr.csv",
        ),
        (
            "post_xenium_colon_prediction_control_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_prediction_fdr_50perm/post_xenium_colon_prediction_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_prediction_fdr_status_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_prediction_fdr_50perm/post_xenium_colon_prediction_fdr_status.csv",
        ),
        (
            "post_xenium_colon_targeted_region_fdr_audit_100perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_region_fdr_audit.csv",
        ),
        (
            "post_xenium_colon_targeted_region_fdr_control_summary_100perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_targeted_region_fdr_manifest_100perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_targeted_region_fdr_100perm/post_xenium_colon_targeted_region_fdr_manifest.json",
        ),
        (
            "post_xenium_colon_region_aware_targeted_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_region_aware_targeted_fdr_200perm/post_xenium_colon_region_aware_targeted_fdr_audit.csv",
        ),
        (
            "post_xenium_colon_region_aware_targeted_fdr_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_region_aware_targeted_fdr_200perm/post_xenium_colon_region_aware_targeted_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_region_aware_targeted_fdr_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_region_aware_targeted_fdr_200perm/post_xenium_colon_region_aware_targeted_fdr_manifest.json",
        ),
        (
            "post_xenium_colon_local_contrast_fdr_all_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_fdr_all.csv",
        ),
        (
            "post_xenium_colon_local_contrast_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_control_summary.csv",
        ),
        (
            "post_xenium_colon_local_contrast_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_audit.csv",
        ),
        (
            "post_xenium_colon_local_contrast_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_local_contrast_fdr_200perm/post_xenium_colon_local_contrast_manifest.json",
        ),
        (
            "post_xenium_colon_competitive_lr_pair_fdr_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_competitive_lr_fdr_1000perm/post_xenium_colon_competitive_lr_pair_fdr.csv",
        ),
        (
            "post_xenium_colon_competitive_control_fdr_summary_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_competitive_lr_fdr_1000perm/post_xenium_colon_competitive_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_competitive_degeneracy_diagnostics_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_competitive_lr_fdr_1000perm/post_xenium_colon_existing_null_degeneracy_diagnostics.csv",
        ),
        (
            "post_xenium_colon_competitive_fdr_manifest_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_competitive_lr_fdr_1000perm/post_xenium_colon_competitive_lr_fdr_manifest.json",
        ),
        (
            "post_xenium_colon_model_rerun_observed_lr_stats_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_observed_lr_stats.csv",
        ),
        (
            "post_xenium_colon_model_rerun_null_lr_stats_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_null_lr_stats.csv",
        ),
        (
            "post_xenium_colon_model_rerun_lr_pair_fdr_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_lr_pair_fdr.csv",
        ),
        (
            "post_xenium_colon_model_rerun_control_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_model_rerun_status_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_status.csv",
        ),
        (
            "post_xenium_colon_model_rerun_audit_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_fdr_audit.csv",
        ),
        (
            "post_xenium_colon_model_rerun_manifest_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_model_rerun_fdr_50perm/post_xenium_colon_model_rerun_fdr_manifest.json",
        ),
        (
            "post_xenium_colon_context_restricted_model_rerun_raw_control",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_context_restricted_model_rerun_fdr_raw/post_xenium_colon_context_restricted_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_context_restricted_model_rerun_raw_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_context_restricted_model_rerun_fdr_raw/post_xenium_colon_context_restricted_model_rerun_fdr_audit.csv",
        ),
        (
            "post_xenium_colon_context_restricted_model_rerun_fp_suppressed_control",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_context_restricted_model_rerun_fdr_fp_suppressed_diagnostic/post_xenium_colon_context_restricted_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_context_restricted_model_rerun_fp_suppressed_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_context_restricted_model_rerun_fdr_fp_suppressed_diagnostic/post_xenium_colon_context_restricted_model_rerun_fdr_audit.csv",
        ),
        (
            "post_xenium_colon_matched_context_target_masks",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_matched_context_enrichment_fdr/post_xenium_colon_matched_context_target_masks.csv",
        ),
        (
            "post_xenium_colon_matched_context_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_matched_context_enrichment_fdr/post_xenium_colon_matched_context_control_fdr_summary.csv",
        ),
        (
            "post_xenium_colon_matched_context_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_matched_context_enrichment_fdr/post_xenium_colon_matched_context_fdr_audit.csv",
        ),
        (
            "post_xenium_colon_matched_context_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_matched_context_enrichment_fdr/post_xenium_colon_matched_context_manifest.json",
        ),
        (
            "post_xenium_colon_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_pathway_specific_downstream/post_xenium_colon_pathway_specific_downstream_summary.csv",
        ),
        (
            "post_xenium_colon_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_pathway_specific_downstream/post_xenium_colon_pathway_specific_downstream_claims.csv",
        ),
        (
            "post_xenium_colon_pathway_specific_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_pathway_specific_downstream/post_xenium_colon_pathway_specific_downstream_manifest.json",
        ),
        (
            "post_xenium_model_calibration_diagnostic_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_model_calibration_diagnostic_summary.csv",
        ),
        (
            "post_xenium_model_calibration_diagnostic_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_model_calibration_diagnostic_manifest.json",
        ),
        (
            "visium_hd_crc_p2crc_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_subset/standardization_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_subset/metadata.json",
        ),
        (
            "visium_hd_crc_p2crc_method_lr_rank_table",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_method_lr_rank_table.csv",
        ),
        (
            "visium_hd_crc_p2crc_method_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_method_control_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_method_runtime_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_method_runtime_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_method_status_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_method_status_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_default_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_baseline_panel/spatialself_wavelet_edge_prior/visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
        ),
        (
            "visium_hd_crc_p2crc_calibrated_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_spatialself_calibrated/spatialself_wavelet_edge_prior/visium_hd_crc_p2crc_subset__spatialself_wavelet_predictions.csv",
        ),
        (
            "visium_hd_crc_p2crc_commot_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_baseline_panel/commot/visium_hd_crc_p2crc_subset__commot_predictions.csv",
        ),
        (
            "visium_hd_crc_p2crc_cellchat_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_baseline_panel/cellchat/visium_hd_crc_p2crc_subset__cellchat_predictions.csv",
        ),
        (
            "visium_hd_crc_p2crc_cellnest_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_baseline_panel/cellnest/visium_hd_crc_p2crc_subset__cellnest_predictions.csv",
        ),
        (
            "visium_hd_crc_p2crc_baseline_panel_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_baseline_panel_time.log",
        ),
        (
            "visium_hd_crc_p2crc_calibrated_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_spatialself_calibrated_time.log",
        ),
        (
            "visium_hd_crc_p2crc_downstream_target_correlation",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_downstream_target_validation/visium_hd_crc_p2crc_downstream_target_correlation.csv",
        ),
        (
            "visium_hd_crc_p2crc_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_downstream_target_validation/visium_hd_crc_p2crc_downstream_method_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_downstream_target_validation/visium_hd_crc_p2crc_downstream_claims.csv",
        ),
        (
            "visium_hd_crc_p2crc_competitive_downstream_correlation_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_correlation.csv",
        ),
        (
            "visium_hd_crc_p2crc_competitive_downstream_method_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_method_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_competitive_downstream_claims_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_claims.csv",
        ),
        (
            "visium_hd_crc_p2crc_competitive_downstream_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_manifest.json",
        ),
        (
            "visium_hd_crc_p2crc_prediction_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_prediction_fdr_50perm/visium_hd_crc_p2crc_prediction_fdr_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_prediction_control_fdr_summary_50perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_prediction_fdr_50perm/visium_hd_crc_p2crc_prediction_control_fdr_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_region_aware_targeted_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/visium_hd_crc_p2crc_region_aware_targeted_fdr_audit.csv",
        ),
        (
            "visium_hd_crc_p2crc_region_aware_targeted_fdr_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/visium_hd_crc_p2crc_region_aware_targeted_control_fdr_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_region_aware_targeted_fdr_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_region_aware_targeted_fdr_200perm/visium_hd_crc_p2crc_region_aware_targeted_fdr_manifest.json",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset/standardization_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset/metadata.json",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_target_signatures",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/visium_hd_crc_p2crc_rich_targets_subset/target_signatures.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_downstream_validation/visium_hd_crc_p2crc_downstream_method_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_downstream_validation/visium_hd_crc_p2crc_downstream_claims.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_downstream_validation/visium_hd_crc_p2crc_downstream_manifest.json",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_competitive_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_method_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_competitive_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_claims.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_competitive_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_competitive_downstream_200perm/visium_hd_crc_competitive_downstream_manifest.json",
        ),
        (
            "visium_hd_crc_p2crc_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_pathway_specific_downstream/visium_hd_crc_p2crc_pathway_specific_downstream_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_pathway_specific_downstream/visium_hd_crc_p2crc_pathway_specific_downstream_claims.csv",
        ),
        (
            "visium_hd_crc_p2crc_pathway_specific_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_pathway_specific_downstream/visium_hd_crc_p2crc_pathway_specific_downstream_manifest.json",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_fully_residual_pathway_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_fully_residual_pathway_downstream/visium_hd_crc_p2crc_pathway_specific_downstream_summary.csv",
        ),
        (
            "visium_hd_crc_p2crc_rich_targets_fully_residual_pathway_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc_p2crc_rich_targets_fully_residual_pathway_downstream/visium_hd_crc_p2crc_pathway_specific_downstream_claims.csv",
        ),
        (
            "slide_tags_tonsil_targeted_region_fdr_audit_100perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_targeted_region_fdr_100perm/slide_tags_tonsil_targeted_region_fdr_audit.csv",
        ),
        (
            "slide_tags_tonsil_targeted_region_fdr_control_summary_100perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_targeted_region_fdr_100perm/slide_tags_tonsil_targeted_control_fdr_summary.csv",
        ),
        (
            "slide_tags_tonsil_targeted_region_fdr_manifest_100perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_targeted_region_fdr_100perm/slide_tags_tonsil_targeted_region_fdr_manifest.json",
        ),
        (
            "slide_tags_tonsil_targeted_region_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_targeted_region_fdr_200perm/slide_tags_tonsil_targeted_region_fdr_audit.csv",
        ),
        (
            "slide_tags_tonsil_targeted_region_fdr_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_targeted_region_fdr_200perm/slide_tags_tonsil_targeted_control_fdr_summary.csv",
        ),
        (
            "slide_tags_tonsil_targeted_region_fdr_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_targeted_region_fdr_200perm/slide_tags_tonsil_targeted_region_fdr_manifest.json",
        ),
        (
            "slide_tags_tonsil_competitive_lr_pair_fdr_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_competitive_lr_fdr_1000perm/slide_tags_tonsil_competitive_lr_pair_fdr.csv",
        ),
        (
            "slide_tags_tonsil_competitive_control_fdr_summary_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_competitive_lr_fdr_1000perm/slide_tags_tonsil_competitive_control_fdr_summary.csv",
        ),
        (
            "slide_tags_tonsil_competitive_degeneracy_diagnostics_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_competitive_lr_fdr_1000perm/slide_tags_tonsil_existing_null_degeneracy_diagnostics.csv",
        ),
        (
            "slide_tags_tonsil_competitive_fdr_manifest_1000perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/slide_tags_tonsil_competitive_lr_fdr_1000perm/slide_tags_tonsil_competitive_lr_fdr_manifest.json",
        ),
        (
            "spamosaic_tonsil_A1_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_subset/standardization_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_subset/metadata.json",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/standardization_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/metadata.json",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_target_signatures",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/target_signatures.csv",
        ),
        (
            "spamosaic_tonsil_A1_method_lr_rank_table",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_method_lr_rank_table.csv",
        ),
        (
            "spamosaic_tonsil_A1_method_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_method_control_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_method_runtime_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_method_runtime_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_spatialself/spatialself_wavelet_edge_prior/replicate_00__spatialself_wavelet_predictions.csv",
        ),
        (
            "spamosaic_tonsil_A1_commot_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_commot/replicate_00__commot_predictions.csv",
        ),
        (
            "spamosaic_tonsil_A1_cellchat_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_cellchat/replicate_00__cellchat_predictions.csv",
        ),
        (
            "spamosaic_tonsil_A1_spatalk_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_spatalk/replicate_00__spatalk_predictions.csv",
        ),
        (
            "spamosaic_tonsil_A1_cellnest_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_cellnest/replicate_00__cellnest_predictions.csv",
        ),
        (
            "spamosaic_tonsil_A1_downstream_target_correlation",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_target_correlation.csv",
        ),
        (
            "spamosaic_tonsil_A1_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_downstream_target_validation/spamosaic_tonsil_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_A1_restricted_downstream_correlation",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_restricted_downstream_validation/spamosaic_tonsil_restricted_downstream_correlation.csv",
        ),
        (
            "spamosaic_tonsil_A1_restricted_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_restricted_downstream_validation/spamosaic_tonsil_restricted_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_restricted_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_restricted_downstream_validation/spamosaic_tonsil_restricted_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_A1_targeted_region_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_region_fdr_audit.csv",
        ),
        (
            "spamosaic_tonsil_A1_targeted_region_fdr_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_control_fdr_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_targeted_region_fdr_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_targeted_region_fdr_200perm/spamosaic_tonsil_A1_targeted_region_fdr_manifest.json",
        ),
        (
            "spamosaic_tonsil_A1_local_contrast_fdr_all_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_fdr_all.csv",
        ),
        (
            "spamosaic_tonsil_A1_local_contrast_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_control_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_local_contrast_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_audit.csv",
        ),
        (
            "spamosaic_tonsil_A1_local_contrast_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_local_contrast_fdr_200perm/spamosaic_tonsil_A1_local_contrast_manifest.json",
        ),
        (
            "spamosaic_tonsil_A1_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_pathway_specific_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_pathway_specific_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_A1_pathway_specific_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_pathway_specific_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_manifest.json",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_downstream_validation/spamosaic_tonsil_A1_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_downstream_validation/spamosaic_tonsil_A1_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_downstream_validation/spamosaic_tonsil_A1_downstream_manifest.json",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_restricted_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_restricted_downstream_validation/spamosaic_tonsil_A1_restricted_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_restricted_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_restricted_downstream_validation/spamosaic_tonsil_A1_restricted_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_pathway_specific_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_pathway_specific_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_fully_residual_pathway_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_fully_residual_pathway_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_summary.csv",
        ),
        (
            "spamosaic_tonsil_A1_rich_targets_fully_residual_pathway_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_A1_rich_targets_fully_residual_pathway_downstream/spamosaic_tonsil_A1_pathway_specific_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_subset/standardization_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_subset/metadata.json",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_rich_targets_subset/standardization_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_rich_targets_subset/metadata.json",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_target_signatures",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_D1_rich_targets_subset/target_signatures.csv",
        ),
        (
            "spamosaic_tonsil_D1_method_lr_rank_table",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_method_lr_rank_table.csv",
        ),
        (
            "spamosaic_tonsil_D1_method_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_method_control_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_method_runtime_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_method_runtime_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_spatialself/spatialself_wavelet_edge_prior/replicate_00__spatialself_wavelet_predictions.csv",
        ),
        (
            "spamosaic_tonsil_D1_commot_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_commot/replicate_00__commot_predictions.csv",
        ),
        (
            "spamosaic_tonsil_D1_cellchat_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_cellchat/replicate_00__cellchat_predictions.csv",
        ),
        (
            "spamosaic_tonsil_D1_spatalk_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_spatalk/replicate_00__spatalk_predictions.csv",
        ),
        (
            "spamosaic_tonsil_D1_cellnest_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_cellnest/replicate_00__cellnest_predictions.csv",
        ),
        (
            "spamosaic_tonsil_D1_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_downstream_target_validation/spamosaic_tonsil_D1_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_downstream_target_validation/spamosaic_tonsil_D1_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_restricted_downstream_correlation",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_restricted_downstream_validation/spamosaic_tonsil_D1_restricted_downstream_correlation.csv",
        ),
        (
            "spamosaic_tonsil_D1_restricted_downstream_method_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_restricted_downstream_validation/spamosaic_tonsil_D1_restricted_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_restricted_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_restricted_downstream_validation/spamosaic_tonsil_D1_restricted_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_targeted_region_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_targeted_region_fdr_200perm/spamosaic_tonsil_D1_targeted_region_fdr_audit.csv",
        ),
        (
            "spamosaic_tonsil_D1_targeted_region_fdr_control_summary_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_targeted_region_fdr_200perm/spamosaic_tonsil_D1_targeted_control_fdr_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_targeted_region_fdr_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_targeted_region_fdr_200perm/spamosaic_tonsil_D1_targeted_region_fdr_manifest.json",
        ),
        (
            "spamosaic_tonsil_D1_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_pathway_specific_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_pathway_specific_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_pathway_specific_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_pathway_specific_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_manifest.json",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_downstream_validation/spamosaic_tonsil_D1_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_downstream_validation/spamosaic_tonsil_D1_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_downstream_validation/spamosaic_tonsil_D1_downstream_manifest.json",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_restricted_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_restricted_downstream_validation/spamosaic_tonsil_D1_restricted_downstream_method_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_restricted_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_restricted_downstream_validation/spamosaic_tonsil_D1_restricted_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_pathway_specific_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_pathway_specific_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_claims.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_fully_residual_pathway_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_fully_residual_pathway_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_summary.csv",
        ),
        (
            "spamosaic_tonsil_D1_rich_targets_fully_residual_pathway_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/spamosaic_tonsil_D1_rich_targets_fully_residual_pathway_downstream/spamosaic_tonsil_D1_pathway_specific_downstream_claims.csv",
        ),
        (
            "melanoma_real_background_constructed_cells",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/constructed_replicate/cells.csv",
        ),
        (
            "melanoma_real_background_constructed_lr_pairs",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/constructed_replicate/lr_pairs.csv",
        ),
        (
            "melanoma_real_background_constructed_candidates",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/constructed_replicate/candidate_interactions.csv",
        ),
        (
            "melanoma_real_background_constructed_expression",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/constructed_replicate/expression.csv",
        ),
        (
            "melanoma_real_background_spatialself_specific_candidates",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_expanded_melanoma_spatialself_specific_candidates.csv",
        ),
        (
            "melanoma_real_background_expanded_audit",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_expanded_melanoma_audit.csv",
        ),
        (
            "melanoma_real_background_targeted_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv",
        ),
        (
            "melanoma_real_background_targeted_fdr_manifest_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit_200perm_manifest.json",
        ),
        (
            "melanoma_real_background_targeted_fdr_time_log",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen_time_rerun.log",
        ),
        (
            "melanoma_real_background_model_rerun_observed_lr_stats_20perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_model_rerun_fdr_20perm/post_xenium_colon_model_rerun_observed_lr_stats.csv",
        ),
        (
            "melanoma_real_background_model_rerun_null_lr_stats_20perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_model_rerun_fdr_20perm/post_xenium_colon_model_rerun_null_lr_stats.csv",
        ),
        (
            "melanoma_real_background_model_rerun_lr_pair_fdr_20perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_model_rerun_fdr_20perm/post_xenium_colon_model_rerun_lr_pair_fdr.csv",
        ),
        (
            "melanoma_real_background_model_rerun_control_summary_20perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_model_rerun_fdr_20perm/post_xenium_colon_model_rerun_control_fdr_summary.csv",
        ),
        (
            "melanoma_real_background_model_rerun_audit_20perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_model_rerun_fdr_20perm/post_xenium_colon_model_rerun_fdr_audit.csv",
        ),
        (
            "melanoma_real_background_model_rerun_manifest_20perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_model_rerun_fdr_20perm/post_xenium_colon_model_rerun_fdr_manifest.json",
        ),
        (
            "melanoma_real_background_matched_context_target_masks",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_matched_context_enrichment_fdr/melanoma_real_background_matched_context_target_masks.csv",
        ),
        (
            "melanoma_real_background_matched_context_control_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_matched_context_enrichment_fdr/melanoma_real_background_matched_context_control_fdr_summary.csv",
        ),
        (
            "melanoma_real_background_matched_context_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_matched_context_enrichment_fdr/melanoma_real_background_matched_context_fdr_audit.csv",
        ),
        (
            "melanoma_real_background_matched_context_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_matched_context_enrichment_fdr/melanoma_real_background_matched_context_manifest.json",
        ),
        (
            "melanoma_real_background_hardneg_feasgate_model_rerun_observed_lr_stats_10perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/post_xenium_colon_model_rerun_observed_lr_stats.csv",
        ),
        (
            "melanoma_real_background_hardneg_feasgate_model_rerun_null_lr_stats_10perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/post_xenium_colon_model_rerun_null_lr_stats.csv",
        ),
        (
            "melanoma_real_background_hardneg_feasgate_model_rerun_control_summary_10perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/post_xenium_colon_model_rerun_control_fdr_summary.csv",
        ),
        (
            "melanoma_real_background_hardneg_feasgate_model_rerun_audit_10perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/post_xenium_colon_model_rerun_fdr_audit.csv",
        ),
        (
            "melanoma_real_background_hardneg_feasgate_model_rerun_manifest_10perm",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/melanoma_real_background_hardneg_feasgate_model_rerun_fdr_10perm/post_xenium_colon_model_rerun_fdr_manifest.json",
        ),
        (
            "phase7_goal5_functional_success_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_goal5_functional_success_audit.csv",
        ),
        (
            "phase7_goal5_functional_success_audit_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_goal5_functional_success_audit_summary.json",
        ),
        (
            "phase7_goal5_functional_fdr_closure_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_goal5_functional_fdr_closure_audit.csv",
        ),
        (
            "phase7_goal5_functional_fdr_closure_audit_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/phase7_goal5_functional_fdr_closure_audit_summary.json",
        ),
        (
            "goal4_melanoma_targeted_fdr_audit",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit.csv",
        ),
        (
            "goal4_melanoma_targeted_fdr_audit_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit_manifest.json",
        ),
        (
            "goal4_melanoma_targeted_fdr_audit_200perm",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit_200perm.csv",
        ),
        (
            "goal4_melanoma_targeted_fdr_audit_200perm_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal4_expanded_melanoma_subset_screen/phase7_goal4_melanoma_targeted_fdr_audit_200perm_manifest.json",
        ),
        (
            "native_human_melanoma_multiome_subset_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/standardization_summary.csv",
        ),
        (
            "native_human_melanoma_multiome_subset_metadata",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/metadata.json",
        ),
        (
            "native_human_melanoma_multiome_baseline_status",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/native_melanoma_baseline_status.csv",
        ),
        (
            "native_human_melanoma_multiome_control_ranking",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/native_melanoma_control_ranking.csv",
        ),
        (
            "native_human_melanoma_multiome_competitive_downstream",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/native_melanoma_competitive_downstream_summary.csv",
        ),
        (
            "native_human_melanoma_multiome_targeted_fdr",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/native_melanoma_targeted_fdr_audit.csv",
        ),
        (
            "native_human_melanoma_multiome_residual_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/residual_downstream_200perm/native_melanoma_residual_downstream_method_summary.csv",
        ),
        (
            "native_human_melanoma_multiome_residual_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/residual_downstream_200perm/native_melanoma_residual_downstream_claims.csv",
        ),
        (
            "native_human_melanoma_multiome_residual_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/residual_downstream_200perm/native_melanoma_residual_downstream_manifest.json",
        ),
        (
            "native_human_melanoma_multiome_claim_audit",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/native_melanoma_claim_audit.csv",
        ),
        (
            "native_human_melanoma_multiome_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/native_melanoma_validation_manifest.json",
        ),
        (
            "native_human_melanoma_multiome_spatialself_predictions",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome/baseline_panel/spatialself_wavelet_edge_prior/native_human_melanoma_multiome_subset__spatialself_wavelet_edge_prior_predictions.csv",
        ),
        (
            "native_human_melanoma_multiome_pathway_specific_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome_pathway_specific_downstream/native_human_melanoma_multiome_pathway_specific_downstream_summary.csv",
        ),
        (
            "native_human_melanoma_multiome_pathway_specific_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome_pathway_specific_downstream/native_human_melanoma_multiome_pathway_specific_downstream_claims.csv",
        ),
        (
            "native_human_melanoma_multiome_pathway_specific_downstream_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome_pathway_specific_downstream/native_human_melanoma_multiome_pathway_specific_downstream_manifest.json",
        ),
        (
            "native_human_melanoma_multiome_fully_residual_pathway_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome_fully_residual_pathway_downstream/native_human_melanoma_multiome_pathway_specific_downstream_summary.csv",
        ),
        (
            "native_human_melanoma_multiome_fully_residual_pathway_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/native_human_melanoma_multiome_fully_residual_pathway_downstream/native_human_melanoma_multiome_pathway_specific_downstream_claims.csv",
        ),
        (
            "post_xenium_colon_fully_residual_pathway_downstream_summary",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_fully_residual_pathway_downstream/post_xenium_colon_pathway_specific_downstream_summary.csv",
        ),
        (
            "post_xenium_colon_fully_residual_pathway_downstream_claims",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/post_xenium_colon_fully_residual_pathway_downstream/post_xenium_colon_pathway_specific_downstream_claims.csv",
        ),
        (
            "xenium_skin_melanoma_access_status",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_skin_melanoma_access_status.csv",
        ),
        (
            "xenium_skin_melanoma_access_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_skin_melanoma_access_manifest.json",
        ),
        (
            "xenium_human_lymph_node_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/xenium_human_lymph_node__standardization_readiness.csv",
        ),
        (
            "visium_hd_crc_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/visium_hd_crc__standardization_readiness.csv",
        ),
        (
            "cosmx_melanoma_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/cosmx_melanoma__standardization_readiness.csv",
        ),
        (
            "cosmx_melanoma_standardization_contract",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/cosmx_melanoma__standardization_contract.json",
        ),
        (
            "cosmx_melanoma_dryad_access_status",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/cosmx_melanoma_dryad_access_status.csv",
        ),
        (
            "cosmx_melanoma_dryad_file_access",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/cosmx_melanoma_dryad_file_access.csv",
        ),
        (
            "cosmx_melanoma_dryad_access_manifest",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/cosmx_melanoma_dryad_access_manifest.json",
        ),
        (
            "merfish_seqfish_slideseqv2_standardization_readiness",
            "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/merfish_or_seqfish_or_slideseqv2__standardization_readiness.csv",
        ),
        ("tonsil_scaling_summary", "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/tonsil_scaling_summary.csv"),
        ("tonsil_n1800_summary", "outputs/phase2/goal4_high_resolution/tonsil_scaling_wavelet_calibrated/n1800/tonsil_fresh_goal4_summary.csv"),
        ("tonsil_baseline_matched_summary", "outputs/phase2/goal4_high_resolution/tonsil_baseline_matched_fresh_subset/tonsil_baseline_matched_summary.csv"),
        ("tonsil_module_overview", "outputs/phase2/goal4_high_resolution/tonsil_ccc_modules_n1800_subset/tonsil_ccc_module_overview.csv"),
    ]
    return pd.DataFrame(
        [{"source_id": source_id, "path": path, "exists": (PROJECT_ROOT / path).exists()} for source_id, path in rows]
    )


def _find_path_matches(hints: list[str]) -> list[str]:
    matches = []
    roots = [
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "high_spatial",
        PROJECT_ROOT / "HumanMelannomaMultiome",
        PROJECT_ROOT / "lung_cancer",
        PROJECT_ROOT / "mouse",
        PROJECT_ROOT / "visium_lr_benchmark",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and not _looks_like_standardized_dataset(path):
                continue
            name = str(path.relative_to(PROJECT_ROOT))
            lowered = name.lower()
            if any(
                part in lowered
                for part in (
                    ".spatialvenv",
                    "site-packages",
                    "__pycache__",
                    ".conda",
                    ".git",
                    "__standardization_readiness.csv",
                    "__standardization_contract.json",
                )
            ):
                continue
            if (
                "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data" in lowered
                and "standardized_inputs" not in lowered
            ):
                continue
            if any(hint.lower() in lowered for hint in hints):
                matches.append(name)
            if len(matches) >= 25:
                return sorted(set(matches))
    return sorted(set(matches))


def _looks_like_standardized_dataset(path: Path) -> bool:
    if "standardized_inputs" not in str(path):
        return True
    if path.is_file():
        parent = path.parent
    else:
        parent = path
    required = ["cells.csv", "expression.csv", "lr_pairs.csv", "candidate_interactions.csv"]
    return all((parent / filename).exists() for filename in required)


def _has_standard_schema_match(matches: list[str]) -> bool:
    for match in matches:
        path = PROJECT_ROOT / match
        parent = path if path.is_dir() else path.parent
        if _looks_like_standardized_dataset(parent):
            return True
    return False


def _read_first(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key, "")
    if pd.isna(value):
        return ""
    return value


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
