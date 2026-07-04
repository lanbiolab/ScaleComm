#!/usr/bin/env python
"""Audit Phase 12 whole-LR atlas readiness and hard-negative templates."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import h5py
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase12/atlas_readiness"

GENE_RE = re.compile(r"^[A-Z][A-Z0-9.-]{1,}$")
HOUSEKEEPING_LIGANDS = ["ACTB", "GAPDH", "RPLP0", "EEF1A1", "B2M", "FTL", "UBC", "MALAT1", "COL1A1"]
HARD_NEGATIVE_RECEIVERS = ["CCR7", "CXCR5", "CXCR4", "CXCR3", "CD44", "CD274", "CD40", "EGFR", "ITGB1", "KRT8", "EPCAM"]
POSITIVE_LR_ANCHORS = [
    ("SPP1", "CD44", "myeloid_remodeling"),
    ("CCL19", "CCR7", "chemokine_recruitment"),
    ("CCL21", "CCR7", "chemokine_recruitment"),
    ("CXCL13", "CXCR5", "chemokine_recruitment"),
    ("CXCL12", "CXCR4", "chemokine_recruitment"),
    ("CXCL9", "CXCR3", "tumor_immune_chemokine"),
    ("CXCL10", "CXCR3", "tumor_immune_chemokine"),
    ("CD48", "CD244", "contact_dependent_immune"),
    ("CD40LG", "CD40", "contact_dependent_immune"),
    ("PDCD1", "CD274", "immune_checkpoint"),
    ("CD274", "PDCD1", "immune_checkpoint"),
    ("VEGFA", "KDR", "angiogenesis"),
    ("TGFB1", "TGFBR2", "ecm_stromal_remodeling"),
]


DATASETS = [
    {
        "dataset_key": "visium_hd_crc_full",
        "label": "Visium HD CRC P2CRC full-transcriptome",
        "platform": "Visium HD",
        "tissue": "CRC",
        "kind": "native_full_transcriptome",
        "genes_csv": "outputs/phase10/full_transcriptome_commot_style/visium_hd_crc_p2crc_subset/full_transcriptome/genes.csv",
    },
    {
        "dataset_key": "post_xenium_colon_full",
        "label": "Post-Xenium/CytAssist colon cancer full-transcriptome",
        "platform": "Visium CytAssist",
        "tissue": "colon_cancer",
        "kind": "native_full_transcriptome",
        "genes_csv": "outputs/phase10/full_transcriptome_commot_style/post_xenium_human_colon_cancer_subset/full_transcriptome/genes.csv",
    },
    {
        "dataset_key": "human_lymph_node_visium_raw_full",
        "label": "10x human lymph node Visium raw full-transcriptome",
        "platform": "Visium",
        "tissue": "lymph_node",
        "kind": "native_full_transcriptome_raw",
        "h5": "data/optional/human_lymph_node_visium_full/raw/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5",
    },
    {
        "dataset_key": "human_lymph_node_phase11_working",
        "label": "Human lymph node Phase 11 working subset",
        "platform": "Visium",
        "tissue": "lymph_node",
        "kind": "curated_lr_marker_target_subset",
        "expression_csv": "outputs/phase11/commot_style_spatial_story/human_lymph_node_visium_full_subset/dataset/expression.csv",
    },
    {
        "dataset_key": "native_melanoma_multiome_targeted",
        "label": "Native human melanoma multiome targeted subset",
        "platform": "multiome",
        "tissue": "melanoma",
        "kind": "targeted_subset",
        "expression_csv": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/expression.csv",
    },
    {
        "dataset_key": "spamosaic_tonsil_A1_rich_targeted",
        "label": "SPAMosaic tonsil A1 rich-target subset",
        "platform": "SPAMosaic/Slide-tags-like",
        "tissue": "tonsil",
        "kind": "targeted_subset",
        "expression_csv": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/expression.csv",
    },
    {
        "dataset_key": "xenium_human_lymph_node_targeted",
        "label": "Xenium human lymph node targeted panel",
        "platform": "Xenium",
        "tissue": "lymph_node",
        "kind": "targeted_panel",
        "expression_csv": "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/expression.csv",
    },
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr_universe = collect_lr_universe()
    datasets = collect_dataset_gene_sets()
    coverage = dataset_coverage(lr_universe, datasets)
    hard_negatives = build_hard_negative_templates(datasets)
    goal_audit = build_goal_audit(datasets, lr_universe, coverage, hard_negatives)

    lr_universe.to_csv(OUT_DIR / "phase12_lr_universe.csv", index=False)
    coverage.to_csv(OUT_DIR / "phase12_dataset_lr_coverage.csv", index=False)
    hard_negatives.to_csv(OUT_DIR / "phase12_hard_negative_templates.csv", index=False)
    goal_audit.to_csv(OUT_DIR / "phase12_goal_audit.csv", index=False)
    write_status(datasets, lr_universe, coverage, hard_negatives, goal_audit)
    write_manifest(datasets, lr_universe, coverage, hard_negatives)

    print("LR universe:", len(lr_universe))
    print(coverage.to_string(index=False))
    print()
    print(goal_audit.to_string(index=False))


def collect_lr_universe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    source_paths = [
        PROJECT_ROOT / "configs/phase4_curated_lr_targets.csv",
        PROJECT_ROOT / "HumanMelannomaMultiome/melanoma_lr_benchmark/lr_table.csv",
        *PROJECT_ROOT.glob("outputs/phase7/**/lr_pairs.csv"),
        *PROJECT_ROOT.glob("outputs/phase10/**/nichenet_targets/lr_pairs__nichenet_targets.csv"),
        *PROJECT_ROOT.glob("outputs/phase11/**/lr_pairs.csv"),
        *PROJECT_ROOT.glob("outputs/phase6/**/lr_pairs.csv"),
    ]
    for path in sorted(set(source_paths)):
        if not path.exists():
            continue
        try:
            table = pd.read_csv(path)
        except Exception:
            continue
        if {"ligand", "receptor"}.issubset(table.columns):
            for row in table.itertuples(index=False):
                ligand = normalize_gene(getattr(row, "ligand", ""))
                receptor = normalize_gene(getattr(row, "receptor", ""))
                if not valid_gene_pair(ligand, receptor):
                    continue
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "lr_pair": f"{ligand}-{receptor}",
                        "source_path": rel(path),
                        "source_type": classify_source(path),
                        "original_lr_id": str(getattr(row, "lr_id", "")),
                        "original_mechanism": str(getattr(row, "mechanism", "")),
                        "original_control_class": str(getattr(row, "control_class", "")),
                        "original_true_lr": safe_bool(getattr(row, "true_lr", "")),
                        "original_false_positive_high_expression": safe_bool(
                            getattr(row, "false_positive_high_expression", "")
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["ligand", "receptor", "lr_pair"])
    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["ligand", "receptor", "lr_pair"], as_index=False)
        .agg(
            n_sources=("source_path", "nunique"),
            source_paths=("source_path", join_unique),
            source_types=("source_type", join_unique),
            mechanisms=("original_mechanism", join_nonempty_unique),
            control_classes=("original_control_class", join_nonempty_unique),
            any_prior_true_lr=("original_true_lr", "max"),
            any_prior_high_expression_negative=("original_false_positive_high_expression", "max"),
        )
        .sort_values(["n_sources", "lr_pair"], ascending=[False, True])
    )
    grouped["biological_category"] = [classify_biology(l, r) for l, r in zip(grouped["ligand"], grouped["receptor"])]
    grouped["phase12_priority_anchor"] = grouped["lr_pair"].isin({f"{l}-{r}" for l, r, _ in POSITIVE_LR_ANCHORS})
    return grouped


def collect_dataset_gene_sets() -> list[dict[str, object]]:
    datasets = []
    for spec in DATASETS:
        genes: set[str] = set()
        path = None
        if "genes_csv" in spec:
            path = PROJECT_ROOT / str(spec["genes_csv"])
            genes = read_genes_csv(path)
        elif "expression_csv" in spec:
            path = PROJECT_ROOT / str(spec["expression_csv"])
            genes = read_expression_header(path)
        elif "h5" in spec:
            path = PROJECT_ROOT / str(spec["h5"])
            genes = read_10x_h5_genes(path)
        datasets.append(
            {
                **spec,
                "path": rel(path) if path is not None else "",
                "path_exists": bool(path and path.exists()),
                "n_genes": len(genes),
                "genes": genes,
            }
        )
    return datasets


def dataset_coverage(lr_universe: pd.DataFrame, datasets: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        genes = dataset["genes"]
        present = lr_universe["ligand"].isin(genes) & lr_universe["receptor"].isin(genes)
        priority = lr_universe["phase12_priority_anchor"].astype(bool)
        high_negative = lr_universe["any_prior_high_expression_negative"].astype(bool)
        receptor_shared_risk = lr_universe["receptor"].isin(["CCR7", "CXCR4", "CXCR3", "CD44", "CD274"])
        categories = lr_universe.loc[present, "biological_category"].value_counts().to_dict()
        rows.append(
            {
                "dataset_key": dataset["dataset_key"],
                "label": dataset["label"],
                "platform": dataset["platform"],
                "tissue": dataset["tissue"],
                "kind": dataset["kind"],
                "path_exists": dataset["path_exists"],
                "n_genes": dataset["n_genes"],
                "n_lr_universe": int(len(lr_universe)),
                "n_lr_both_genes_present": int(present.sum()),
                "lr_coverage_fraction": float(present.mean()) if len(lr_universe) else 0.0,
                "n_priority_anchor_present": int((present & priority).sum()),
                "n_prior_high_expression_negative_present": int((present & high_negative).sum()),
                "n_receptor_shared_risk_present": int((present & receptor_shared_risk).sum()),
                "top_biological_categories": json.dumps(categories, sort_keys=True),
                "source_path": dataset["path"],
            }
        )
    return pd.DataFrame(rows)


def build_hard_negative_templates(datasets: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        genes = dataset["genes"]
        for ligand, receptor, category in POSITIVE_LR_ANCHORS:
            if ligand not in genes or receptor not in genes:
                continue
            positive = f"{ligand}-{receptor}"
            for hk in HOUSEKEEPING_LIGANDS:
                if hk in genes and hk != ligand:
                    rows.append(template_row(dataset, positive, "housekeeping_ligand_same_receptor", hk, receptor, category))
            for alt_receptor in HARD_NEGATIVE_RECEIVERS:
                if alt_receptor in genes and alt_receptor != receptor:
                    rows.append(template_row(dataset, positive, "same_ligand_wrong_receptor", ligand, alt_receptor, category))
            for hk in HOUSEKEEPING_LIGANDS:
                for alt_receptor in HARD_NEGATIVE_RECEIVERS:
                    if hk in genes and alt_receptor in genes and (hk, alt_receptor) != (ligand, receptor):
                        rows.append(template_row(dataset, positive, "housekeeping_ligand_wrong_receptor", hk, alt_receptor, category))
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).drop_duplicates(
        ["dataset_key", "positive_lr_pair", "negative_type", "negative_ligand", "negative_receptor"]
    )
    return table.sort_values(["dataset_key", "positive_lr_pair", "negative_type", "negative_pair"])


def template_row(dataset: dict[str, object], positive: str, negative_type: str, ligand: str, receptor: str, category: str) -> dict[str, object]:
    return {
        "dataset_key": dataset["dataset_key"],
        "dataset_kind": dataset["kind"],
        "positive_lr_pair": positive,
        "positive_category": category,
        "negative_type": negative_type,
        "negative_ligand": ligand,
        "negative_receptor": receptor,
        "negative_pair": f"{ligand}-{receptor}",
        "is_receptor_shared": positive.split("-")[1] == receptor,
        "is_ligand_shared": positive.split("-")[0] == ligand,
        "available": True,
    }


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_globbed_csv(root: Path, pattern: str) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    tables = []
    for path in sorted(root.glob(pattern)):
        table = read_optional_csv(path)
        if not table.empty:
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def fmt_short(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3g}"
    return str(value)


def build_goal_audit(
    datasets: list[dict[str, object]],
    lr_universe: pd.DataFrame,
    coverage: pd.DataFrame,
    hard_negatives: pd.DataFrame,
) -> pd.DataFrame:
    full_datasets = coverage[coverage["kind"].str.contains("full_transcriptome", na=False)]
    atlas_ready = full_datasets[full_datasets["n_lr_both_genes_present"].ge(100)]
    expanded_summary = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/atlas_model_expanded/phase12_expanded_atlas_calibrated_dataset_summary.csv"
    )
    expanded_perm = read_optional_csv(
        PROJECT_ROOT
        / "outputs/phase12/atlas_model_expanded/permutation_audit/phase12_expanded_atlas_permutation_dataset_summary.csv"
    )
    gene_programs = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/gene_program_enrichment/phase12_gene_program_enrichment_summary.csv"
    )
    downstream = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/atlas_downstream_scan/phase12_atlas_downstream_lr_summary.csv"
    )
    pathway_specific = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/pathway_specific_programs/phase12_pathway_specific_dataset_summary.csv"
    )
    hard_stress = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/hard_negative_stress/phase12_hard_negative_stress_dataset_summary.csv"
    )
    edge_hard_stress = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/edge_hard_negative_audit/phase12_edge_hard_negative_dataset_summary.csv"
    )
    second_story = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/second_story/phase12_second_story_candidate_audit.csv"
    )
    cross_platform = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/cross_platform_story/phase12_cross_platform_story_audit.csv"
    )
    directionality_edge = read_optional_csv(
        PROJECT_ROOT
        / "outputs/phase12/directionality_edge_permutation/phase12_directionality_edge_permutation_dataset_summary.csv"
    )
    directionality_label = read_optional_csv(
        PROJECT_ROOT / "outputs/phase12/directionality_permutation/phase12_directionality_permutation_dataset_summary.csv"
    )
    coordinate_smoke = read_globbed_csv(
        PROJECT_ROOT / "outputs/phase12/coordinate_model_rerun_smoke",
        "*/*_coordinate_model_rerun_summary.csv",
    )
    rendered = read_optional_csv(PROJECT_ROOT / "outputs/phase12/figures/phase12_rendered_figure_status.csv")

    if expanded_summary.empty:
        goal1_status = "input_layer_ready_smoke"
        goal1_evidence = (
            f"{len(lr_universe)} unique LR pairs collected; "
            f"{len(atlas_ready)} full-transcriptome datasets have >=100 covered LR pairs."
        )
        goal1_gap = "Need run SpatialSelf/COMMOT/CellChat/CellNEST on atlas candidate edges and compute global ranking/FDR/modules."
    else:
        n_datasets = expanded_summary["dataset_key"].nunique()
        n_lr_min = int(expanded_summary["n_lr"].min()) if "n_lr" in expanded_summary else "NA"
        n_lr_max = int(expanded_summary["n_lr"].max()) if "n_lr" in expanded_summary else "NA"
        top10_true = int(expanded_summary["top10_true_lr"].sum()) if "top10_true_lr" in expanded_summary else "NA"
        top10_risk = (
            float(expanded_summary["top10_negative_or_risk_fraction"].max())
            if "top10_negative_or_risk_fraction" in expanded_summary
            else "NA"
        )
        if not expanded_perm.empty and "all_lr_conservative_fdr_0_1" in expanded_perm:
            discoveries = int(expanded_perm["all_lr_conservative_fdr_0_1"].sum())
            goal1_status = "expanded_atlas_ranked_posthoc_fdr_incomplete"
            goal1_gap = (
                "Conservative whole-atlas BH FDR has no q<=0.1 discoveries; still need stronger "
                "end-to-end null calibration and fully tuned official baselines."
            )
        else:
            discoveries = "NA"
            goal1_status = "expanded_atlas_ranked_fdr_missing"
            goal1_gap = "Need post-hoc and end-to-end permutation FDR audits for expanded whole-LR atlas rankings."
        goal1_evidence = (
            f"Expanded calibrated atlas scored {n_datasets} datasets with {n_lr_min}-{n_lr_max} covered LR; "
            f"top10 true LR total={top10_true}, max top10 risk fraction={fmt_short(top10_risk)}, "
            f"whole-atlas q<=0.1 discoveries={discoveries}."
        )

    if not gene_programs.empty or not downstream.empty:
        n_program_rows = len(gene_programs)
        n_downstream_lr = downstream["lr_pair"].nunique() if "lr_pair" in downstream else len(downstream)
        pathway_text = ""
        if not pathway_specific.empty:
            pathway_text = (
                f" pathway-specific audit: clean={int(pathway_specific['n_clean_specific_support'].sum())}, "
                f"mixed-broad={int(pathway_specific['n_mixed_specific_support_broad_dominant'].sum())}, "
                f"negative leakage={int(pathway_specific['n_negative_specific_leakage'].sum())}."
            )
        goal2_status = "whole_transcriptome_program_scan_added"
        goal2_evidence = (
            f"Downstream LR summaries={n_downstream_lr}; gene-program enrichment rows={n_program_rows}; "
            f"GO/Reactome/KEGG/CellChatDB/OmniPath resources are included where available.{pathway_text}"
        )
        goal2_gap = (
            "The expanded pathway-specific receiver-program layer is leakage-confounded across post-Xenium colon, lymph node and Visium HD CRC; use target-rank support instead of claiming clean pathway-specific closure."
        )
    else:
        goal2_status = "resources_available_partial"
        goal2_evidence = "Phase10 CRC/colon gene association scans and Phase11 lymph-node whole-transcriptome target-rank scan exist; pathway gene sets available for CRC/colon."
        goal2_gap = "Need systematic module-level all-gene association and GO/Reactome/KEGG/NicheNet enrichment for top atlas LR/programs."

    if not hard_stress.empty:
        datasets_scored = hard_stress["dataset_key"].nunique() if "dataset_key" in hard_stress else len(hard_stress)
        neg_beats = int(hard_stress["n_negative_beats_positive_comparisons"].sum()) if "n_negative_beats_positive_comparisons" in hard_stress else "NA"
        edge_text = ""
        if not edge_hard_stress.empty and "n_leakage_lr_class_tests" in edge_hard_stress:
            raw_edge = edge_hard_stress[edge_hard_stress.get("score_mode", pd.Series("", index=edge_hard_stress.index)).astype(str).eq("raw")]
            guard_edge = edge_hard_stress[edge_hard_stress.get("score_mode", pd.Series("", index=edge_hard_stress.index)).astype(str).eq("edge_guardrail")]
            if not raw_edge.empty and not guard_edge.empty:
                edge_text = (
                    f" Edge-level hard-negative audit covers {edge_hard_stress['dataset_key'].nunique()} dataset(s); "
                    f"raw leakage class-tests={int(raw_edge['n_leakage_lr_class_tests'].sum())}/"
                    f"{int(raw_edge['n_lr_class_tests'].sum())}, guardrail={int(guard_edge['n_leakage_lr_class_tests'].sum())}/"
                    f"{int(guard_edge['n_lr_class_tests'].sum())}."
                )
            else:
                edge_text = (
                    f" Edge-level hard-negative audit covers {edge_hard_stress['dataset_key'].nunique()} dataset(s), "
                    f"leakage class-tests={int(edge_hard_stress['n_leakage_lr_class_tests'].sum())}/"
                    f"{int(edge_hard_stress['n_lr_class_tests'].sum())}."
                )
        goal3_status = "hard_negative_stress_scored"
        goal3_evidence = (
            f"{len(hard_negatives)} templates generated; hard-negative stress scored across {datasets_scored} datasets; "
            f"negative-beats-positive comparisons={neg_beats}.{edge_text}"
        )
        goal3_gap = (
            "Some receptor-shared/high-expression and edge-level spatial/identity negatives still compete; need improved calibration or stricter decision rule before claiming complete FDR control."
        )
    else:
        goal3_status = "hard_negative_templates_ready"
        goal3_evidence = f"{len(hard_negatives)} hard-negative templates generated across datasets."
        goal3_gap = "Need score these templates through the model/baselines and compute leakage/FDR/target-rank metrics."

    if not second_story.empty or not cross_platform.empty:
        if not second_story.empty and {"best_candidate", "story_class"}.issubset(second_story.columns):
            top_story = second_story.iloc[0]
            story_text = f"best lymph-node candidate={top_story.best_candidate} ({top_story.story_class})"
        else:
            story_text = "lymph-node/tonsil/cross-platform audits available"
        if not cross_platform.empty and {"dataset_key", "story_class"}.issubset(cross_platform.columns):
            supported = cross_platform[
                cross_platform["story_class"].astype(str).str.contains("strong|support", case=False, na=False)
            ]["dataset_key"].astype(str).tolist()
            platform_text = f"supportive cross-platform datasets={','.join(supported[:3]) if supported else 'none clean'}"
        else:
            platform_text = "cross-platform summary missing"
        goal4_status = "second_story_supported_with_caveats"
        goal4_evidence = f"{story_text}; {platform_text}."
        goal4_gap = (
            "Second story is biological-consistency/supportive, not a clean SpatialSelf-exclusive discovery; "
            "negative leakage/downstream caveats remain in tonsil and lymph node."
        )
    else:
        goal4_status = "candidate_backgrounds_identified"
        goal4_evidence = "Lymph node, melanoma, tonsil and Xenium targeted/native backgrounds are present locally; lymph-node CCR7 already has Phase11 support."
        goal4_gap = "Need choose one non-CRC story and satisfy rank/FDR/downstream/source-sink/negative-control criteria."

    if not directionality_edge.empty or not directionality_label.empty:
        recv_support = (
            int(directionality_edge["n_receiver_pairing_fdr_0_1"].sum())
            if "n_receiver_pairing_fdr_0_1" in directionality_edge
            else "NA"
        )
        vector_support = (
            int(directionality_edge["n_edge_vector_fdr_0_1"].sum())
            if "n_edge_vector_fdr_0_1" in directionality_edge
            else "NA"
        )
        coord_text = ""
        if not coordinate_smoke.empty and "n_completed_permutations" in coordinate_smoke:
            coord_text = (
                f" coordinate model-rerun smoke completed for {coordinate_smoke['dataset_key'].nunique()} dataset(s), "
                f"permutations={int(coordinate_smoke['n_completed_permutations'].sum())}."
            )
        goal5_status = "quantitative_directionality_audits_added"
        goal5_evidence = (
            f"Aggregated label permutation and fixed-score edge permutation audits exist; "
            f"receiver-pairing q<=0.1 counts={recv_support}, vector q<=0.1 counts={vector_support}.{coord_text}"
        )
        goal5_gap = (
            "Vector-null support is weak/absent for key SpatialSelf signals; coordinate model-rerun is only a low-permutation smoke and still needs scaled formal reruns."
        )
    else:
        goal5_status = "phase11_metrics_reusable"
        goal5_evidence = "Phase11 source-sink/niche-flow outputs exist for CRC and lymph node."
        goal5_gap = "Need standardize source-sink alignment, vector coherence, boundary flow and permutation p-values across atlas LR pairs."

    if not rendered.empty and {"all_expected_outputs_exist", "svg_keeps_editable_text"}.issubset(rendered.columns):
        n_rendered = int(rendered["all_expected_outputs_exist"].sum())
        n_editable = int(rendered["svg_keeps_editable_text"].sum())
        goal6_status = "draft_manuscript_figures_rendered"
        goal6_evidence = f"Rendered figure drafts={n_rendered}/{len(rendered)}; editable SVG text={n_editable}/{len(rendered)}; source-data package has four planned main figures."
        goal6_gap = "Figures are manuscript drafts; final art and captions should wait until whole-atlas FDR/baseline/directionality gaps are resolved."
    else:
        goal6_status = "not_started"
        goal6_evidence = "Phase10/11 figures exist, but Phase12 whole-atlas/hard-negative figures are not generated."
        goal6_gap = "Need generate four manuscript-grade Phase12 figures after atlas scoring and downstream modules are complete."

    rows = [
        {
            "goal": "Goal 1 Whole-LR Atlas Validation",
            "status": goal1_status,
            "evidence": goal1_evidence,
            "remaining_gap": goal1_gap,
        },
        {
            "goal": "Goal 2 Whole-Transcriptome Communication-Dependent Genes",
            "status": goal2_status,
            "evidence": goal2_evidence,
            "remaining_gap": goal2_gap,
        },
        {
            "goal": "Goal 3 Hard Negative Stress Test",
            "status": goal3_status,
            "evidence": goal3_evidence,
            "remaining_gap": goal3_gap,
        },
        {
            "goal": "Goal 4 Second Strong Biological Story",
            "status": goal4_status,
            "evidence": goal4_evidence,
            "remaining_gap": goal4_gap,
        },
        {
            "goal": "Goal 5 Quantitative Directionality Benchmark",
            "status": goal5_status,
            "evidence": goal5_evidence,
            "remaining_gap": goal5_gap,
        },
        {
            "goal": "Goal 6 Manuscript-Grade Figure and Story Consolidation",
            "status": goal6_status,
            "evidence": goal6_evidence,
            "remaining_gap": goal6_gap,
        },
    ]
    return pd.DataFrame(rows)


def write_status(
    datasets: list[dict[str, object]],
    lr_universe: pd.DataFrame,
    coverage: pd.DataFrame,
    hard_negatives: pd.DataFrame,
    goal_audit: pd.DataFrame,
) -> None:
    top_cov = coverage.sort_values("n_lr_both_genes_present", ascending=False)
    lines = [
        "# Phase 12 Atlas Readiness Status",
        "",
        "## What was initialized",
        "",
        "- Built a merged LR universe from local curated panels, Phase 10/11 LR tables, Phase 6/7 datasets and melanoma LR resources.",
        "- Audited LR ligand/receptor gene coverage across full-transcriptome and targeted backgrounds.",
        "- Generated receptor-shared and housekeeping-ligand hard-negative templates.",
        "- Created a six-goal Phase 12 audit table tied to the COMMOT-style narrative.",
        "",
        "## Key numbers",
        "",
        f"- Unique LR pairs in merged universe: {len(lr_universe)}.",
        f"- Datasets audited: {len(datasets)}.",
        f"- Hard-negative templates generated: {len(hard_negatives)}.",
        "",
        "## Dataset LR coverage",
        "",
        "| dataset | kind | genes | covered LR | priority anchors | hard negatives |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_cov.itertuples(index=False):
        lines.append(
            f"| {row.dataset_key} | {row.kind} | {row.n_genes} | {row.n_lr_both_genes_present} | "
            f"{row.n_priority_anchor_present} | {row.n_prior_high_expression_negative_present} |"
        )
    lines.extend(
        [
            "",
            "## Goal audit",
            "",
            "| goal | status | remaining gap |",
            "| --- | --- | --- |",
        ]
    )
    for row in goal_audit.itertuples(index=False):
        lines.append(f"| {row.goal} | {row.status} | {row.remaining_gap} |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is an input/readiness layer, not a completed Phase 12 atlas result. It proves that enough local LR and dataset resources exist to start whole-atlas and hard-negative scoring, but does not yet prove global FDR, atlas modules or manuscript-grade biological conclusions.",
            "",
            "## Outputs",
            "",
            "- `phase12_lr_universe.csv`",
            "- `phase12_dataset_lr_coverage.csv`",
            "- `phase12_hard_negative_templates.csv`",
            "- `phase12_goal_audit.csv`",
            "- `phase12_atlas_readiness_manifest.json`",
        ]
    )
    (OUT_DIR / "phase12_status_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    datasets: list[dict[str, object]],
    lr_universe: pd.DataFrame,
    coverage: pd.DataFrame,
    hard_negatives: pd.DataFrame,
) -> None:
    manifest = {
        "phase": "phase12",
        "task": "atlas_readiness_and_hard_negative_template_audit",
        "n_lr_pairs": int(len(lr_universe)),
        "n_datasets": int(len(datasets)),
        "n_hard_negative_templates": int(len(hard_negatives)),
        "outputs": [
            "phase12_lr_universe.csv",
            "phase12_dataset_lr_coverage.csv",
            "phase12_hard_negative_templates.csv",
            "phase12_goal_audit.csv",
            "phase12_status_summary.md",
        ],
        "claim_boundary": "Readiness audit only. Model scoring, global FDR and manuscript-grade figures remain required for Phase 12 completion.",
    }
    (OUT_DIR / "phase12_atlas_readiness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_genes_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    table = pd.read_csv(path)
    if "gene" in table.columns:
        values = table["gene"]
    elif "genes" in table.columns:
        values = table["genes"]
    else:
        values = table.iloc[:, 0]
    return {normalize_gene(x) for x in values if valid_gene(normalize_gene(x))}


def read_expression_header(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    return {normalize_gene(x) for x in header if normalize_gene(x) not in {"CELL_ID", "BARCODE"} and valid_gene(normalize_gene(x))}


def read_10x_h5_genes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as handle:
        names = handle["matrix"]["features"]["name"][()]
    return {normalize_gene(x.decode() if isinstance(x, bytes) else x) for x in names if valid_gene(normalize_gene(x.decode() if isinstance(x, bytes) else x))}


def classify_source(path: Path) -> str:
    text = str(path)
    if "phase10" in text and "nichenet" in text:
        return "phase10_nichenet_targets"
    if "phase11" in text:
        return "phase11_curated_validation"
    if "phase7" in text:
        return "phase7_cross_platform_panel"
    if "phase6" in text:
        return "phase6_expanded_panel"
    if "melanoma_lr_benchmark" in text:
        return "melanoma_lr_benchmark_table"
    if "phase4_curated" in text:
        return "phase4_curated_targets"
    return "other"


def classify_biology(ligand: str, receptor: str) -> str:
    pair = f"{ligand}-{receptor}"
    if ligand.startswith(("CCL", "CXCL")) or receptor.startswith(("CCR", "CXCR")):
        return "chemokine_recruitment"
    if pair in {"CD274-PDCD1", "PDCD1-CD274", "CD48-CD244", "CD40LG-CD40"} or ligand in {"CD274", "PDCD1", "CD48", "CD40LG"}:
        return "immune_contact_checkpoint"
    if ligand.startswith(("COL", "LAMA", "LAMB")) or receptor.startswith("ITG") or ligand in {"FN1", "SPP1"}:
        return "ecm_myeloid_remodeling"
    if ligand.startswith("VEGF") or receptor in {"KDR", "FLT1", "FLT4", "TEK"}:
        return "angiogenesis"
    if ligand in {"TGFB1", "TGFB2", "TGFB3"} or receptor.startswith("TGFBR"):
        return "tgfb_stromal"
    if ligand.startswith(("DLL", "JAG")) or receptor.startswith("NOTCH"):
        return "notch_contact"
    if ligand in HOUSEKEEPING_LIGANDS:
        return "high_expression_negative_risk"
    return "other"


def normalize_gene(value: object) -> str:
    return str(value).strip().upper()


def valid_gene(value: str) -> bool:
    return bool(value and GENE_RE.match(value) and "_" not in value and ":" not in value and "COMPLEX" not in value)


def valid_gene_pair(ligand: str, receptor: str) -> bool:
    return valid_gene(ligand) and valid_gene(receptor) and ligand != receptor


def safe_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(x) for x in values if str(x) and str(x) != "nan"}))


def join_nonempty_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(x) for x in values if str(x).strip() and str(x) != "nan"}))


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
