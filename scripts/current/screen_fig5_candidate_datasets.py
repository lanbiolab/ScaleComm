#!/usr/bin/env python
"""Screen Fig5 candidate real datasets without requiring heavy scientific deps."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/current_fig5_new_dataset_screen"
MANIFEST = OUT_DIR / "fig5_candidate_dataset_manifest.csv"

EXPECTED_FILES = {
    "cosmx_nsclc_lung9_rep1": [
        "data/optional/cosmx_nsclc_lung9_rep1/raw/Lung9_Rep1_SMI_Flat_data.tar.gz",
    ],
    "xenium_breast_biomarkers_s1_top": [
        "data/optional/xenium_breast_biomarkers/raw/Human_Breast_Biomarkers_S1_Top_cell_groups.csv",
        "data/optional/xenium_breast_biomarkers/raw/Human_Breast_Biomarkers_S1_Top_gene_panel.json",
        "data/optional/xenium_breast_biomarkers/raw/Human_Breast_Biomarkers_S1_Top_analysis_summary.html",
    ],
    "xenium_prime_breast_cancer_ffpe": [
        "data/optional/xenium_prime_breast_cancer_ffpe/raw/Xenium_Prime_Breast_Cancer_FFPE_gene_panel.json",
    ],
    "spamosaic_tonsil_a1_d1": [
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/cells.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/expression.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/lr_pairs.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/spamosaic_tonsil_A1_rich_targets_subset/candidate_interactions.csv",
    ],
    "xenium_human_lymph_node": [
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/cells.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/expression.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/lr_pairs.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/xenium_human_lymph_node_rich_targets_subset/candidate_interactions.csv",
    ],
    "lung_cancer_visium_subset": [
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/lung_cancer_visium_subset/cells.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/lung_cancer_visium_subset/expression.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/lung_cancer_visium_subset/lr_pairs.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/lung_cancer_visium_subset/candidate_interactions.csv",
    ],
    "native_human_melanoma_multiome_subset": [
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/cells.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/expression.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/lr_pairs.csv",
        "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data/standardized_inputs/native_human_melanoma_multiome_subset/candidate_interactions.csv",
    ],
}

LR_QUERY_AXES = {
    "TGFB1-TGFBR2": ("TGFB1", "TGFBR2"),
    "SPP1-CD44": ("SPP1", "CD44"),
    "CXCL12-CXCR4": ("CXCL12", "CXCR4"),
    "CXCL9-CXCR3": ("CXCL9", "CXCR3"),
    "CXCL10-CXCR3": ("CXCL10", "CXCR3"),
    "CCL19-CCR7": ("CCL19", "CCR7"),
    "PDCD1-CD274": ("PDCD1", "CD274"),
    "CD86-CD28": ("CD86", "CD28"),
    "VEGFA-KDR": ("VEGFA", "KDR"),
    "PDGFD-PDGFRB": ("PDGFD", "PDGFRB"),
    "COL4A1-ITGB1": ("COL4A1", "ITGB1"),
}

GENE_PANEL_FILES = {
    "xenium_breast_biomarkers_s1_top": "data/optional/xenium_breast_biomarkers/raw/Human_Breast_Biomarkers_S1_Top_gene_panel.json",
    "xenium_prime_breast_cancer_ffpe": "data/optional/xenium_prime_breast_cancer_ffpe/raw/Xenium_Prime_Breast_Cancer_FFPE_gene_panel.json",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in _read_manifest(MANIFEST):
        cid = row["candidate_id"]
        expected = EXPECTED_FILES.get(cid, [])
        present = []
        missing = []
        total_size = 0
        for rel in expected:
            path = PROJECT_ROOT / rel
            if path.exists():
                present.append(rel)
                total_size += path.stat().st_size
            else:
                missing.append(rel)
        coverage = _lr_coverage(cid)
        status = "ready_or_existing" if expected and not missing else row.get("status", "")
        if cid == "cosmx_nsclc_lung9_rep1" and present:
            status = "partial_download" if total_size < 1_000_000_000 else "downloaded"
        rows.append(
            {
                "candidate_id": cid,
                "priority": row.get("priority", ""),
                "platform": row.get("platform", ""),
                "tissue": row.get("tissue", ""),
                "screen_status": status,
                "n_expected_files": len(expected),
                "n_present_files": len(present),
                "local_size_mb": round(total_size / 1_000_000, 3),
                "missing_files": ";".join(missing),
                "lr_axes_present": ";".join(coverage["present"]),
                "lr_axes_missing": ";".join(coverage["missing"]),
                "current_decision": row.get("current_decision", ""),
                "main_risk": row.get("main_risk", ""),
            }
        )
    _write_csv(OUT_DIR / "fig5_candidate_screen_status.csv", rows)
    (OUT_DIR / "fig5_candidate_screen_status.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'fig5_candidate_screen_status.csv'}")


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lr_coverage(candidate_id: str) -> dict[str, list[str]]:
    rel = GENE_PANEL_FILES.get(candidate_id)
    if rel is None:
        return {"present": [], "missing": []}
    path = PROJECT_ROOT / rel
    if not path.exists():
        return {"present": [], "missing": list(LR_QUERY_AXES)}
    genes = set(_read_gene_panel(path))
    present = []
    missing = []
    for axis, pair in LR_QUERY_AXES.items():
        if pair[0] in genes and pair[1] in genes:
            present.append(axis)
        else:
            missing.append(axis)
    return {"present": present, "missing": missing}


def _read_gene_panel(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    genes = []
    for target in payload.get("payload", {}).get("targets", []):
        typ = target.get("type", {})
        if typ.get("descriptor") == "gene":
            name = typ.get("data", {}).get("name")
            if name:
                genes.append(name)
    return sorted(set(genes))


if __name__ == "__main__":
    main()

