#!/usr/bin/env python
"""Generate source-table-backed figures for revised Phase 2.

The current outputs are source-ready smoke figures. They are intentionally
conservative: labels and manifests preserve the distinction between evidence
assembly and final manuscript-scale validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "legend.frameon": False,
    }
)

PALETTE = {
    "spatialself_wavelet_edge_prior": "#2A9D8F",
    "spatialself_wavelet": "#457B9D",
    "commot": "#E76F51",
    "cellchat": "#8D99AE",
    "spatalk": "#B56576",
    "cellnest": "#E9C46A",
    "deeptalk": "#6D6875",
    "juxtacrine_contact": "#6A8D92",
    "short_medium_paracrine": "#E9C46A",
    "long_range_chemokine_niche": "#8AB17D",
    "false_positive_high_expression": "#B56576",
    "neutral": "#6D6875",
    "light": "#E8E8E8",
    "dark": "#222222",
}

METHOD_LABELS = {
    "spatialself_wavelet_edge_prior": "SpatialSelf\nwavelet+prior",
    "spatialself_wavelet": "SpatialSelf\nwavelet",
    "commot": "COMMOT",
    "cellchat": "CellChat",
    "spatalk": "SpaTalk",
    "cellnest": "CellNEST",
    "deeptalk": "DeepTalk",
}

METHOD_ORDER = [
    "spatialself_wavelet_edge_prior",
    "spatialself_wavelet",
    "commot",
    "cellchat",
    "spatalk",
    "cellnest",
    "deeptalk",
]

DATASET_LABELS = {
    "lymph_node_real_background_smoke": "Lymph node",
    "tonsil_real_background_smoke": "Tonsil",
    "melanoma_real_background_smoke": "Melanoma",
}

MECHANISM_LABELS = {
    "juxtacrine_contact": "Contact",
    "short_medium_paracrine": "Short/medium",
    "long_range_chemokine_niche": "Long-range niche",
    "false_positive_high_expression": "High-expression\nnegative",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly-dir",
        default="outputs/phase2_revised/goal6_manuscript_assembly",
        help="Directory produced by run_goal6_manuscript_assembly.py.",
    )
    parser.add_argument(
        "--output-subdir",
        default="figures",
        help="Subdirectory under assembly-dir for figure exports.",
    )
    args = parser.parse_args()

    assembly_dir = PROJECT_ROOT / args.assembly_dir
    source_dir = assembly_dir / "source_tables"
    figure_dir = assembly_dir / args.output_subdir
    figure_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(assembly_dir, source_dir)
    contract = _figure_contract()
    exports: list[Path] = []
    exports.extend(_plot_fig1_evidence_grid(tables, figure_dir))
    exports.extend(_plot_individual_panels(tables, figure_dir))

    contract_path = figure_dir / "figure_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    qa = _write_qa(figure_dir, exports, assembly_dir, source_dir, tables, contract_path)
    manifest = {
        "analysis_label": "phase2_revised_goal6_figures",
        "evidence_status": "source_ready_smoke_figures_not_final_manuscript",
        "figure_contract": str(contract_path.relative_to(PROJECT_ROOT)),
        "figure_qa_summary": str((figure_dir / "figure_qa_summary.csv").relative_to(PROJECT_ROOT)),
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in exports],
        "source_tables": {
            name: str(_source_path(assembly_dir, source_dir, name).relative_to(PROJECT_ROOT))
            for name in SOURCE_FILES
            if _source_path(assembly_dir, source_dir, name).exists()
        },
        "limitations": [
            "Figures are generated from smoke/source-ready revised Phase 2 tables.",
        "FDR panel uses the stronger 200-permutation smoke output when available; final 500-permutation panels remain pending.",
            "Downstream validation is currently a receiver-response proxy, not curated LR-target validation.",
            "Baseline panels include adapter-light/status-aware smoke comparisons, not final tuned baseline claims.",
            "Manual visual review is still required before manuscript submission.",
        ],
    }
    manifest_path = figure_dir / "phase2_revised_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Figure manifest: {manifest_path}")
    print(f"Figure QA:       {figure_dir / 'figure_qa_summary.csv'}")
    print(qa.to_string(index=False))


SOURCE_FILES = {
    "claim_matrix": "claim_evidence_matrix.csv",
    "limitations": "manuscript_limitations.csv",
    "figure_plan": "figure_panel_plan.csv",
    "goal1_method_summary": "goal1_method_summary.csv",
    "goal1_lr_control_summary": "goal1_lr_control_summary.csv",
    "goal2_metrics": "goal2_metrics.csv",
    "goal2_claims": "goal2_claims.csv",
    "goal2_formal_metrics": "goal2_formal_metrics.csv",
    "goal2_formal_summary": "goal2_formal_summary.csv",
    "goal2_formal_claims": "goal2_formal_claims.csv",
    "goal2_formal_status": "goal2_formal_status.csv",
    "goal3_mechanism_summary": "goal3_mechanism_summary.csv",
    "goal3_formal_benchmark_summary": "goal3_formal_benchmark_summary.csv",
    "goal3_formal_claims": "goal3_formal_claims.csv",
    "goal4_lymph_validation": "goal4_lymph_validation.csv",
    "goal4_tonsil_validation": "goal4_tonsil_validation.csv",
    "goal4_melanoma_validation": "goal4_melanoma_validation.csv",
    "goal4_formal_real_background_summary": "goal4_formal_real_background_summary.csv",
    "goal4_formal_claims": "goal4_formal_claims.csv",
    "goal5_fdr_summary": "goal5_fdr_summary.csv",
    "goal5_fdr_group_summary": "goal5_fdr_group_summary.csv",
    "goal5_fdr_200perm_summary": "goal5_fdr_200perm_summary.csv",
    "goal5_fdr_200perm_group_summary": "goal5_fdr_200perm_group_summary.csv",
    "goal6_runtime_method_profile": "goal6_runtime_method_profile.csv",
    "goal6_runtime_highres_scaling_runs": "goal6_runtime_highres_scaling_runs.csv",
    "goal6_runtime_claims": "goal6_runtime_claims.csv",
}


def _load_tables(assembly_dir: Path, source_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for name, filename in SOURCE_FILES.items():
        path = _source_path(assembly_dir, source_dir, name)
        if not path.exists():
            missing.append(filename)
            continue
        tables[name] = pd.read_csv(path)
    required = [
        "claim_matrix",
        "goal1_method_summary",
        "goal2_metrics",
        "goal3_mechanism_summary",
        "goal4_lymph_validation",
        "goal4_tonsil_validation",
        "goal4_melanoma_validation",
        "goal5_fdr_summary",
    ]
    missing_required = [name for name in required if name not in tables]
    if missing_required:
        raise FileNotFoundError(f"Missing required source tables: {missing_required}; missing files={missing}")
    return tables


def _source_path(assembly_dir: Path, source_dir: Path, name: str) -> Path:
    root_level_tables = {"claim_matrix", "limitations", "figure_plan"}
    base_dir = assembly_dir if name in root_level_tables else source_dir
    return base_dir / SOURCE_FILES[name]


def _figure_contract() -> dict[str, object]:
    return {
        "core_conclusion": (
            "Revised Phase 2 has a traceable evidence chain linking mechanism-layered "
            "simulation, real-background injection, weak-signal/false-positive tradeoff, "
            "baseline-matched controls, adapter-aware baseline comparison, and formal FDR schema outputs."
        ),
        "evidence_chain": [
            "Fig1A: six claim nodes and source-ready status define the manuscript evidence chain.",
            "Fig1B: simulator v2 includes three communication mechanisms plus high-expression negatives.",
            "Fig1C: real-background smoke datasets preserve realistic sparsity, density, and injected truth.",
            "Fig2A: weak-signal recall is interpreted under high-expression FP and FDR guardrails.",
            "Fig2B: positive/negative control ranking compares methods on identical injected controls.",
            "Fig2C: permutation null outputs expose current FDR schema and smoke-scale limitations.",
        ],
        "archetype": "quantitative grid",
        "backend": "python_matplotlib",
        "journal_export_contract": {
            "formats": ["svg", "pdf", "tiff", "png"],
            "editable_text": True,
            "source_data_traceability": "all panels read from Goal 6 copied source tables",
            "statistics_scope": "descriptive smoke/source-table summaries only",
            "image_integrity_note": "no microscopy or primary image manipulation",
        },
        "review_risks": [
            "smoke scale is not manuscript-final evidence",
            "CellChat/SpaTalk/CellNEST are adapter-light smoke outputs",
            "FDR uses 20 permutations and is schema evidence rather than powered discovery",
        ],
    }


def _plot_fig1_evidence_grid(tables: dict[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    _panel_evidence_chain(ax_a, tables["claim_matrix"])
    _panel_simulator_mechanisms(ax_b, tables["goal3_mechanism_summary"])
    _panel_real_background(ax_c, _real_background_table(tables))
    _panel_tradeoff(ax_d, tables["goal2_metrics"])
    _panel_control_ranking(ax_e, tables["goal1_method_summary"])

    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    _panel_label(ax_c, "C")
    _panel_label(ax_d, "D")
    _panel_label(ax_e, "E")
    fig.suptitle("Revised Phase 2 source-ready evidence chain", x=0.01, y=1.01, ha="left", fontsize=9)
    return _save_all(fig, figure_dir / "phase2_revised_evidence_grid")


def _plot_individual_panels(tables: dict[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    exports: list[Path] = []
    panel_specs = [
        ("fig1a_evidence_chain", (7.2, 1.8), _panel_evidence_chain, tables["claim_matrix"]),
        ("fig1b_simulator_mechanism_coverage", (3.3, 2.4), _panel_simulator_mechanisms, tables["goal3_mechanism_summary"]),
        ("fig1c_real_background_properties", (3.3, 2.4), _panel_real_background, _real_background_table(tables)),
        ("fig2a_weak_signal_fp_tradeoff", (3.3, 2.4), _panel_tradeoff, tables["goal2_metrics"]),
        ("fig2b_control_ranking", (3.3, 2.4), _panel_control_ranking, tables["goal1_method_summary"]),
        ("fig2c_fdr_null_outputs", (3.3, 2.4), _panel_fdr_nulls, tables.get("goal5_fdr_200perm_summary", tables["goal5_fdr_summary"])),
    ]
    for name, size, func, data in panel_specs:
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        func(ax, data)
        exports.extend(_save_all(fig, figure_dir / name))
    return exports


def _panel_evidence_chain(ax: plt.Axes, claim_matrix: pd.DataFrame) -> None:
    ax.set_axis_off()
    claims = claim_matrix.copy()
    labels = [
        ("Simulator v2", "mechanism truth"),
        ("Real background", "semi-synthetic"),
        ("Tradeoff", "weak signal / FP"),
        ("Controls", "baseline-matched"),
        ("FDR schema", "permutation nulls"),
        ("Audit", "source tables"),
    ]
    x_positions = np.linspace(0.07, 0.93, len(labels))
    for idx, ((title, subtitle), x) in enumerate(zip(labels, x_positions)):
        status = "source_ready"
        if "evidence_status" in claims and idx < len(claims):
            status = str(claims.iloc[idx]["evidence_status"])
        color = "#2A9D8F" if status == "source_ready" else "#6D6875"
        box = mpl.patches.FancyBboxPatch(
            (x - 0.07, 0.34),
            0.14,
            0.36,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="#F7F7F7",
            edgecolor=color,
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x, 0.57, title, ha="center", va="center", fontsize=7.3, color="#222222", weight="bold")
        ax.text(x, 0.44, subtitle, ha="center", va="center", fontsize=6.4, color="#555555")
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.085, 0.52),
                xytext=(x + 0.085, 0.52),
                arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "#999999"},
            )
    ax.text(
        0.01,
        0.14,
        "Claim scope: calibrated LR-aware CCC framework; current plots are smoke/source-ready, not final manuscript-scale validation.",
        ha="left",
        va="center",
        fontsize=6.8,
        color="#555555",
    )


def _panel_simulator_mechanisms(ax: plt.Axes, mechanism_summary: pd.DataFrame) -> None:
    df = mechanism_summary.copy()
    grouped = (
        df.groupby("mechanism", as_index=False)
        .agg(candidates=("candidates", "sum"), positives=("positives", "sum"), weak_positives=("weak_positives", "sum"))
        .sort_values("mechanism")
    )
    grouped["strong_positives"] = grouped["positives"] - grouped["weak_positives"]
    grouped["label"] = grouped["mechanism"].map(MECHANISM_LABELS).fillna(grouped["mechanism"])
    x = np.arange(len(grouped))
    strong = grouped["strong_positives"].to_numpy(float)
    weak = grouped["weak_positives"].to_numpy(float)
    colors = [PALETTE.get(mech, PALETTE["neutral"]) for mech in grouped["mechanism"]]
    ax.bar(x, strong, color=colors, alpha=0.78, label="strong true edges")
    ax.bar(x, weak, bottom=strong, color=colors, alpha=0.35, hatch="///", label="weak true edges")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["label"], rotation=25, ha="right")
    ax.set_ylabel("True edges")
    ax.set_title("Simulator v2 mechanism coverage", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    _light_grid(ax, axis="y")


def _panel_real_background(ax: plt.Axes, validation: pd.DataFrame) -> None:
    df = validation.copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    metrics = [
        ("mean_observed_sparsity", "Sparsity"),
        ("positive_interactions", "True edges"),
        ("weak_positive_interactions", "Weak edges"),
        ("truth_spatial_niches", "Truth niches"),
    ]
    normalized = []
    for column, label in metrics:
        values = df[column].astype(float)
        denom = values.max() if values.max() > 0 else 1.0
        for dataset, value in zip(df["dataset_label"], values):
            normalized.append({"dataset": dataset, "metric": label, "value": value / denom})
    plot_df = pd.DataFrame(normalized)
    datasets = list(df["dataset_label"])
    x = np.arange(len(metrics))
    width = 0.23
    dataset_colors = ["#457B9D", "#2A9D8F", "#B56576"]
    for i, dataset in enumerate(datasets):
        vals = [plot_df[(plot_df["dataset"] == dataset) & (plot_df["metric"] == label)]["value"].iloc[0] for _, label in metrics]
        ax.bar(x + (i - 1) * width, vals, width=width, color=dataset_colors[i % len(dataset_colors)], label=dataset)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Scaled to max")
    ax.set_title("Real-background semi-synthetic properties", loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    _light_grid(ax, axis="y")


def _panel_tradeoff(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    df = metrics.copy()
    df = df[df["method"].astype(str).isin(METHOD_ORDER)].copy()
    for method in [method for method in METHOD_ORDER if method in set(df["method"].astype(str))]:
        group = df[df["method"].astype(str) == method]
        color = PALETTE.get(method, PALETTE["neutral"])
        ax.scatter(
            group["false_positive_high_expression_rate_at_threshold"],
            group["weak_signal_recall_at_threshold"],
            s=38,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=METHOD_LABELS.get(method, method),
            zorder=3,
        )
        for _, row in group.iterrows():
            label = DATASET_LABELS.get(str(row["dataset"]), str(row["dataset"]).split("_")[0])
            ax.text(
                float(row["false_positive_high_expression_rate_at_threshold"]) + 0.0015,
                float(row["weak_signal_recall_at_threshold"]),
                label,
                fontsize=5.8,
                color="#444444",
                va="center",
            )
    ax.axvline(0.05, color="#222222", lw=0.7, ls="--")
    ax.text(0.052, 0.04, "FP guardrail", fontsize=5.8, color="#555555", rotation=90, va="bottom")
    ax.set_xlabel("High-expression FP rate")
    ax.set_ylabel("Weak-signal recall")
    ax.set_title("Weak signal versus high-expression false positives", loc="left", fontsize=8)
    ax.set_xlim(left=-0.002)
    ax.set_ylim(bottom=-0.02)
    ax.legend(fontsize=5.6, loc="center right")
    _light_grid(ax)


def _panel_control_ranking(ax: plt.Axes, method_summary: pd.DataFrame) -> None:
    df = method_summary.copy()
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    methods = [m for m in METHOD_ORDER if m in set(df["method"].astype(str))]
    x = np.arange(len(df["dataset_label"].unique()))
    width = min(0.11, 0.72 / max(len(methods), 1))
    for i, method in enumerate(methods):
        group = df[df["method"] == method].set_index("dataset_label")
        datasets = list(df["dataset_label"].drop_duplicates())
        vals = [float(group.loc[d, "negative_top3_count"]) if d in group.index else np.nan for d in datasets]
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            vals,
            width=width,
            color=PALETTE.get(method, PALETTE["neutral"]),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(df["dataset_label"].drop_duplicates()), rotation=20, ha="right")
    ax.set_ylabel("High-expression negatives in top 3")
    ax.set_title("Baseline-matched negative-control ranking", loc="left", fontsize=8)
    ax.set_ylim(0, max(3, float(df["negative_top3_count"].max()) + 0.6))
    ax.legend(fontsize=5.3, loc="upper left", ncol=2)
    _light_grid(ax, axis="y")


def _panel_fdr_nulls(ax: plt.Axes, fdr_summary: pd.DataFrame) -> None:
    df = fdr_summary.copy()
    df["null_label"] = df["null_model"].map(
        {
            "spatial_permutation": "Spatial",
            "receiver_permutation": "Receiver",
            "lr_identity_permutation": "LR identity",
        }
    ).fillna(df["null_model"])
    x = np.arange(len(df))
    ax.bar(x, df["min_bh_fdr"].astype(float), color="#457B9D", width=0.55, label="min BH FDR")
    ax.scatter(x, df["min_p_value"].astype(float), color="#E76F51", s=24, zorder=3, label="min p")
    ax.axhline(0.05, color="#222222", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["null_label"], rotation=25, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Formal FDR schema smoke", loc="left", fontsize=8)
    n_perm = int(df["n_permutations"].dropna().iloc[0]) if "n_permutations" in df and df["n_permutations"].notna().any() else 0
    ax.text(0.02, 0.94, f"{n_perm} permutations", transform=ax.transAxes, fontsize=6, color="#555555", va="top")
    ax.legend(fontsize=6, loc="upper right")
    _light_grid(ax, axis="y")


def _real_background_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for dataset, key in [
        ("lymph_node_real_background_smoke", "goal4_lymph_validation"),
        ("tonsil_real_background_smoke", "goal4_tonsil_validation"),
        ("melanoma_real_background_smoke", "goal4_melanoma_validation"),
    ]:
        frame = tables[key].copy()
        frame["dataset"] = dataset
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _light_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color="#E8E8E8", linewidth=0.5)
    ax.set_axisbelow(True)


def _save_all(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [
        stem.with_suffix(".svg"),
        stem.with_suffix(".pdf"),
        stem.with_suffix(".tiff"),
        stem.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _write_qa(
    figure_dir: Path,
    exports: list[Path],
    assembly_dir: Path,
    source_dir: Path,
    tables: dict[str, pd.DataFrame],
    contract_path: Path,
) -> pd.DataFrame:
    rows = []
    for path in exports:
        suffix = path.suffix.lower().lstrip(".")
        size = path.stat().st_size if path.exists() else 0
        rows.append(
            {
                "artifact_type": "figure",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "format": suffix,
                "exists": path.exists(),
                "size_bytes": size,
                "non_empty": size > 1_000,
                "source_traceable": True,
                "manual_visual_review_required": True,
            }
    )
    for name, frame in tables.items():
        path = _source_path(assembly_dir, source_dir, name)
        rows.append(
            {
                "artifact_type": "source_table",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "format": "csv",
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "non_empty": len(frame) > 0,
                "source_traceable": True,
                "manual_visual_review_required": False,
            }
        )
    rows.append(
        {
            "artifact_type": "figure_contract",
            "path": str(contract_path.relative_to(PROJECT_ROOT)),
            "format": "json",
            "exists": contract_path.exists(),
            "size_bytes": contract_path.stat().st_size if contract_path.exists() else 0,
            "non_empty": contract_path.exists() and contract_path.stat().st_size > 500,
            "source_traceable": True,
            "manual_visual_review_required": False,
        }
    )
    qa = pd.DataFrame(rows)
    qa["qa_pass"] = qa["exists"].astype(bool) & qa["non_empty"].astype(bool) & qa["source_traceable"].astype(bool)
    qa.to_csv(figure_dir / "figure_qa_detail.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "analysis_label": "phase2_revised_goal6_figures",
                "evidence_status": "source_ready_smoke_figures_not_final_manuscript",
                "n_figure_files": int((qa["artifact_type"] == "figure").sum()),
                "n_source_tables": int((qa["artifact_type"] == "source_table").sum()),
                "n_failed_qa_rows": int((~qa["qa_pass"]).sum()),
                "formats": ",".join(sorted(set(qa.loc[qa["artifact_type"] == "figure", "format"]))),
                "manual_visual_review_required": True,
                "claim_scope": "smoke/source-ready figure bundle; not manuscript-final evidence",
            }
        ]
    )
    summary.to_csv(figure_dir / "figure_qa_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    main()
