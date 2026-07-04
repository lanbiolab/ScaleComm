#!/usr/bin/env python
"""Plot cross-dataset communication program summaries for Phase 11."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs/phase11/commot_style_spatial_story/communication_program_summary"
FIG_DIR = OUT_DIR / "figures"
SOURCE_DIR = FIG_DIR / "source_data"

TEXT = "#202020"
SUBTLE = "#666666"
GRID = "#D9D9D9"
PROGRAM_COLORS = {
    "myeloid-remodeling": "#6B8E6E",
    "chemokine recruitment": "#0F4D92",
    "angiogenesis": "#7A9E9F",
    "immune checkpoint": "#9A6B9E",
    "ECM/stromal remodeling": "#8A7A5C",
    "contact-dependent immune signaling": "#4E8C86",
    "inflammatory/paracrine remodeling": "#B88A44",
    "high-expression negative": "#B64342",
    "background LR-like decoy": "#B8B8B8",
}
DISPLAY_FAMILIES = [
    "myeloid-remodeling",
    "chemokine recruitment",
    "angiogenesis",
    "immune checkpoint",
    "ECM/stromal remodeling",
    "contact-dependent immune signaling",
    "inflammatory/paracrine remodeling",
    "high-expression negative",
]


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    programs = pd.read_csv(OUT_DIR / "phase11_spatialself_communication_program_summary.csv")
    matrix = pd.read_csv(OUT_DIR / "phase11_spatialself_program_family_matrix.csv")
    source = build_source(programs, matrix)
    for name, table in source.items():
        table.to_csv(SOURCE_DIR / f"phase11_communication_program_{name}.csv", index=False)

    fig = plt.figure(figsize=(7.35, 5.65), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.12, 0.95, 1.15],
        height_ratios=[1.0, 1.05],
        left=0.075,
        right=0.985,
        top=0.86,
        bottom=0.16,
        wspace=0.46,
        hspace=0.56,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 1:])

    plot_program_heatmap(ax_a, source["family_matrix"])
    plot_module_evidence(ax_b, source["programs"])
    plot_negative_leakage(ax_c, source["programs"])
    plot_interpretation_table(ax_d, source["table"])

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)

    fig.suptitle(
        "SpatialSelf modules organize LR pairs into communication programs",
        x=0.075,
        y=0.97,
        ha="left",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.075,
        0.91,
        "Cross-dataset LR activity clustering separates chemokine, inflammatory/paracrine, myeloid-remodeling and negative-control-heavy programs, with downstream or flow support depending on tissue context.",
        fontsize=6.2,
        color=SUBTLE,
    )
    fig.text(
        0.075,
        0.055,
        "Program labels combine LR mechanism priors, pathway enrichment where available, module activity, downstream impact and expected-flow summaries. Small panels are not whole-LR atlases.",
        fontsize=5.7,
        color=SUBTLE,
    )
    save_figure(fig, FIG_DIR / "phase11_communication_program_summary")
    plt.close(fig)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def build_source(programs: pd.DataFrame, matrix: pd.DataFrame) -> dict[str, pd.DataFrame]:
    programs = programs.copy()
    programs["module_label"] = programs["dataset"].map(short_dataset) + " " + programs["module"]
    programs["evidence_score"] = programs[["mean_downstream_delta_r2", "mean_expected_flow_fraction"]].fillna(0).max(axis=1)
    table = choose_interpretation_rows(programs)
    return {"programs": programs, "family_matrix": matrix, "table": table}


def plot_program_heatmap(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    mat = matrix[matrix["program_family"].isin(DISPLAY_FAMILIES)].copy()
    mat["module_label"] = mat["dataset"].map(short_dataset) + " " + mat["module"]
    modules = list(dict.fromkeys(mat["module_label"]))
    data = mat.pivot_table(index="module_label", columns="program_family", values="module_fraction", aggfunc="first").reindex(modules).reindex(columns=DISPLAY_FAMILIES).fillna(0)
    im = ax.imshow(data.to_numpy(dtype=float), cmap="Blues", norm=Normalize(0, 1), aspect="auto")
    ax.set_title("Program-family composition", loc="left", pad=5)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=5.1)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels([short_family(x) for x in data.columns], rotation=38, ha="right", fontsize=5.0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=4.6, color="white" if val > 0.58 else TEXT)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=5, length=2)


def plot_module_evidence(ax: plt.Axes, programs: pd.DataFrame) -> None:
    plot = programs[~programs["top_program_family"].isin(["high-expression negative", "background LR-like decoy"])].copy()
    x = plot["mean_expected_flow_fraction"].fillna(0)
    y = plot["mean_downstream_delta_r2"].fillna(0)
    sizes = 35 + 22 * plot["n_lr"].clip(lower=1)
    colors = [PROGRAM_COLORS.get(fam, "#999999") for fam in plot["top_program_family"]]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.88, edgecolor="white", linewidth=0.45)
    for row in plot.itertuples(index=False):
        if row.mean_expected_flow_fraction > 0.4 or row.mean_downstream_delta_r2 > 0.05 or row.top_program_family in {"myeloid-remodeling", "angiogenesis"}:
            ax.text(
                (0 if pd.isna(row.mean_expected_flow_fraction) else row.mean_expected_flow_fraction) + 0.018,
                (0 if pd.isna(row.mean_downstream_delta_r2) else row.mean_downstream_delta_r2) + 0.002,
                f"{short_dataset(row.dataset)} {row.module}",
                fontsize=4.9,
                color=SUBTLE,
            )
    ax.set_title("Program evidence", loc="left", pad=5)
    ax.set_xlabel("mean expected-flow fraction")
    ax.set_ylabel("mean downstream delta R2")
    ax.grid(color=GRID, linewidth=0.45)
    ax.set_xlim(-0.02, max(0.85, float(x.max()) + 0.08))
    ax.set_ylim(-0.005, max(0.10, float(y.max()) + 0.02))


def plot_negative_leakage(ax: plt.Axes, programs: pd.DataFrame) -> None:
    plot = programs.copy()
    plot["module_label"] = plot["dataset"].map(short_dataset) + " " + plot["module"]
    y = np.arange(len(plot))
    ax.barh(y, plot["n_high_expression_negative"], color=PROGRAM_COLORS["high-expression negative"], height=0.55, edgecolor="none")
    ax.scatter(plot["n_true_lr"], y, color="#0F4D92", s=22, zorder=3, label="true LR count")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["module_label"], fontsize=5.1)
    ax.invert_yaxis()
    ax.set_xlabel("negative LR count")
    ax.set_title("Negative-control leakage", loc="left", pad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.0)


def plot_interpretation_table(ax: plt.Axes, table: pd.DataFrame) -> None:
    ax.axis("off")
    rows = []
    for row in table.itertuples(index=False):
        evidence = []
        if pd.notna(row.mean_expected_flow_fraction):
            evidence.append(f"flow {row.mean_expected_flow_fraction:.2f}")
        if pd.notna(row.mean_downstream_delta_r2):
            evidence.append(f"dR2 {row.mean_downstream_delta_r2:.3f}")
        if pd.notna(row.pathway_or_enrichment_fdr) and row.dataset_key == "visium_hd_crc":
            evidence.append(f"path FDR {row.pathway_or_enrichment_fdr:.1e}")
        rows.append(
            [
                short_dataset(row.dataset),
                row.module,
                short_family(row.top_program_family),
                short_niche(row.top_niche),
                "; ".join(evidence) if evidence else "supportive",
                compact_interpretation(row.interpretation),
            ]
        )
    tbl = ax.table(
        cellText=rows,
        colLabels=["Dataset", "Module", "program", "top niche", "evidence", "interpretation"],
        colWidths=[0.12, 0.08, 0.15, 0.14, 0.22, 0.29],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.1)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.34)
        cell.set_edgecolor("#D6D6D6")
        if r == 0:
            cell.set_facecolor("#ECEFF3")
            cell.set_text_props(weight="bold", color=TEXT)
        elif c == 2:
            fam = table.iloc[r - 1]["top_program_family"] if r > 0 else ""
            cell.set_facecolor(lighten(PROGRAM_COLORS.get(fam, "#FFFFFF"), 0.82))
        else:
            cell.set_facecolor("white")
    ax.set_title("Representative SpatialSelf communication programs", loc="left", pad=5)


def choose_interpretation_rows(programs: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        ("visium_hd_crc", "M3"),
        ("visium_hd_crc", "M1"),
        ("lymph_node_expanded", "M1"),
        ("lymph_node_expanded", "M4"),
        ("lymph_node_full", "M2"),
        ("lymph_node_full", "M1"),
    ]
    rows = []
    for dataset_key, module in wanted:
        sub = programs[programs["dataset_key"].eq(dataset_key) & programs["module"].eq(module)]
        if not sub.empty:
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows)


def short_dataset(value: str) -> str:
    return str(value).replace("Visium HD CRC", "CRC").replace("Lymph node expanded benchmark", "LN bench").replace("Native human lymph node Visium", "LN full")


def short_family(value: str) -> str:
    return (
        str(value)
        .replace(" recruitment", "")
        .replace(" remodeling", "")
        .replace("contact-dependent immune signaling", "contact")
        .replace("inflammatory/paracrine", "paracrine")
        .replace("high-expression negative", "negative")
        .replace("background LR-like decoy", "decoy")
    )


def short_niche(value: object) -> str:
    text = str(value)
    text = text.replace("tumor_myeloid_boundary", "tumor-myeloid")
    text = text.replace("tumor_immune_boundary", "tumor-immune")
    text = text.replace("tumor_stromal_boundary", "tumor-stromal")
    text = text.replace("T_cell_zone_proxy", "T zone")
    text = text.replace("B_follicle_proxy", "B follicle")
    text = text.replace("_proxy", "")
    text = text.replace("_", " ")
    return text


def compact_interpretation(value: object) -> str:
    text = str(value)
    text = text.replace("module with spatial-flow support.", "+ flow")
    text = text.replace("module with downstream target-response support.", "+ target response")
    text = text.replace("program localized to", "localized:")
    text = text.replace("/ negative-control module; interpret as leakage or background stress.", "negative-control stress")
    text = text.replace("tumor_myeloid_boundary", "tumor-myeloid")
    text = text.replace("tumor_immune_boundary", "tumor-immune")
    text = text.replace("_", " ")
    return text


def lighten(hex_color: str, amount: float) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
    rgb = rgb + (255 - rgb) * amount
    return "#" + "".join(f"{int(x):02x}" for x in np.clip(rgb, 0, 255))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    y = 1.14 if label == "d" else 1.08
    ax.text(-0.16, y, label, transform=ax.transAxes, fontsize=8, fontweight="bold", color=TEXT, ha="left", va="top")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
