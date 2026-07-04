#!/usr/bin/env python
"""Compose the SpatialSelf Explorer tool figure package.

The figure combines the interactive web UI screenshot with the static 2D
rendering generated from the same SpatialSelf Explorer payload. It is intended
as a manuscript/supplement tool showcase rather than a biological validation
panel.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


OUT_DIR = ROOT / "outputs/current_web_ui"
UI_SCREENSHOT = OUT_DIR / "spatialself_explorer_product_ui_qc_tool.png"
PAPER_PANEL = OUT_DIR / "spatialself_explorer_2d_paper_panel.png"
UI_MANIFEST = OUT_DIR / "spatialself_explorer_ui_qc_manifest.json"
PANEL_MANIFEST = OUT_DIR / "spatialself_explorer_2d_paper_panel_manifest.json"
BASE = OUT_DIR / "spatialself_explorer_tool_figure"

INK = "#172126"
MUTED = "#67757D"
LINE = "#DCE5E8"
ACCENT = "#237A78"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_img(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def crop_ui(img: np.ndarray) -> np.ndarray:
    """Crop the full web screenshot to focus on the interactive explorer."""
    h, w = img.shape[:2]
    # Keep the main canvas plus right inspector, remove only excessive outer
    # browser-like whitespace. Coordinates are fractions of the generated QC
    # screenshot, so this remains stable across machines.
    y0 = int(0.060 * h)
    y1 = int(0.905 * h)
    x0 = int(0.006 * w)
    x1 = int(0.995 * w)
    return img[y0:y1, x0:x1]


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 - 0.008, bbox.y1 + 0.010, label, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="left")


def build_figure() -> tuple[plt.Figure, dict]:
    setup_style()
    ui = crop_ui(read_img(UI_SCREENSHOT))
    panel = read_img(PAPER_PANEL)
    ui_manifest = json.loads(UI_MANIFEST.read_text())
    panel_manifest = json.loads(PANEL_MANIFEST.read_text())

    fig = plt.figure(figsize=(7.2, 2.72), dpi=300)
    ax_a = fig.add_axes([0.035, 0.110, 0.455, 0.765])
    ax_b = fig.add_axes([0.535, 0.110, 0.430, 0.765])
    for ax in [ax_a, ax_b]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE)
            spine.set_linewidth(0.55)

    ax_a.imshow(ui)
    ax_b.imshow(panel)

    add_panel_label(fig, ax_a, "a")
    add_panel_label(fig, ax_b, "b")
    fig.text(0.035, 0.940, "Interactive web Explorer", fontsize=7.0, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.035, 0.912, "LR search, cell-type filtering, thresholding, event inspection, and export.", fontsize=5.2, color=MUTED, ha="left", va="bottom")
    fig.text(0.535, 0.940, "Static paper rendering", fontsize=7.0, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.535, 0.912, "Same SpatialSelf payload rendered as a publication panel.", fontsize=5.2, color=MUTED, ha="left", va="bottom")

    # Small consistency badge: it helps readers see that panel b is not a
    # hand-drawn schematic but the same SpatialSelf output rendered statically.
    badge = (
        f"{panel_manifest['lr']} | "
        f"{panel_manifest['visible_edges']:,}/{panel_manifest['filter_pool']:,} top events | "
        f"dominant scale: {panel_manifest['dominant_scale']}"
    )
    fig.text(0.535, 0.060, badge, fontsize=5.2, color=ACCENT, fontweight="bold", ha="left", va="bottom")

    manifest = {
        "purpose": "SpatialSelf Explorer tool showcase figure",
        "panel_a": {
            "source": str(UI_SCREENSHOT),
            "description": "Interactive web Explorer screenshot",
            "smoke_test": ui_manifest.get("smoke_test", {}),
        },
        "panel_b": {
            "source": str(PAPER_PANEL),
            "description": "Static 2D rendering generated from the same Explorer payload",
            "data_manifest": panel_manifest,
        },
        "claim_boundary": "Tool completeness and browsable result visualization; not a standalone biological validation figure.",
    }
    return fig, manifest


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, manifest = build_figure()
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(BASE.with_suffix(f".{ext}"), dpi=450 if ext == "png" else None)
    plt.close(fig)
    manifest["png"] = str(BASE.with_suffix(".png"))
    manifest["pdf"] = str(BASE.with_suffix(".pdf"))
    manifest["svg"] = str(BASE.with_suffix(".svg"))
    manifest_path = BASE.with_name(BASE.name + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
