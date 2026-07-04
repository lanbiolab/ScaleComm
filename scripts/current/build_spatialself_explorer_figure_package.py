#!/usr/bin/env python
"""Build the complete SpatialSelf Explorer figure package.

Pipeline:
1. Regenerate/QA the web Explorer and screenshots.
2. Render the static 2D paper panel from the same payload.
3. Compose the two-panel tool showcase figure.
4. Write a package manifest with figure-use guidance.
"""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/current_web_ui"
PACKAGE_MANIFEST = OUT_DIR / "spatialself_explorer_figure_package_manifest.json"


def run(script: str) -> None:
    subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run("scripts/current/qa_spatialself_explorer_ui.py")
    run("scripts/current/plot_spatialself_explorer_2d_paper_panel.py")
    run("scripts/current/plot_spatialself_explorer_tool_figure.py")

    ui_manifest = load_json(OUT_DIR / "spatialself_explorer_ui_qc_manifest.json")
    panel_manifest = load_json(OUT_DIR / "spatialself_explorer_2d_paper_panel_manifest.json")
    tool_manifest = load_json(OUT_DIR / "spatialself_explorer_tool_figure_manifest.json")

    package = {
        "name": "SpatialSelf Explorer figure package",
        "recommended_use": {
            "main_text_candidate": {
                "figure": panel_manifest["png"],
                "reason": "The 2D rendering is readable after manuscript scaling and preserves the selected LR edge count and scale summary.",
            },
            "supplement_candidate": {
                "figure": tool_manifest["png"],
                "reason": "The two-panel tool showcase documents both the interactive Explorer and the static paper rendering.",
            },
            "interactive_demo": {
                "html": ui_manifest["product_html"],
                "reason": "Full browsable Explorer with LR search, cell-type pair filtering, threshold control, click inspection, and export.",
            },
        },
        "claim_boundary": "This package supports tool completeness and browsable SpatialSelf result visualization. It should not be used as a standalone biological validation panel.",
        "shared_data_assertions": {
            "lr": panel_manifest["lr"],
            "visible_edges": panel_manifest["visible_edges"],
            "filter_pool": panel_manifest["filter_pool"],
            "dominant_scale": panel_manifest["dominant_scale"],
            "rank_q95": panel_manifest["rank_q95"],
            "ui_initial_visible_edges": ui_manifest["smoke_test"]["initial_visible_edges"],
            "ui_initial_filter_pool": ui_manifest["smoke_test"]["initial_filter_pool"],
        },
        "artifacts": {
            "web_product_html": ui_manifest["product_html"],
            "web_paper_html": ui_manifest["paper_html"],
            "web_tool_screenshot": ui_manifest["tool_screenshot"],
            "web_paper_screenshot": ui_manifest["paper_screenshot"],
            "static_2d_png": panel_manifest["png"],
            "static_2d_pdf": panel_manifest["pdf"],
            "static_2d_svg": panel_manifest["svg"],
            "tool_figure_png": tool_manifest["png"],
            "tool_figure_pdf": tool_manifest["pdf"],
            "tool_figure_svg": tool_manifest["svg"],
        },
        "captions": {
            "static_2d_panel": (
                "SpatialSelf Explorer static rendering for the Cxcl12-Cxcr4 axis in "
                "Slide-seqV2 hippocampus. The panel shows the top 20% SpatialSelf "
                "communication events from the embedded Explorer payload, together "
                "with the score threshold, learned scale profile, and dominant "
                "annotated cell-type interactions."
            ),
            "tool_showcase": (
                "SpatialSelf Explorer provides an interactive browser for LR axes, "
                "cell-type interaction filters, top-signal thresholds, learned scale "
                "profiles, event inspection, and export. A static rendering generated "
                "from the same payload provides a publication-ready view of the "
                "selected communication axis."
            ),
        },
    }

    # Hard consistency checks so the package cannot silently mix outputs from
    # different runs.
    assert str(panel_manifest["visible_edges"]) == str(ui_manifest["smoke_test"]["initial_visible_edges"]).replace(",", "")
    assert f"{panel_manifest['filter_pool']:,}" == ui_manifest["smoke_test"]["initial_filter_pool"]
    assert tool_manifest["panel_b"]["data_manifest"]["visible_edges"] == panel_manifest["visible_edges"]
    assert tool_manifest["panel_b"]["data_manifest"]["filter_pool"] == panel_manifest["filter_pool"]
    assert "not a standalone biological validation" in tool_manifest["claim_boundary"]

    PACKAGE_MANIFEST.write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps(package, indent=2))


if __name__ == "__main__":
    main()
