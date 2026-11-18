"""Regenerate the six Figure 2 simulated benchmark settings from the spatial references."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from injection import _build_dataset
from noise import NOISE_LEVELS, _apply_expression_noise
from semisynthetic import SemiSyntheticConfig, generate_semisynthetic_real_background

HERE = Path(__file__).resolve().parent
# Realized NumPy seeds for the archived perturbations. The historical runner
# derived them using process-dependent Python hashes.
SEEDS = {
    "lymph": {"clean": 20260629, "low_noise": 20524170, "high_noise": 20515523},
    "melanoma": {"clean": 20260647, "low_noise": 20437829, "high_noise": 20601864},
}


def write_regions(h5ad: Path, output: Path) -> None:
    sample = ad.read_h5ad(h5ad, backed="r")
    try:
        abundance = sample.obsm["means_cell_abundance_w_sf"]
        fractions = abundance.div(abundance.sum(axis=1), axis=0).fillna(0.0)
        gc = fractions.filter(like="B_GC")
        regions = np.select(
            [
                (gc.sum(axis=1) > 0.2) | (gc > 0.05).any(axis=1),
                fractions.filter(like="B_").sum(axis=1) > 0.35,
                (fractions.filter(like="T_CD4").sum(axis=1) > 0.35)
                | (fractions.filter(like="T_TIM3").sum(axis=1) > 0.2),
            ],
            ["Germinal center", "B follicle", "T cell zone"],
            default="Mixed",
        )
        pd.DataFrame({"spot": sample.obs_names, "region": regions}).to_csv(output, index=False)
    finally:
        sample.file.close()


def generate(lymph_h5ad: Path, melanoma_dir: Path, output: Path, background_mode: str) -> None:
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".figure2-") as tmp:
        stage = Path(tmp)
        files = {}
        region_csv = stage / "lymph_regions.csv"
        write_regions(lymph_h5ad, region_csv)
        for tissue, seeds in SEEDS.items():
            cfg = json.loads((HERE / f"{tissue}_background.json").read_text())
            cfg.update(output_dir=str(stage / "backgrounds"), simulator_r_script=str(HERE / "reference_simulator_bridge.R"))
            if tissue == "lymph":
                cfg.update(h5ad_path=str(lymph_h5ad), region_csv=str(region_csv))
            else:
                cfg.update(matrix_dir=str(melanoma_dir),
                           spatial_csv=str(melanoma_dir / "HumanMelanomaMultiome_spatial.csv"),
                           metadata_csv=str(melanoma_dir / "HumanMelanomaMultiome_metadata.csv"))
            print(f"Generating {tissue} scDesign3 background", flush=True)
            background = generate_semisynthetic_real_background(SemiSyntheticConfig(**cfg))
            if background_mode == "archived":
                current = pd.read_csv(background / "expression.csv")
                row_ids = current.pop("cell_id")
                fingerprint = hashlib.sha256(("\n".join(row_ids.astype(str)) + "\n").encode()).hexdigest()
                draw_dir = HERE / "reference_draws"
                expected = json.loads((draw_dir / "row_order_sha256.json").read_text())[tissue]
                archived = pd.read_csv(draw_dir / f"{tissue}.csv")
                if fingerprint != expected or list(archived.columns) != list(current.columns):
                    raise ValueError(f"The {tissue} reference cell order or genes do not match the archived draw.")
                archived.insert(0, "cell_id", row_ids)
                # The six-LR stage supplies expression and cells to the 200-LR generator;
                # leave its independently refitted intermediate files intact.
                source = stage / "archived_inputs" / tissue
                source.mkdir(parents=True)
                archived.to_csv(source / "expression.csv", index=False)
                for filename in ("cells.csv", "metadata.json"):
                    shutil.copy2(background / filename, source / filename)
                background = source
            tissue_name = "lymph_node" if tissue == "lymph" else tissue
            clean = _build_dataset(
                dataset_id=f"{tissue}_scdesign3_200lr", source=background, output_dir=stage / "clean",
                injection_set=f"range_{tissue_name}_200", dataset_name=f"reference_{tissue_name}_scdesign3_200lr",
                tissue=tissue_name, seed=seeds["clean"], max_candidates_per_lr=700,
                max_candidates_per_sender_lr=32, negative_edge_fraction=0.35,
            )
            files[f"{tissue}_clean"] = str(clean.relative_to(stage))
            for level, params in NOISE_LEVELS.items():
                dest = stage / level / clean.name
                shutil.copytree(clean, dest)
                _apply_expression_noise(dest, params, np.random.default_rng(seeds[level]))
                files[f"{tissue}_{level}"] = str(dest.relative_to(stage))
        (stage / "settings.json").write_text(json.dumps(files, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lymph-h5ad", type=Path, required=True)
    p.add_argument("--melanoma-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New directory for the generated datasets")
    p.add_argument("--background-mode", choices=("archived", "refit"), default="archived")
    args = p.parse_args()
    generate(args.lymph_h5ad.resolve(), args.melanoma_dir.resolve(), args.output.resolve(), args.background_mode)
