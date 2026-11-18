"""Run the native COMMOT or CellChat estimator on a generated benchmark setting."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def run(source, output, method, rscript):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        pred = tmp / "predictions.csv"
        cell_cols = ["cell_id", "x", "y", "cell_type"]
        candidate_cols = ["candidate_id", "sender_id", "receiver_id", "ligand", "receptor",
                          "distance", "sender_cell_type", "receiver_cell_type"]
        cells = pd.read_csv(source / "cells.csv", usecols=cell_cols)
        candidates = pd.read_csv(source / "candidate_interactions.csv", usecols=candidate_cols)
        if method == "commot":
            from commot_adapter import score_commot
            expression = pd.read_csv(source / "expression.csv")
            pairs = pd.read_csv(source / "lr_pairs.csv", usecols=["ligand", "receptor", "mechanism"])
            score_commot(cells, expression, pairs, candidates, cot_eps_p=0.1,
                         cot_nitermax=500).to_csv(pred, index=False)
        else:
            # The historical CellChat adapter used simulator communication
            # class only for the contact/secreted annotation in the LR database.
            pairs = pd.read_csv(source / "lr_pairs.csv", usecols=["ligand", "receptor", "mechanism"])
            cells.to_csv(tmp / "cells.csv", index=False)
            candidates.to_csv(tmp / "candidates.csv", index=False)
            pairs.to_csv(tmp / "pairs.csv", index=False)
            subprocess.run([rscript, str(HERE / "cellchat.R"), str(tmp / "cells.csv"),
                            str(source / "expression.csv"), str(tmp / "pairs.csv"),
                            str(tmp / "candidates.csv"), str(pred), "10"], check=True)
        pred.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["commot", "cellchat"], required=True)
    p.add_argument("--setting", action="append")
    p.add_argument("--rscript", default="Rscript", help="Rscript with CellChat installed")
    args = p.parse_args()
    settings = json.loads((args.data / "settings.json").read_text())
    for key in args.setting or settings:
        print(f"Running {args.method} on {key}", flush=True)
        run(args.data / settings[key], args.output / key / f"{args.method}_predictions.csv",
            args.method, args.rscript)
