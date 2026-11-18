"""Train the graph-enabled ScaleComm model on one or more simulated settings."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tonsil_a1"))
from run_scalecomm import run_spatialself_on_replicate

OBSERVED_COLUMNS = [
    "candidate_id", "sender_id", "receiver_id", "lr_id", "ligand", "receptor",
    "distance", "sender_cell_type", "receiver_cell_type", "sender_niche",
    "receiver_niche", "ligand_expr", "receptor_expr",
]
MODES = ("observed", "historical_inputs")
MODE_HELP = ("observed uses observable features and anonymous LR IDs. historical_inputs "
             "also uses simulator mechanism and event-level rule/decoy annotations in training.")


def fit(source, output, device, epochs, input_mode):
    if output.exists():
        raise FileExistsError(output)
    if input_mode not in MODES or epochs < 1:
        raise ValueError("Select a supported input mode and a positive number of epochs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        temp = Path(temp)
        inputs = temp / "inputs"
        inputs.mkdir()
        for name in ("cells.csv", "expression.csv"):
            shutil.copyfile(source / name, inputs / name)
        candidate_file = source / "candidate_interactions.csv"
        candidates = pd.read_csv(candidate_file)
        pairs = pd.read_csv(source / "lr_pairs.csv")
        if not pairs.lr_id.is_unique or not candidates.lr_id.isin(pairs.lr_id).all():
            raise ValueError("Candidate LR identifiers do not map to the catalogue.")
        if input_mode == "observed":
            candidates = candidates[OBSERVED_COLUMNS].copy()
            pairs = pairs[["lr_id", "ligand", "receptor"]].copy()
            keys = {key: f"lr_{i:04d}" for i, key in enumerate(sorted(pairs.lr_id))}
            candidates["lr_id"] = candidates.lr_id.map(keys)
            pairs["lr_id"] = pairs.lr_id.map(keys)
        else:
            candidates = candidates.drop(columns=["ground_truth"])
        candidates.to_csv(inputs / "candidate_interactions.csv", index=False)
        pairs.to_csv(inputs / "lr_pairs.csv", index=False)
        config = json.loads((HERE / "training.json").read_text())
        config.update(epochs=epochs, device=device)
        result = temp / "results"
        run_spatialself_on_replicate(inputs, result, **config)
        metadata = dict(configuration=config, input_mode=input_mode,
                        candidate_fields=list(candidates), lr_fields=list(pairs),
                        candidate_table_sha256=hashlib.sha256(candidate_file.read_bytes()).hexdigest())
        (result / "fit_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        result.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", action="append", help="Repeat to select settings. Default: all six.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input-mode", choices=MODES, required=True, help=MODE_HELP)
    args = parser.parse_args()
    settings = json.loads((args.data / "settings.json").read_text())
    for setting in args.setting or settings:
        if setting not in settings:
            parser.error(f"Unknown setting: {setting}")
        print(f"Training {setting}", flush=True)
        fit(args.data / settings[setting], args.output / setting, args.device, args.epochs, args.input_mode)
