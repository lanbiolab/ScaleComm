"""Regenerate six semi-synthetic datasets, train ScaleComm, and score the resulting predictions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from fit import MODES, MODE_HELP

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lymph-h5ad", type=Path, required=True,
                        help="Lymph-node spatial reference with means_cell_abundance_w_sf in obsm")
    parser.add_argument("--melanoma-dir", type=Path, required=True,
                        help="Slide-tags melanoma Matrix Market counts, spatial and metadata CSV directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for datasets, fits and metrics")
    parser.add_argument("--background-mode", choices=("archived", "refit"), default="archived",
                        help="Reuse the archived simulated expression draw or sample a fresh scDesign3 draw")
    parser.add_argument("--input-mode", choices=MODES, required=True, help=MODE_HELP)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    if args.epochs < 1 or not args.lymph_h5ad.is_file() or not args.melanoma_dir.is_dir():
        parser.error("Provide the two spatial references and a positive number of epochs.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data, fits, metrics = out / "datasets", out / "fits", out / "metrics"
    subprocess.run([sys.executable, HERE / "generate.py", "--lymph-h5ad", args.lymph_h5ad,
                    "--melanoma-dir", args.melanoma_dir, "--background-mode", args.background_mode,
                    "--output", data], check=True)
    subprocess.run([sys.executable, HERE / "fit.py", "--data", data, "--output", fits,
                    "--input-mode", args.input_mode, "--device", args.device, "--epochs", str(args.epochs)], check=True)
    subprocess.run([sys.executable, HERE / "evaluate.py", "--data", data, "--predictions", fits,
                    "--output", metrics], check=True)
    (out / "run.json").write_text(json.dumps({
        "input_mode": args.input_mode, "background_mode": args.background_mode,
        "epochs": args.epochs, "device": args.device, "settings": list(json.loads((data / "settings.json").read_text()))
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
