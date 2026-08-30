#!/usr/bin/env python
"""Launch the generic ScaleComm viewer for one completed standardized run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--scale-profile", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.prepared_dir.is_dir() or not args.results_dir.is_dir():
        raise SystemExit("--prepared-dir and --results-dir must be existing directories.")
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit("The Run Viewer requires Streamlit. Install it with: pip install -r apps/spatialself_viewer/requirements.txt")

    app_path = Path(__file__).with_name("run_viewer.py")
    command = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", args.host,
        "--server.port", str(args.port),
    ]
    if args.no_browser:
        command.extend(["--server.headless", "true"])
    command.extend(["--", "--prepared-dir", str(args.prepared_dir.resolve()), "--results-dir", str(args.results_dir.resolve())])
    if args.prediction:
        command.extend(["--prediction", str(args.prediction.resolve())])
    if args.scale_profile:
        command.extend(["--scale-profile", str(args.scale_profile.resolve())])

    print(f"Starting ScaleComm Run Viewer at http://{args.host}:{args.port}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
