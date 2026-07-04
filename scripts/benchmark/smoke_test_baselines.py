#!/usr/bin/env python3
"""Smoke-test baseline method environments.

This checks importability and command entry points only. It does not validate
method correctness or run a benchmark dataset.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


ROOT = "/home/xd/y/SpitialSelf"
BASELINES = "/home/xd/y/BenchmarkBaselines"
CONDA_ENVS_PATH = "/home/xd/.conda/envs"
SPATALK_R_LIBS = f"{BASELINES}/r-libs/spatalk_env"


@dataclass(frozen=True)
class Check:
    name: str
    cmd: list[str]
    required: bool = True
    cwd: str = ROOT
    env: dict[str, str] | None = None


def merged_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CONDA_ENVS_PATH", CONDA_ENVS_PATH)
    if extra:
        env.update(extra)
    return env


def run_check(check: Check) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            check.cmd,
            cwd=check.cwd,
            env=merged_env(check.env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"TIMEOUT after {exc.timeout}s"

    output = (proc.stdout or "").strip()
    if proc.returncode == 0:
        return True, output.splitlines()[-1] if output else "OK"
    tail = "\n".join(output.splitlines()[-12:])
    return False, tail or f"exit code {proc.returncode}"


def main() -> int:
    python_imports = (
        "import numpy, scanpy, anndata, commot, torch, torch_geometric, DeepTalk_ST;"
        "from torch_geometric.nn import GATv2Conv;"
        "import altair, pyvis, qnorm, django;"
        "print('Benchmark Python stack OK:', numpy.__version__, torch.__version__, torch_geometric.__version__)"
    )
    checks = [
        Check(
            "Benchmark Python stack / COMMOT / DeepTalk",
            ["conda", "run", "-n", "Benchmark", "python", "-c", python_imports],
        ),
        Check(
            "CellNEST CLI",
            [
                "conda",
                "run",
                "-n",
                "Benchmark",
                "python",
                f"{BASELINES}/CellNEST/run_CellNEST.py",
                "--help",
            ],
            cwd=f"{BASELINES}/CellNEST",
        ),
        Check(
            "CellChat R package",
            [
                "conda",
                "run",
                "-n",
                "cellchat_env",
                "Rscript",
                "-e",
                "library(CellChat); cat('CellChat OK', as.character(packageVersion('CellChat')), '\\n')",
            ],
        ),
        Check(
            "SpaTalk R package",
            [
                "conda",
                "run",
                "-n",
                "spatalk_env",
                "Rscript",
                "-e",
                "library(SpaTalk); cat('SpaTalk OK', as.character(packageVersion('SpaTalk')), '\\n')",
            ],
        ),
        Check(
            "NicheNet R package",
            [
                "conda",
                "run",
                "-n",
                "nichenet_env",
                "Rscript",
                "-e",
                "library(nichenetr); cat('nichenetr OK', as.character(packageVersion('nichenetr')), '\\n')",
            ],
        ),
        Check(
            "NICHES R package",
            [
                "conda",
                "run",
                "-n",
                "spatalk_env",
                "Rscript",
                "-e",
                "library(NICHES); cat('NICHES OK', as.character(packageVersion('NICHES')), '\\n')",
            ],
            required=False,
            env={"R_LIBS_USER": SPATALK_R_LIBS},
        ),
    ]

    failed_required = False
    for check in checks:
        ok, message = run_check(check)
        status = "PASS" if ok else ("FAIL" if check.required else "SKIP")
        print(f"[{status}] {check.name}")
        if message:
            print(f"  {message}")
        if not ok and check.required:
            failed_required = True

    return 1 if failed_required else 0


if __name__ == "__main__":
    sys.exit(main())
