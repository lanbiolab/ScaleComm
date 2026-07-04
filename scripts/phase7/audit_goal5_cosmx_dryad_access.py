#!/usr/bin/env python
"""Audit Dryad access status for the CosMx melanoma Goal 5 dataset."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOAL5_DIR = PROJECT_ROOT / "outputs/phase7/functional_cross_platform_validation/goal5_cross_platform_real_data"
DATASET_DOI = "10.5061/dryad.ksn02v7b1"
DATASET_API = "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.ksn02v7b1"
DATASET_PAGE = "https://datadryad.org/dataset/doi:10.5061/dryad.ksn02v7b1"
USER_AGENT = "Mozilla/5.0 SpatialSelf Phase7 audit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(GOAL5_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    dataset_status, dataset_payload = _get_json(DATASET_API, timeout=int(args.timeout))
    rows.append(_status_row("dataset_api", DATASET_API, dataset_status, dataset_payload))

    versions_url = f"{DATASET_API}/versions"
    versions_status, versions_payload = _get_json(versions_url, timeout=int(args.timeout))
    rows.append(_status_row("versions_api", versions_url, versions_status, versions_payload))

    version_id = _extract_version_id(versions_payload)
    files_payload: dict[str, Any] = {}
    if version_id:
        files_url = f"https://datadryad.org/api/v2/versions/{version_id}/files"
        files_status, files_payload = _get_json(files_url, timeout=int(args.timeout))
        rows.append(_status_row("version_files_api", files_url, files_status, files_payload))
    else:
        rows.append(
            {
                "probe": "version_files_api",
                "url": "",
                "http_status": "",
                "ok": False,
                "error": "version_id_not_found",
                "response_note": "",
            }
        )

    page_status, page_note = _probe(DATASET_PAGE, timeout=int(args.timeout), read_bytes=256)
    rows.append(_probe_row("dataset_landing_page", DATASET_PAGE, page_status, page_note))

    files = _extract_files(files_payload)
    file_rows = []
    for file_meta in files:
        file_id = str(file_meta.get("id", ""))
        filename = str(file_meta.get("path") or file_meta.get("filename") or "")
        api_download = f"https://datadryad.org/api/v2/files/{file_id}/download" if file_id else ""
        stream_download = f"https://datadryad.org/downloads/file_stream/{file_id}" if file_id else ""
        api_status, api_note = _probe(api_download, timeout=int(args.timeout), read_bytes=1) if api_download else ("", "")
        stream_status, stream_note = (
            _probe(stream_download, timeout=int(args.timeout), read_bytes=1) if stream_download else ("", "")
        )
        file_rows.append(
            {
                "file_id": file_id,
                "filename": filename,
                "size_bytes": file_meta.get("size", ""),
                "mime_type": file_meta.get("mimeType", ""),
                "api_download_url": api_download,
                "api_download_status": api_status,
                "api_download_note": api_note,
                "file_stream_url": stream_download,
                "file_stream_status": stream_status,
                "file_stream_note": stream_note,
                "download_accessible": str(api_status).startswith("2") or str(stream_status).startswith("2"),
            }
        )

    status_table = pd.DataFrame(rows)
    file_table = pd.DataFrame(file_rows)
    status_path = output_dir / "cosmx_melanoma_dryad_access_status.csv"
    file_path = output_dir / "cosmx_melanoma_dryad_file_access.csv"
    status_table.to_csv(status_path, index=False)
    file_table.to_csv(file_path, index=False)

    manifest = {
        "dataset": "cosmx_melanoma",
        "doi": DATASET_DOI,
        "dataset_page": DATASET_PAGE,
        "dataset_api": DATASET_API,
        "version_id": version_id,
        "n_files_listed": int(len(file_table)),
        "n_files_download_accessible": int(file_table.get("download_accessible", pd.Series(dtype=bool)).astype(bool).sum())
        if not file_table.empty
        else 0,
        "status_path": _rel(status_path),
        "file_access_path": _rel(file_path),
        "claim_boundary": (
            "This is a data access audit only. File listing availability does not mean the expression matrix is "
            "available locally or that SpatialSelf/baselines have been run."
        ),
    }
    manifest_path = output_dir / "cosmx_melanoma_dryad_access_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(status_table.to_string(index=False))
    print(file_table.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def _get_json(url: str, *, timeout: int) -> tuple[str, dict[str, Any]]:
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return str(response.status), payload
    except urllib.error.HTTPError as exc:
        return str(exc.code), {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - audit script records transport failures.
        return "error", {"error": f"{type(exc).__name__}: {exc}"}


def _probe(url: str, *, timeout: int, read_bytes: int) -> tuple[str, str]:
    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as response:
            data = response.read(read_bytes)
            return str(response.status), f"content_type={response.headers.get('Content-Type')};bytes_read={len(data)}"
    except urllib.error.HTTPError as exc:
        return str(exc.code), str(exc)
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _extract_version_id(payload: dict[str, Any]) -> str:
    versions = payload.get("_embedded", {}).get("stash:versions", []) if isinstance(payload, dict) else []
    if not versions:
        return ""
    href = versions[0].get("_links", {}).get("self", {}).get("href", "")
    return str(href).rstrip("/").split("/")[-1] if href else ""


def _extract_files(payload: dict[str, Any]) -> list[dict[str, Any]]:
    files = payload.get("_embedded", {}).get("stash:files", []) if isinstance(payload, dict) else []
    out = []
    for file_meta in files:
        row = dict(file_meta)
        href = file_meta.get("_links", {}).get("stash:download", {}).get("href", "")
        if href:
            row["id"] = str(href).rstrip("/").split("/")[-2] if href.rstrip("/").endswith("download") else ""
        out.append(row)
    return out


def _status_row(probe: str, url: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe": probe,
        "url": url,
        "http_status": status,
        "ok": str(status).startswith("2"),
        "error": payload.get("error", "") if isinstance(payload, dict) else "",
        "response_note": _payload_note(payload),
    }


def _probe_row(probe: str, url: str, status: str, note: str) -> dict[str, Any]:
    return {
        "probe": probe,
        "url": url,
        "http_status": status,
        "ok": str(status).startswith("2"),
        "error": "" if str(status).startswith("2") else note,
        "response_note": note if str(status).startswith("2") else "",
    }


def _payload_note(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    if "title" in payload:
        return f"title={payload.get('title')}"
    if "count" in payload or "total" in payload:
        return f"count={payload.get('count')};total={payload.get('total')}"
    return ""


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
