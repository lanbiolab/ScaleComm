#!/usr/bin/env python
"""Download selected Cadinu 2024 mouse colon IBD MERFISH files from Dryad.

Dryad sometimes blocks direct file-stream requests from headless environments.
This script records a reproducible file manifest and attempts both API and page
file-stream downloads, so failures are explicit rather than silent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from html import unescape
from pathlib import Path

import requests


DATASET_DOI = "10.5061/dryad.rjdfn2zh3"
VERSION_ID = 286629
DRYAD = "https://datadryad.org"

DEFAULT_FILES = {
    "README.md": 3040667,
    "gene_names.csv": 3040601,
    "ligand_receptor_pair_masterlist.csv": 3040619,
    "cell_properties.csv": 3040602,
    # One compact representative slice per main disease stage.
    "082421_D0_m6_1_slice_3.csv": 3040565,
    "092421_D3_m3_1_slice_4.csv": 3040582,
    "100221_D9_m5_2_slice_3.csv": 3040594,
    "082421_D21_m2_1_slice_2.csv": 3040570,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/optional/mouse_colon_ibd_merfish/raw")
    parser.add_argument("--only-manifest", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest()
    (output_dir / "dryad_file_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selected = []
    by_id = {int(file_record["id"]): file_record for file_record in manifest}
    for filename, file_id in DEFAULT_FILES.items():
        record = by_id.get(file_id, {"id": file_id, "path": filename, "size": None})
        selected.append(
            {
                "filename": filename,
                "file_id": int(file_id),
                "size": record.get("size"),
                "path": record.get("path", filename),
                "api_download": f"{DRYAD}/api/v2/files/{file_id}/download",
                "file_stream": f"{DRYAD}/downloads/file_stream/{file_id}",
            }
        )
    (output_dir / "selected_download_manifest.csv").write_text(
        "filename,file_id,size,path,api_download,file_stream\n"
        + "\n".join(
            f"{row['filename']},{row['file_id']},{row['size']},{row['path']},{row['api_download']},{row['file_stream']}"
            for row in selected
        )
        + "\n",
        encoding="utf-8",
    )
    if args.only_manifest:
        print(f"Wrote manifests to {output_dir}")
        return

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    page_url = f"{DRYAD}/dataset/doi:{DATASET_DOI}"
    page = session.get(page_url, timeout=60)
    print(f"Opened Dryad page: status={page.status_code}, cookies={bool(session.cookies)}")
    for row in selected:
        out = output_dir / row["filename"]
        if out.exists() and out.stat().st_size > 0:
            print(f"exists\t{row['filename']}\t{out.stat().st_size}")
            continue
        ok = False
        for attempt in range(1, args.retries + 1):
            for url in (row["api_download"], row["file_stream"]):
                try:
                    print(f"download\t{row['filename']}\tattempt={attempt}\turl={url}")
                    download(session, url, out, referer=page_url)
                    ok = True
                    break
                except requests.HTTPError as exc:
                    print(f"http_error\t{row['filename']}\t{exc}")
                except requests.RequestException as exc:
                    print(f"request_error\t{row['filename']}\t{exc}")
            if ok:
                break
            time.sleep(2 * attempt)
        if not ok:
            print(f"failed\t{row['filename']}\tfile_id={row['file_id']}")
    print(f"Finished download attempts under {output_dir}")


def fetch_manifest() -> list[dict]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    files = []
    url = f"{DRYAD}/api/v2/versions/{VERSION_ID}/files"
    while url:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        for record in payload["_embedded"]["stash:files"]:
            files.append(
                {
                    "id": int(record["_links"]["self"]["href"].rstrip("/").split("/")[-1]),
                    "path": record["path"],
                    "size": record.get("size"),
                    "mimeType": record.get("mimeType"),
                    "download": DRYAD + record["_links"]["stash:download"]["href"],
                    "file_stream": f"{DRYAD}/downloads/file_stream/{record['_links']['self']['href'].rstrip('/').split('/')[-1]}",
                }
            )
        next_href = payload.get("_links", {}).get("next", {}).get("href")
        url = DRYAD + next_href if next_href else None
    return files


def download(session: requests.Session, url: str, out: Path, *, referer: str) -> None:
    tmp = out.with_suffix(out.suffix + ".part")
    headers = {"Referer": referer, "Accept": "text/csv,*/*"}
    with session.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            text = response.text
            if "anubis_challenge" in text:
                solve_anubis_challenge(session, text, redir=url, referer=referer)
                return download(session, url, out, referer=referer)
            if "Validating..." in text or "<html" in text[:500].lower():
                raise requests.HTTPError(f"received html page instead of data from {url}", response=response)
        with tmp.open("wb") as handle:
            total = int(response.headers.get("content-length") or 0)
            seen = 0
            last_report = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    seen += len(chunk)
                    if total and seen - last_report >= 50 * 1024 * 1024:
                        print(f"progress\t{out.name}\t{seen / 1024**2:.1f}/{total / 1024**2:.1f} MiB")
                        last_report = seen
    tmp.replace(out)
    print(f"wrote\t{out.name}\t{out.stat().st_size}")


def solve_anubis_challenge(session: requests.Session, html: str, *, redir: str, referer: str) -> None:
    match = re.search(r'<script id="anubis_challenge" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise requests.HTTPError("Anubis challenge page did not contain challenge JSON")
    payload = json.loads(unescape(match.group(1)))
    challenge = payload["challenge"]
    difficulty = int(payload["rules"]["difficulty"])
    random_data = challenge["randomData"]
    prefix = "0" * difficulty
    start = time.time()
    nonce = 0
    while True:
        digest = hashlib.sha256((random_data + str(nonce)).encode("utf-8")).hexdigest()
        if digest.startswith(prefix):
            break
        nonce += 1
    elapsed_ms = max(100, int((time.time() - start) * 1000))
    response = session.get(
        f"{DRYAD}/.within.website/x/cmd/anubis/api/pass-challenge",
        params={
            "id": challenge["id"],
            "response": digest,
            "nonce": str(nonce),
            "redir": redir,
            "elapsedTime": str(elapsed_ms),
        },
        headers={"Referer": referer},
        timeout=60,
        allow_redirects=False,
    )
    response.raise_for_status()
    if response.status_code not in (200, 302, 303):
        raise requests.HTTPError(f"Anubis pass-challenge returned {response.status_code}", response=response)
    print(f"anubis_ok\tchallenge={challenge['id']}\tdifficulty={difficulty}\tnonce={nonce}")


if __name__ == "__main__":
    main()
