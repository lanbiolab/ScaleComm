#!/usr/bin/env python
"""Download selected Dryad files through Anubis + S3 range requests.

This is a fallback for environments where Dryad's file_stream route works but
single S3 connections are extremely slow. It solves the Anubis proof-of-work,
captures the S3 presigned URL, then downloads the file in parallel byte ranges.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import threading
import time
from html import unescape
from pathlib import Path

import requests


DRYAD = "https://datadryad.org"

DEFAULT_FILES = {
    "cell_properties.csv": 3040602,
    "082421_D0_m6_1_slice_3.csv": 3040565,
    "082421_D0_m7_1_slice_1.csv": 3040568,
    "092421_D3_m3_1_slice_4.csv": 3040582,
    "092421_D3_m1_1_slice_1.csv": 3040572,
    "100221_D9_m5_2_slice_3.csv": 3040594,
    "100221_D9_m3_2_slice_2.csv": 3040588,
    "082421_D21_m2_1_slice_2.csv": 3040570,
    "082421_D21_m2_1_slice_1.csv": 3040571,
}

KNOWN_SIZES = {
    "cell_properties.csv": 299133694,
    "082421_D0_m6_1_slice_3.csv": 193268174,
    "082421_D0_m7_1_slice_1.csv": 296068219,
    "092421_D3_m3_1_slice_4.csv": 170627394,
    "092421_D3_m1_1_slice_1.csv": 207939475,
    "100221_D9_m5_2_slice_3.csv": 138154821,
    "100221_D9_m3_2_slice_2.csv": 143605974,
    "082421_D21_m2_1_slice_2.csv": 147097834,
    "082421_D21_m2_1_slice_1.csv": 292503119,
}

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/optional/mouse_colon_ibd_merfish/raw")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mb", type=float, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--file", action="append", choices=sorted(DEFAULT_FILES), help="Download only this file.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {name: DEFAULT_FILES[name] for name in args.file} if args.file else DEFAULT_FILES

    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    for filename, file_id in selected.items():
        out = output_dir / filename
        url_path = output_dir / f"{filename}.s3url"
        url = url_path.read_text(encoding="utf-8").strip() if url_path.exists() else ""
        if not url:
            url = get_s3_url(session, file_id)
            url_path.write_text(url, encoding="utf-8")
        try:
            total = get_total_size(url)
        except requests.RequestException as exc:
            if filename not in KNOWN_SIZES:
                raise
            total = KNOWN_SIZES[filename]
            print(f"size_fallback\t{filename}\t{total}\t{exc.__class__.__name__}", flush=True)
        if out.exists() and out.stat().st_size == total:
            print(f"exists\t{filename}\t{total}")
            continue
        print(f"range_download\t{filename}\t{total / 1024**2:.1f} MiB\tworkers={args.workers}", flush=True)
        range_download(url, out, total, workers=args.workers, chunk_size=int(args.chunk_mb * 1024 * 1024), retries=args.retries)
        print(f"wrote\t{filename}\t{out.stat().st_size}")


def get_s3_url(session: requests.Session, file_id: int) -> str:
    url = f"{DRYAD}/downloads/file_stream/{file_id}"
    response = None
    for attempt in range(1, 6):
        try:
            response = session.get(url, timeout=60, allow_redirects=False)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
    assert response is not None
    if response.status_code == 200 and "anubis_challenge" in response.text:
        solve_anubis(session, response.text, redir=url)
        for attempt in range(1, 6):
            try:
                response = session.get(url, timeout=60, allow_redirects=False)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2 * attempt)
    location = response.headers.get("location")
    if not location:
        raise RuntimeError(f"Dryad did not return S3 location for file_id={file_id}: status={response.status_code}")
    return location


def solve_anubis(session: requests.Session, html: str, *, redir: str) -> None:
    match = re.search(r'<script id="anubis_challenge" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise RuntimeError("Anubis challenge page did not contain challenge JSON")
    payload = json.loads(unescape(match.group(1)))
    challenge = payload["challenge"]
    difficulty = int(payload["rules"]["difficulty"])
    prefix = "0" * difficulty
    nonce = 0
    start = time.time()
    while True:
        digest = hashlib.sha256((challenge["randomData"] + str(nonce)).encode("utf-8")).hexdigest()
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
        timeout=60,
        allow_redirects=False,
    )
    response.raise_for_status()
    print(f"anubis_ok\tchallenge={challenge['id']}\tdifficulty={difficulty}\tnonce={nonce}", flush=True)


def get_total_size(url: str) -> int:
    response = None
    for attempt in range(1, 6):
        try:
            response = requests.get(url, headers={"Range": "bytes=0-0", "User-Agent": UA}, timeout=(10, 30))
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
    assert response is not None
    content_range = response.headers.get("content-range")
    if not content_range or "/" not in content_range:
        raise RuntimeError(f"S3 did not return content-range: {content_range}")
    return int(content_range.rsplit("/", 1)[1])


def range_download(url: str, out: Path, total: int, *, workers: int, chunk_size: int, retries: int) -> None:
    tmp = out.with_suffix(out.suffix + ".range_part")
    done_path = out.with_suffix(out.suffix + ".ranges_done.json")
    ranges = [(start, min(start + chunk_size - 1, total - 1)) for start in range(0, total, chunk_size)]
    done = set()
    if done_path.exists():
        done = {tuple(item) for item in json.loads(done_path.read_text(encoding="utf-8"))}
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("ab") as handle:
        handle.truncate(total)
    lock = threading.Lock()
    completed_bytes = sum(end - start + 1 for start, end in done)
    started = time.time()

    def fetch_range(byte_range: tuple[int, int]) -> tuple[int, int]:
        start, end = byte_range
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": UA}
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, headers=headers, timeout=(10, 60))
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"expected 206, got {response.status_code}")
                data = response.content
                expected = end - start + 1
                if len(data) != expected:
                    raise RuntimeError(f"expected {expected} bytes, got {len(data)}")
                with tmp.open("r+b") as handle:
                    handle.seek(start)
                    handle.write(data)
                return byte_range
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError("unreachable")

    pending = [item for item in ranges if item not in done]
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(fetch_range, item): item for item in pending}
        for future in futures.as_completed(future_map):
            item = future.result()
            done.add(item)
            completed_bytes += item[1] - item[0] + 1
            if len(done) % 5 == 0 or completed_bytes == total:
                done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"progress\t{out.name}\t{completed_bytes / 1024**2:.1f}/{total / 1024**2:.1f} MiB"
                    f"\t{completed_bytes / 1024**2 / elapsed:.2f} MiB/s",
                    flush=True,
                )
    os.replace(tmp, out)
    done_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
