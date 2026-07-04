#!/usr/bin/env python
"""Resume CosMx NSCLC Lung9 Rep1 flat-data archive with parallel range requests."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = (
    "https://nanostring-public-share.s3.us-west-2.amazonaws.com/"
    "SMI-Compressed/Lung9_Rep1/Lung9_Rep1+SMI+Flat+data.tar.gz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/optional/cosmx_nsclc_lung9_rep1/raw/Lung9_Rep1_SMI_Flat_data.tar.gz"


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    path: Path

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-mb", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--total-bytes", type=int, default=None, help="Known remote size; skips HEAD when provided.")
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    total = args.total_bytes or _remote_size(args.url, timeout=args.timeout, retries=args.retries)
    prefix_size = output.stat().st_size if output.exists() else 0
    if prefix_size > total:
        raise RuntimeError(f"Local file is larger than remote: local={prefix_size}, remote={total}")
    if prefix_size == total:
        print(f"Already complete: {output} ({total} bytes)")
        return

    chunk_dir = output.with_suffix(output.suffix + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = _make_chunks(prefix_size, total, args.chunk_mb * 1024 * 1024, chunk_dir)
    print(
        f"Remote bytes={total}; local prefix={prefix_size}; "
        f"remaining={total - prefix_size}; chunks={len(chunks)}; workers={args.workers}",
        flush=True,
    )

    started = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_download_chunk, args.url, chunk, args.timeout, args.retries) for chunk in chunks]
        completed = 0
        for future in cf.as_completed(futures):
            chunk = future.result()
            completed += 1
            done_bytes = prefix_size + sum(c.size for c in chunks if c.path.exists() and c.path.stat().st_size == c.size)
            elapsed = max(time.time() - started, 1e-6)
            rate = done_bytes / elapsed / 1024 / 1024
            print(
                f"chunk {chunk.index:03d}/{len(chunks):03d} done; "
                f"{done_bytes / total:0.1%}; {rate:0.2f} MB/s",
                flush=True,
            )

    _assemble(output, total, chunks)
    print(f"Complete: {output} ({output.stat().st_size} bytes)")


def _remote_size(url: str, *, timeout: int, retries: int) -> int:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                size = response.headers.get("Content-Length")
                if size is None:
                    raise RuntimeError("Remote server did not return Content-Length")
                return int(size)
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Could not read remote size after {retries} attempts: {last_error}")


def _make_chunks(prefix_size: int, total: int, chunk_size: int, chunk_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = prefix_size
    index = 0
    while start < total:
        end = min(total - 1, start + chunk_size - 1)
        path = chunk_dir / f"chunk_{index:04d}_{start}_{end}.part"
        chunks.append(Chunk(index=index, start=start, end=end, path=path))
        start = end + 1
        index += 1
    return chunks


def _download_chunk(url: str, chunk: Chunk, timeout: int, retries: int) -> Chunk:
    if chunk.path.exists() and chunk.path.stat().st_size == chunk.size:
        return chunk
    tmp = chunk.path.with_suffix(chunk.path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        if tmp.exists():
            tmp.unlink()
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={chunk.start}-{chunk.end}"})
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            observed = tmp.stat().st_size
            if observed != chunk.size:
                raise RuntimeError(f"Chunk {chunk.index} size mismatch: expected={chunk.size}, observed={observed}")
            os.replace(tmp, chunk.path)
            return chunk
        except Exception as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Chunk {chunk.index} failed after {retries} attempts: {last_error}")


def _assemble(output: Path, total: int, chunks: list[Chunk]) -> None:
    prefix_size = output.stat().st_size if output.exists() else 0
    assembled = output.with_suffix(output.suffix + ".assembled")
    if assembled.exists():
        assembled.unlink()
    with assembled.open("wb") as out:
        if output.exists():
            with output.open("rb") as prefix:
                shutil.copyfileobj(prefix, out, length=1024 * 1024)
        for chunk in chunks:
            if not chunk.path.exists() or chunk.path.stat().st_size != chunk.size:
                raise RuntimeError(f"Missing or incomplete chunk before assembly: {chunk.path}")
            with chunk.path.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=1024 * 1024)
    observed = assembled.stat().st_size
    expected = prefix_size + sum(c.size for c in chunks)
    if observed != expected or observed != total:
        assembled.unlink(missing_ok=True)
        raise RuntimeError(f"Assembly size mismatch: observed={observed}, expected={expected}, remote={total}")
    os.replace(assembled, output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
