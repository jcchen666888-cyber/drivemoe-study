#!/usr/bin/env python3
"""Reliably download one large public Hugging Face file through range requests."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download_part(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    retries: int,
) -> int:
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        current = part_path.stat().st_size if part_path.exists() else 0
        if current == expected:
            return expected
        if current > expected:
            part_path.unlink()
            current = 0
        range_start = start + current
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={range_start}-{end}"},
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                content_range = response.headers.get("content-range", "")
                if not content_range.startswith(f"bytes {range_start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")
                with part_path.open("ab") as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            actual = part_path.stat().st_size
            if actual == expected:
                return actual
            raise RuntimeError(f"part length {actual} != {expected}")
        except Exception as exc:  # network retries are intentional here
            if attempt == retries:
                raise RuntimeError(f"failed range {start}-{end}: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == args.size:
        digest = sha256(output)
        if digest == args.sha256:
            print(f"PASS existing file: {output}")
            return

    parts_dir = output.parent / f".{output.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = []
    for index, start in enumerate(range(0, args.size, chunk_size)):
        end = min(start + chunk_size - 1, args.size - 1)
        ranges.append((index, start, end, parts_dir / f"{index:05d}.part"))

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_part,
                args.url,
                part_path,
                start,
                end,
                args.retries,
            ): (index, start, end)
            for index, start, end, part_path in ranges
        }
        for future in as_completed(futures):
            completed += future.result()
            print(
                f"progress {completed / args.size:7.2%} "
                f"({completed / 1024**3:.2f}/{args.size / 1024**3:.2f} GiB)",
                flush=True,
            )

    temporary = output.with_suffix(output.suffix + ".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as target:
        for index, start, end, part_path in ranges:
            expected = end - start + 1
            if part_path.stat().st_size != expected:
                raise RuntimeError(f"bad part {index}: {part_path.stat().st_size} != {expected}")
            with part_path.open("rb") as source:
                while chunk := source.read(16 * 1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
    if temporary.stat().st_size != args.size:
        raise RuntimeError(f"assembled size mismatch: {temporary.stat().st_size}")
    actual_hash = digest.hexdigest()
    if actual_hash != args.sha256:
        raise RuntimeError(f"assembled SHA-256 mismatch: {actual_hash}")
    os.replace(temporary, output)
    for _, _, _, part_path in ranges:
        part_path.unlink()
    parts_dir.rmdir()
    print(f"PASS size: {output.stat().st_size:,} bytes")
    print(f"PASS sha256: {actual_hash}")
    print(f"SAVED {output}")


if __name__ == "__main__":
    main()
