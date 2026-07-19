#!/usr/bin/env python3
"""Verify every asset in the minimal DriveMoE contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_assets import CHECKPOINT, LABEL_FILES, ROUTES, sha256, verify_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hash", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint = root / "ckpts" / CHECKPOINT["filename"]
    verify_size(checkpoint, int(CHECKPOINT["size"]))
    if args.hash:
        actual = sha256(checkpoint)
        assert actual == CHECKPOINT["sha256"], actual
        print(f"PASS sha256 {checkpoint.name}: {actual}")
    for name, (size, expected_hash) in ROUTES.items():
        path = root / "artifacts" / "bench2drive_archives" / name
        verify_size(path, size)
        if args.hash:
            actual = sha256(path)
            assert actual == expected_hash, actual
            print(f"PASS sha256 {name}: {actual}")
    routes = list((root / "data" / "Bench2Drive-Base").glob("*"))
    labels = [root / "data" / filename for filename in LABEL_FILES]
    missing_labels = [path for path in labels if not path.is_file()]
    if missing_labels:
        raise FileNotFoundError(missing_labels)
    tokenizer = root / "ckpts" / "paligemma-3b-pt-224" / "tokenizer.model"
    print(f"PASS extracted routes: {sum(path.is_dir() for path in routes)}")
    print(f"PASS required labels present: {len(labels)}")
    print(f"TOKENIZER {'PASS' if tokenizer.is_file() else 'PENDING_LICENSE'}: {tokenizer}")


if __name__ == "__main__":
    main()
