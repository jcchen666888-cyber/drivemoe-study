#!/usr/bin/env python3
"""Download the minimal official DriveMoE study assets with verification."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


CHECKPOINT = {
    "repo_id": "rethinklab/DriveMoE",
    "filename": "DriveMoE_base_bf16.pt",
    "size": 13_520_189_565,
    "sha256": "af97a1d8f415240320d9f0ac0f8a57cb15ba065cce5d8fbb41f2a90d8f6d6d4f",
}

ROUTES = {
    "LaneChange_Town06_Route307_Weather21.tar.gz": (
        132_357_666,
        "8d62546eb29d8f0b8dc423ea41932241a92f0637f124b6591423f6a38df42aad",
    ),
    "ParkingExit_Town12_Route922_Weather12.tar.gz": (
        153_840_673,
        "1d70a167a904bb3e2e6129f937cb740d2cf67f0ecf96968fd864f3447b19302f",
    ),
    "ConstructionObstacle_Town10HD_Route74_Weather22.tar.gz": (
        235_784_985,
        "c553b45ccd07319b2a620af2cb17a88e74c75be406c50559f8f819ddd1030ebb",
    ),
}

LABEL_FILES = tuple(
    f"labels/{label_type}/{Path(route).stem.removesuffix('.tar')}.json"
    for route in ROUTES
    for label_type in ("camera_labels", "scenario_labels")
)

TOKENIZER_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_size(path: Path, expected: int) -> None:
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"size mismatch for {path}: {actual} != {expected}")
    print(f"PASS size {path.name}: {actual:,} bytes")


def download_public(root: Path) -> None:
    checkpoint_dir = root / "ckpts"
    archive_dir = root / "artifacts" / "bench2drive_archives"
    data_dir = root / "data"
    for directory in (checkpoint_dir, archive_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_dir / CHECKPOINT["filename"]
    if not checkpoint.exists():
        checkpoint = Path(
            hf_hub_download(
                repo_id=CHECKPOINT["repo_id"],
                filename=CHECKPOINT["filename"],
                local_dir=checkpoint_dir,
            )
        )
    verify_size(checkpoint, int(CHECKPOINT["size"]))
    actual_hash = sha256(checkpoint)
    if actual_hash != CHECKPOINT["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {checkpoint}: {actual_hash}")
    print(f"PASS sha256 {checkpoint.name}: {actual_hash}")

    for filename in LABEL_FILES:
        path = hf_hub_download(
            repo_id="rethinklab/DriveMoE",
            filename=filename,
            local_dir=data_dir,
        )
        print(f"PASS label: {path}")

    for filename, (expected_size, expected_hash) in ROUTES.items():
        route = Path(
            hf_hub_download(
                repo_id="rethinklab/Bench2Drive",
                repo_type="dataset",
                filename=filename,
                local_dir=archive_dir,
            )
        )
        verify_size(route, expected_size)
        actual_hash = sha256(route)
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch for {route}: {actual_hash}")
        print(f"PASS sha256 {route.name}: {actual_hash}")


def download_tokenizer(root: Path) -> None:
    tokenizer_dir = root / "ckpts" / "paligemma-3b-pt-224"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES:
        path = hf_hub_download(
            repo_id="google/paligemma-3b-pt-224",
            filename=filename,
            local_dir=tokenizer_dir,
        )
        print(f"PASS tokenizer asset: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--public", action="store_true", help="download checkpoint, labels and routes")
    parser.add_argument("--tokenizer", action="store_true", help="download gated PaliGemma tokenizer files")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.public and not args.tokenizer:
        parser.error("select --public and/or --tokenizer")
    if args.public:
        download_public(root)
    if args.tokenizer:
        download_tokenizer(root)


if __name__ == "__main__":
    main()
