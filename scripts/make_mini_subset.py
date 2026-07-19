#!/usr/bin/env python3
"""Create a deterministic three-route Bench2Drive open-loop subset."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path


DEFAULT_SELECTION = {
    "LaneChange_Town06_Route307_Weather21": 80,
    "ParkingExit_Town12_Route922_Weather12": 20,
    "ConstructionObstacle_Town10HD_Route74_Weather22": 60,
}


def validate_sample(path: Path) -> dict:
    with path.open("rb") as stream:
        sample = pickle.load(stream)
    required = {
        "his_image_front",
        "his_image_front_left",
        "his_image_front_right",
        "his_image_back",
        "his_image_back_left",
        "his_image_back_right",
        "his_camera_id",
        "his_scenario_label",
        "fur_x",
        "fur_y",
    }
    missing = required.difference(sample)
    if missing:
        raise KeyError(f"{path.name}: missing fields {sorted(missing)}")
    image_path = Path(sample["his_image_front"].numpy().decode("utf-8"))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if sample["fur_x"].shape[-1] != 10 or sample["fur_y"].shape[-1] != 10:
        raise ValueError(f"{path.name}: expected a 10-step future trajectory")
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("exp/b2d_action/val"))
    parser.add_argument("--output", type=Path, default=Path("exp/b2d_action_mini/val"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    selected = []
    for route, step in DEFAULT_SELECTION.items():
        source = args.source / f"{route}_step{step}.pkl"
        if not source.is_file():
            raise FileNotFoundError(source)
        sample = validate_sample(source)
        destination = args.output / source.name
        shutil.copy2(source, destination)
        selected.append(
            {
                "route": route,
                "sample": destination.name,
                "camera_label": sample["his_camera_id"].numpy().decode("utf-8"),
                "scenario_label": sample["his_scenario_label"].numpy().decode("utf-8"),
            }
        )

    existing = sorted(p.name for p in args.output.glob("*.pkl"))
    expected = sorted(item["sample"] for item in selected)
    if existing != expected:
        raise RuntimeError(
            f"Output must contain exactly the selected samples; got {existing}, expected {expected}"
        )
    for item in selected:
        print(
            f"PASS {item['sample']}: camera={item['camera_label']} "
            f"scenario={item['scenario_label']}"
        )
    print(f"PASS: {len(selected)} deterministic samples across {len(DEFAULT_SELECTION)} routes")


if __name__ == "__main__":
    main()
