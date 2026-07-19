#!/usr/bin/env python3
"""Generate the three-sample DriveMoE subset directly from raw Bench2Drive routes.

The coordinate transforms and window semantics mirror the pinned upstream
``generate_action.py`` and ``window.py`` implementation, while avoiding the
full-dataset Ray pass for a three-sample teaching run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from make_mini_subset import DEFAULT_SELECTION


def load_annotation(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def world_to_ego(theta: float, ego_x: float, ego_y: float, x: float, y: float):
    rotation = np.asarray(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    return rotation @ np.asarray([x - ego_x, y - ego_y], dtype=np.float64)


def generate(route: Path, step: int, camera_labels: dict, scenario_labels: dict) -> dict:
    history_steps = [max(step + offset, 0) for offset in range(-4, 1)]
    future_steps = list(range(step, step + 10))
    needed = sorted(set(history_steps + future_steps))
    annotations = {
        index: load_annotation(route / "anno" / f"{index:05d}.json.gz")
        for index in needed
    }
    current = annotations[step]
    ego_x = current["bounding_boxes"][0]["location"][0]
    ego_y = current["bounding_boxes"][0]["location"][1]
    headings = [
        annotations[index]["theta"] - np.pi / 2
        if not np.isnan(annotations[index]["theta"])
        else 0.0
        for index in history_steps
    ]
    ego_theta = headings[-1]

    command_xy = [
        world_to_ego(
            ego_theta,
            ego_x,
            ego_y,
            annotations[index]["x_command_far"],
            annotations[index]["y_command_far"],
        )
        for index in history_steps
    ]
    future_xy = [
        world_to_ego(
            ego_theta,
            ego_x,
            ego_y,
            annotations[index]["bounding_boxes"][0]["location"][0],
            annotations[index]["bounding_boxes"][0]["location"][1],
        )
        for index in future_steps
    ]
    camera_root = route / "camera"
    camera_dirs = {
        "front": "rgb_front",
        "front_left": "rgb_front_left",
        "front_right": "rgb_front_right",
        "back": "rgb_back",
        "back_left": "rgb_back_left",
        "back_right": "rgb_back_right",
    }
    result = {
        "his_theta": np.asarray([value - ego_theta for value in headings], dtype=np.float32),
        "x_command_far": np.asarray([value[0] for value in command_xy], dtype=np.float32),
        "y_command_far": np.asarray([value[1] for value in command_xy], dtype=np.float32),
        "fur_x": np.asarray([[value[0] for value in future_xy]], dtype=np.float32),
        "fur_y": np.asarray([[value[1] for value in future_xy]], dtype=np.float32),
        "his_speed": np.asarray([annotations[index]["speed"] for index in history_steps], dtype=np.float32),
        "his_acceleration": np.asarray(
            [annotations[index]["acceleration"] for index in history_steps], dtype=np.float32
        ),
        "his_angular_velocity": np.asarray(
            [annotations[index]["angular_velocity"] for index in history_steps], dtype=np.float32
        ),
        "his_camera_id": tf.constant(camera_labels[str(step)]),
        "his_scenario_label": tf.constant(scenario_labels[str(step)]),
        "episode_id": 0,
        "sample_id": step,
        "unique_id": f"mini{step:04d}",
    }
    for name, directory in camera_dirs.items():
        image = (camera_root / directory / f"{step:05d}.jpg").resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        # The pinned dataset loader extracts the frame id with split("/").
        # Forward slashes remain valid on Windows and keep the pickle portable
        # across the upstream Windows/Linux data path implementations.
        result[f"his_image_{name}"] = tf.constant(str(image).replace("\\", "/"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/Bench2Drive-Base"))
    parser.add_argument("--camera-labels", type=Path, default=Path("data/labels/camera_labels"))
    parser.add_argument("--scenario-labels", type=Path, default=Path("data/labels/scenario_labels"))
    parser.add_argument("--output", type=Path, default=Path("exp/b2d_action_mini_direct/val"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    expected_names = []
    for route_name, step in DEFAULT_SELECTION.items():
        route = args.dataset / route_name
        camera_path = args.camera_labels / f"{route_name}.json"
        scenario_path = args.scenario_labels / f"{route_name}.json"
        if not route.is_dir():
            raise FileNotFoundError(route)
        camera_labels = json.loads(camera_path.read_text(encoding="utf-8"))
        scenario_labels = json.loads(scenario_path.read_text(encoding="utf-8"))
        sample = generate(route.resolve(), step, camera_labels, scenario_labels)
        destination = args.output / f"{route_name}_step{step}.pkl"
        with destination.open("wb") as stream:
            pickle.dump(sample, stream)
        expected_names.append(destination.name)
        print(
            f"PASS {destination.name}: camera={camera_labels[str(step)]} "
            f"scenario={scenario_labels[str(step)]}"
        )

    actual_names = sorted(path.name for path in args.output.glob("*.pkl"))
    if actual_names != sorted(expected_names):
        raise RuntimeError(f"Unexpected files in {args.output}: {actual_names}")
    print(f"PASS: generated {len(expected_names)} direct mini samples")


if __name__ == "__main__":
    main()
