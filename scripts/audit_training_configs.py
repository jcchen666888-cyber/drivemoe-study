#!/usr/bin/env python3
"""Check the pinned release configs against the documented two-stage protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


EXPECTED = {
    "stage1_closed_loop.yaml": {
        "stage": 1,
        "n_epochs": 12,
        "action_lr": 5e-5,
        "vlm_lr": 5e-5,
        "camera": 10.0,
        "router": 10.0,
        "action": 1.0,
    },
    "stage2_closed_loop.yaml": {
        "stage": 2,
        "n_epochs": 6,
        "action_lr": 5e-6,
        "vlm_lr": 5e-6,
        "camera": 5.0,
        "router": 5.0,
        "action": 1.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("_deps/DriveMoE"))
    args = parser.parse_args()
    config_root = args.upstream / "config/train/DriveMoE"
    for name, expected in EXPECTED.items():
        cfg = OmegaConf.load(config_root / name)
        actual = {
            "stage": cfg.stage,
            "n_epochs": cfg.n_epochs,
            "action_lr": cfg.action_lr,
            "vlm_lr": cfg.vlm_lr,
            "camera": cfg.criterion.camera_router_weight,
            "router": cfg.criterion.action_router_weight,
            "action": cfg.criterion.action_weight,
        }
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")
        if cfg.joint.config.num_skill_experts != 7 or cfg.joint.config.num_experts_per_tok != 3:
            raise AssertionError(f"{name}: released code no longer uses 7 skill experts / top-3")
        print(f"PASS {name}: {actual}")
    print("PASS: 2/2 DriveMoE training configs match the documented release")


if __name__ == "__main__":
    main()
