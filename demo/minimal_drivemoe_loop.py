#!/usr/bin/env python3
"""Transparent NumPy teaching proxy for DriveMoE's two routers and flow ODE."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CAMERAS = ["front_left", "front_right", "back", "back_left", "back_right"]
SKILLS = ["merging", "parking_exit", "overtaking", "emergency_brake", "giveway", "traffic_sign", "normal"]


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / exp.sum()


def camera_router(goal_xy: np.ndarray) -> np.ndarray:
    x, y = goal_xy
    logits = np.asarray([y, -y, -0.3 * x, 0.5 * y - 0.2 * x, -0.5 * y - 0.2 * x])
    return softmax(logits)


def skill_router(scene: np.ndarray) -> np.ndarray:
    weights = np.asarray(
        [
            [1.7, 0.2, -0.2],
            [-0.4, -0.6, 1.8],
            [0.4, 1.8, -0.2],
            [-1.1, 0.3, 1.4],
            [0.1, -0.9, 1.0],
            [0.0, 0.0, 0.7],
            [0.1, 0.1, 0.1],
        ]
    )
    return softmax(weights @ scene)


def expert_velocity(index: int, action: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, len(action))
    targets = np.zeros_like(action)
    targets[:, 0] = progress * max(goal_xy[0], 2.0)
    lateral = [0.8, -1.8, 1.3, 0.0, -0.5, 0.0, 0.0][index]
    targets[:, 1] = lateral * progress**2
    gains = [1.05, 0.7, 1.15, 0.45, 0.8, 0.9, 0.95]
    return gains[index] * (targets - action)


def integrate(goal_xy: np.ndarray, scene: np.ndarray, steps: int = 10):
    probabilities = skill_router(scene)
    top3 = np.argsort(probabilities)[-3:][::-1]
    weights = probabilities[top3]
    weights = weights / weights.sum()
    action = np.random.default_rng(42).normal(0, 0.15, size=(10, 2))
    for _ in range(steps):
        velocity = sum(
            weight * expert_velocity(int(expert), action, goal_xy)
            for weight, expert in zip(weights, top3)
        )
        action = action + velocity / steps
    return action, probabilities, top3, weights


def self_test() -> None:
    passed = 0
    p = softmax(np.asarray([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1.0) and np.argmax(p) == 2
    passed += 1
    cam = camera_router(np.asarray([8.0, 3.0]))
    assert np.isclose(cam.sum(), 1.0) and len(cam) == 5
    passed += 1
    skill = skill_router(np.asarray([1.0, 0.0, 0.0]))
    assert np.argmax(skill) == 0
    passed += 1
    action, _, top3, weights = integrate(np.asarray([8.0, 2.0]), np.asarray([0.0, 1.0, 0.0]))
    assert action.shape == (10, 2) and len(top3) == 3 and np.isclose(weights.sum(), 1.0)
    passed += 1
    x0 = np.asarray([1.0, -1.0])
    x1 = np.asarray([3.0, 2.0])
    sigma = 0.001
    t = 0.4
    psi = (1 - (1 - sigma) * t) * x0 + t * x1
    derivative = x1 - (1 - sigma) * x0
    eps = 1e-5
    numerical = (
        (1 - (1 - sigma) * (t + eps)) * x0 + (t + eps) * x1 - psi
    ) / eps
    assert np.allclose(derivative, numerical, atol=1e-8)
    passed += 1
    print(f"PASS: {passed}/5 mathematical and routing self-tests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/minimal_drivemoe_loop.png"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
    goal = np.asarray([8.0, 2.5])
    scene = np.asarray([0.2, 1.0, 0.1])
    trajectory, skills, top3, weights = integrate(goal, scene)
    cameras = camera_router(goal)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(CAMERAS, cameras)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title(f"Vision router: {CAMERAS[int(np.argmax(cameras))]}")
    axes[0].set_ylabel("probability")
    axes[1].bar(SKILLS, skills)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Action router top-3: " + ", ".join(SKILLS[int(i)] for i in top3))
    axes[2].plot(trajectory[:, 0], trajectory[:, 1], "o-")
    axes[2].scatter([0], [0], marker="^", s=90, c="black")
    axes[2].scatter([goal[0]], [goal[1]], marker="*", s=150, c="gold")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("x forward")
    axes[2].set_ylabel("y lateral")
    axes[2].set_title("10-step Euler flow trajectory")
    fig.suptitle("Minimal DriveMoE teaching loop (not the official neural network)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"camera={CAMERAS[int(np.argmax(cameras))]}")
    print("action_top3=" + ",".join(SKILLS[int(i)] for i in top3))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
