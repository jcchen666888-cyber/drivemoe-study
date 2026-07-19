#!/usr/bin/env python3
"""Run official DriveMoE Base on three deterministic Bench2Drive samples."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


CAMERA_NAMES = ["front_left", "front_right", "back", "back_left", "back_right"]
SCENARIO_NAMES = [
    "merging",
    "parking_exit",
    "overtaking",
    "emergency_brake",
    "giveway",
    "traffic_sign",
    "normal",
]


def configure(root: Path, upstream: Path, tokenizer: Path, checkpoint: Path):
    cfg = OmegaConf.load(upstream / "config/eval/DriveMoE/open_loop.yaml")
    cfg.pretrained_model_path = str(tokenizer)
    cfg.checkpoint_path = str(checkpoint)
    cfg.data.work_dir = str(root / "exp/b2d_action_mini_direct")
    cfg.data.statistics_path = str(upstream / "config/statistics/b2d_statistics.json")
    cfg.device_batch_size = 1
    cfg.num_workers = 0
    cfg.use_bf16 = True
    cfg.gpu_id = 0
    return cfg


def build_model(cfg, checkpoint: Path):
    from src.model.DriveMoE.drivemoe import DriveMoE

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = DriveMoE(cfg)
    finally:
        torch.set_default_dtype(previous_dtype)

    data = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if data.get("stage") != 2:
        raise ValueError(f"Expected stage-2 checkpoint, got {data.get('stage')}")
    for name in ("horizon_steps", "cond_steps"):
        if data.get(name) != cfg[name]:
            raise ValueError(f"Checkpoint {name}={data.get(name)}, config {name}={cfg[name]}")
    state = {key.replace("_orig_mod.", ""): value for key, value in data["model"].items()}
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    del state, data
    gc.collect()
    model.freeze_all_weights()
    model.to(dtype=torch.bfloat16, device="cuda:0")
    model.eval()
    return model


def build_inputs(batch, processor, model, device: torch.device):
    image_keys = (
        "image_front",
        "image_front_time",
        "image_front_left",
        "image_front_right",
        "image_back",
        "image_back_left",
        "image_back_right",
    )
    images = [einops.rearrange(batch[key], "B H W C -> B C H W").unsqueeze(1) for key in image_keys]
    image_tensor = torch.cat(images, dim=1)
    model_inputs = processor(text=batch["language_instruction"], images=image_tensor)
    causal_mask, vlm_positions, state_positions, action_positions = (
        model.build_causal_mask_and_position_ids(
            model_inputs["attention_mask"], torch.bfloat16
        )
    )
    image_text_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)
    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(torch.bfloat16),
        "image_text_proprio_mask": image_text_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_positions,
        "proprio_position_ids": state_positions,
        "action_position_ids": action_positions,
        "proprios": batch["state"].to(torch.bfloat16),
        "waypoints": batch["waypoints"].to(torch.bfloat16),
    }
    return {name: value.to(device) for name, value in inputs.items()}


def render(results: list[dict], output: Path, root: Path) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 4 * len(results)))
    if len(results) == 1:
        axes = np.asarray([axes])
    for row, result in enumerate(results):
        image = plt.imread(root / result["front_image"])
        axes[row, 0].imshow(image)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(
            f"{result['route']}\n"
            f"camera: {result['camera_true']} -> {result['camera_pred']} | "
            f"skill: {result['scenario_true']} -> {result['scenario_pred']}"
        )
        pred = np.asarray(result["trajectory_pred_m"])
        truth = np.asarray(result["trajectory_gt_m"])
        axes[row, 1].plot(truth[:, 0], truth[:, 1], "o-", label="ground truth")
        axes[row, 1].plot(pred[:, 0], pred[:, 1], "o-", label="DriveMoE")
        axes[row, 1].scatter([0], [0], marker="^", s=90, c="black", label="ego")
        axes[row, 1].set_aspect("equal", adjustable="datalim")
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].set_xlabel("x forward (m)")
        axes[row, 1].set_ylabel("y lateral (m)")
        axes[row, 1].legend()
        axes[row, 1].set_title(f"10-step open-loop trajectory | L2 RMSE={result['rmse_m']:.3f} m")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    upstream = (args.upstream or root / "_deps/DriveMoE").resolve()
    tokenizer = (args.tokenizer or root / "ckpts/paligemma-3b-pt-224").resolve()
    checkpoint = (args.checkpoint or root / "ckpts/DriveMoE_base_bf16.pt").resolve()
    output = (args.output or root / "outputs/official_mini").resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not tokenizer.joinpath("tokenizer.model").is_file():
        raise FileNotFoundError(
            f"PaliGemma tokenizer not found at {tokenizer}. Accept its Hugging Face license "
            "and run scripts/download_assets.py --tokenizer."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(upstream))

    from src.agent.dataset import Bench2DriveDataset
    from src.data.utils.normalization import Normalize
    from src.model.DrivePi0.processing import VLAProcessor

    cfg = configure(root, upstream, tokenizer, checkpoint)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, padding_side="right")
    processor = VLAProcessor(
        tokenizer_obj,
        num_image_tokens=cfg.vision.config.num_image_tokens,
        max_seq_len=cfg.max_seq_len,
        tokenizer_padding=cfg.tokenizer_padding,
    )
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official DriveMoE Base checkpoint")

    started = time.perf_counter()
    model = build_model(cfg, checkpoint)
    load_seconds = time.perf_counter() - started
    dataset = Bench2DriveDataset(cfg.data).dataset
    dataset.list_data_dict = sorted(dataset.list_data_dict)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    normalizer = Normalize.get_instance(cfg.data.statistics_path)
    results = []

    with torch.inference_mode():
        for index, batch in enumerate(loader):
            torch.manual_seed(cfg.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed + index)
            inputs = build_inputs(batch, processor, model, device)
            before = time.perf_counter()
            pred, camera_logits, action_logits_list = model.infer_action(**inputs)
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - before

            truth = batch["trajectory"].squeeze(1).float()
            pred_x, pred_y = normalizer.infer_traj(pred.float().cpu())
            true_x, true_y = normalizer.infer_traj(truth.cpu())
            pred_xy = torch.stack((pred_x, pred_y), dim=1).float().numpy()
            true_xy = torch.stack((true_x, true_y), dim=1).float().numpy()
            camera_probs = torch.softmax(camera_logits.float(), dim=-1)[0].cpu()
            scenario_probs = torch.softmax(action_logits_list[-1].float(), dim=-1)[0].cpu()
            camera_pred = int(camera_probs.argmax())
            scenario_pred = int(scenario_probs.argmax())
            source_name = dataset.list_data_dict[index]
            route = source_name.rsplit("_step", 1)[0]
            with (Path(dataset.data_path) / source_name).open("rb") as stream:
                import pickle

                source = pickle.load(stream)
            front_image_absolute = Path(source["his_image_front"].numpy().decode("utf-8")).resolve()
            front_image = front_image_absolute.relative_to(root).as_posix()
            record = {
                "route": route,
                "sample": source_name,
                "front_image": front_image,
                "camera_true": CAMERA_NAMES[int(batch["cam_id"][0])],
                "camera_pred": CAMERA_NAMES[camera_pred],
                "camera_probabilities": dict(zip(CAMERA_NAMES, camera_probs.tolist())),
                "scenario_true": SCENARIO_NAMES[int(batch["scenario_id"][0])],
                "scenario_pred": SCENARIO_NAMES[scenario_pred],
                "scenario_probabilities": dict(zip(SCENARIO_NAMES, scenario_probs.tolist())),
                "action_top3": [
                    SCENARIO_NAMES[i]
                    for i in torch.topk(scenario_probs, 3).indices.tolist()
                ],
                "trajectory_pred_m": pred_xy.tolist(),
                "trajectory_gt_m": true_xy.tolist(),
                "mae_m": float(np.abs(pred_xy - true_xy).mean()),
                "rmse_m": float(np.sqrt(np.square(pred_xy - true_xy).mean())),
                "inference_seconds": inference_seconds,
            }
            results.append(record)
            print(
                f"PASS {source_name}: camera={record['camera_pred']} "
                f"scenario_top3={record['action_top3']} rmse={record['rmse_m']:.3f}m "
                f"time={inference_seconds:.2f}s"
            )

    summary = {
        "upstream_commit": "e39df2f610b8ebc09efaab510abd65d3ebf38e55",
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "dtype": "bfloat16",
        "inference_steps": int(cfg.num_inference_steps),
        "horizon_steps": int(cfg.horizon_steps),
        "model_load_seconds": load_seconds,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "samples": results,
    }
    result_path = output / "results.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    render(results, output / "drive_moe_predictions.png", root)
    print(f"PASS: {len(results)} official DriveMoE predictions")
    print(f"RESULTS {result_path}")


if __name__ == "__main__":
    main()
