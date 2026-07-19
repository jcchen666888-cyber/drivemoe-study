# DriveMoE 完整训练教学

> 本章提供正式协议和命令，但本仓库没有执行完整训练。

## 1. 训练前资产

需要完整 Bench2Drive Base、完整 camera/scenario labels、PaliGemma-3b-pt-224 预训练权重、统计文件、足够磁盘和 Linux 多卡 CUDA。仅有本仓库 3 条路线只能做 data/forward/backward smoke test，不能训练有效模型。

## 2. Stage 1：Teacher-forcing Router

发布配置训练 12 epoch：VLM、Action MoE 和两个 router 联合优化；camera 分支使用真值 camera id 选择候选视角，帮助视觉 router 学会基本选择。关键值：

| 参数 | Stage 1 |
|---|---:|
| global batch | 128 |
| per-device batch | 8 |
| VLM / action LR | 5e-5 / 5e-5 |
| warmup | 200 updates |
| max grad norm | 1.0 |
| `λ_cam, λ_skill, λ_FM` | 10, 10, 1 |

```bash
export WANDB_ENTITY=your_entity
bash script/training/train_drivemoe_stage1_closed_loop.sh
```

## 3. Stage 2：Adaptive Routing

从 Stage 1 checkpoint 初始化，训练 6 epoch；视角与技能由 router 动态选择。发布代码 LR 降为 `5e-6`，loss 权重为 `5,5,1`：

```bash
torchrun --nproc_per_node=8 --standalone script/run.py \
  --config-path=../config/train/DriveMoE \
  --config-name=stage2_closed_loop \
  ckpt_path=/path/to/stage1.pt
```

注意论文补充文字称 Stage 2 “其他超参一致”，但 YAML 明确把 LR 从 `5e-5` 降到 `5e-6`。复现发布代码应以 YAML 为准。

## 4. 初始化与恢复不可混淆

- `ckpt_path`：Stage 1 → Stage 2，只加载模型并断言源 checkpoint 为 stage 1；
- `resume_checkpoint_path`：中断续训，恢复模型、两个 optimizer、scheduler、update/batch、W&B id；
- `load_pretrained_weights=True`：Stage 1 从 PaliGemma safetensors 初始化。

## 5. 两个优化器与梯度累积

代码将 VLM/视觉/router 参数和 action/proprio/MoE 参数分别交给两个 AdamW。`global_batch_size / (per_device_batch_size × world_size)` 决定累积步数；不能只改 GPU 数而忽略整除关系、学习率和更新数。

## 6. 单卡 smoke test

复制 Stage 1 配置，至少修改：工作目录指向一个小 train 子集、`per_device_batch_size=1`、`global_batch_size=1`、`num_workers=0`、`n_updates=2`、关闭 W&B/compile。目标是验证 forward、loss、backward、optimizer 和 checkpoint，不是看收敛。

## 7. 应监控什么

- total/action/camera/scenario loss；
- camera 与 action router 的类别混淆矩阵；
- expert 选择频率、top-k 概率熵与是否 collapse；
- VLM/action 学习率、grad norm、显存、data time；
- Stage 2 动态路由是否与标签完全脱钩或长期只选 normal。

运行配置审计：

```powershell
python scripts\audit_training_configs.py
```

## 8. 评估

open-loop 比较必须明确 horizon。发布 checkpoint 是 10 步，而 README 要求公平 baseline 比较使用 20 步，二者不能混用。closed-loop 要用同一 PID、CARLA 0.9.15 和 Bench2Drive leaderboard 协议，报告 DS、SR、Efficiency、Comfort 与多能力指标。
