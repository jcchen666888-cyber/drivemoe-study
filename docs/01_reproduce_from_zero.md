# 从零完成 DriveMoE 小样本预测

## 1. 硬件与系统

推荐 NVIDIA GPU 24 GiB 以上；官方 Base bf16 在本机 RTX 4090 D 48 GiB 上验证。上游要求 Python 3.10、CUDA ≥ 12.1。Windows 可做 open-loop；closed-loop 官方生态更适合 Ubuntu。

```powershell
nvidia-smi
python --version
```

## 2. 建立隔离环境

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

本机使用 PyTorch 2.5.1+cu124；上游声明 2.5.0。二者为补丁级差异，仍要记录在实验报告中。训练还需 `wandb`、完整依赖和 Linux 多卡环境。

## 3. 获取固定代码

```powershell
git clone https://github.com/Thinklab-SJTU/DriveMoE.git _deps\DriveMoE
git -C _deps\DriveMoE checkout e39df2f610b8ebc09efaab510abd65d3ebf38e55
```

若 `git clone` 在 443 重置，可下载该 commit 的 GitHub ZIP；必须记录 commit，不能只写“main”。

## 4. 下载资产

```powershell
python scripts\download_assets.py --public
python scripts\verify_assets.py --hash
```

大文件下载使用 Range 分片脚本更稳：

```powershell
python scripts\download_hf_ranges.py --help
```

PaliGemma 仓库受 Google 条款保护。先登录 Hugging Face、接受 [模型条款](https://huggingface.co/google/paligemma-3b-pt-224)，再执行：

```powershell
huggingface-cli login
python scripts\download_assets.py --tokenizer
```

脚本只下载 tokenizer，不下载 PaliGemma 权重。

## 5. 解压三条路线

```powershell
New-Item -ItemType Directory -Force data\Bench2Drive-Base | Out-Null
powershell -ExecutionPolicy Bypass -File scripts\extract_routes.ps1
```

## 6. 生成最小窗口

```powershell
python scripts\prepare_mini_data.py
```

它选择：

- LaneChange step 80：`MERGING_HIGHWAY`；
- ParkingExit step 20：`PARKING_EXIT`；
- ConstructionObstacle step 60：`OVERTAKING`。

快速脚本逐公式复刻上游 5 帧状态窗口、10 步未来、世界系到 ego 系变换。开发时已用上游 `generate_action.py + window.py` 的 491 个窗口交叉验证。

## 7. 运行官方权重

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
python scripts\run_minimal_inference.py
```

强制单卡是为了绕开 Windows 上 `torchrun + NCCL`。脚本用 bf16 创建模型并 `mmap + assign=True` 严格加载，降低约 13B 参数模型的 CPU 峰值。

预期产物：

```text
outputs/official_mini/results.json
outputs/official_mini/drive_moe_predictions.png
```

## 8. 什么算成功

看到 `PASS: 3 official DriveMoE predictions` 仍只是第一层。还要检查 JSON 中：checkpoint 大小、3 个 route、10×2 轨迹、camera 概率和 7 类 scenario 概率；图中输入应能回查原始 JPEG。
