# 范围、空间与验收

## 1. 本次到底复现什么

实际执行的是 open-loop 最小闭环：真实相机图像与状态进入发布版 DriveMoE Base，得到 10 步轨迹、Vision Router 选择和 Action Router top-3。它不启动 CARLA，也不执行反向传播。

完整 closed-loop 才需要 CARLA 0.9.15、Bench2Drive leaderboard、路线 XML、场景 JSON 和多进程模拟。CARLA 不是理解模型或完成小样本预测的前置条件。

## 2. 为什么不下载“所有大模型”

发布的 `DriveMoE_base_bf16.pt` 是完整严格 `state_dict`，已包含视觉塔、VLM、状态/动作模块、两个 router 和专家权重。推理只额外调用 PaliGemma tokenizer，因此：

- 下载 DriveMoE Base bf16：12.592 GiB；
- PaliGemma 只取 tokenizer：约 21 MiB；
- 不下载 PaliGemma 3B 的约 10.9 GiB safetensors；
- 不下载 Driveπ0、DriveMoE fp32 或重复 checkpoint。

## 3. 空间合同

| 资产 | 压缩/文件大小 |
|---|---:|
| DriveMoE Base bf16 | 12.592 GiB |
| 3 条 Bench2Drive 路线归档 | 0.486 GiB |
| 3 条路线解压 | 0.692 GiB |
| 标签与 tokenizer | < 0.03 GiB |
| 独立 Conda 环境 | 约 5.2 GiB |
| 源码、缓存、结果与余量 | 约 3–8 GiB |

预计干净安装约 20–28 GiB，建议预留 40 GiB；本任务授权上限为 50 GiB。完整 Bench2Drive 与多份训练 checkpoint 需要数百 GiB，不属于本次范围。

## 4. 成功判据

- checkpoint 字节数与 SHA256 均正确；
- 3 个路线归档逐个 SHA256 正确；
- 6 个 camera/scenario label 文件存在且长度与帧数一致；
- 官方转换得到 491 个窗口；
- 快速转换器与上游结果数值误差小于 `2e-6`；
- 模型严格加载，无 missing/unexpected keys；
- CUDA bf16 前向执行，3 个样本均输出 10×2 轨迹；
- 保存 router 概率、top-3、米制轨迹和可视化；
- 教学自测、配置审计、链接与脚本语法全部通过。
