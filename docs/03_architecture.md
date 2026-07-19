# Drive-π0 与 DriveMoE 总架构

## 1. Drive-π0 基线

Drive-π0 把 PaliGemma 的 SigLIP 视觉塔和 Gemma 语言模块扩展为 VLA：图像/固定文本构成 VLM tokens，5 帧自车状态变成 proprio tokens，10 个带噪轨迹点变成 action tokens。联合 Transformer 预测条件流速度。

## 2. DriveMoE 加了什么

DriveMoE 保留 Drive-π0 的连续轨迹生成，在感知和动作两端加入稀疏条件计算：

1. Vision Router：从 5 个候选视角选 top-1；
2. 视角位置 embedding：告诉 VLM 选中的是哪一方向；
3. Action Router：根据整条 action token 表示选择技能专家；
4. 4 个 MoE 层中使用 top-3 技能专家与共享专家。

## 3. 张量形状（发布配置）

| 张量 | 形状/维度 |
|---|---|
| 图像 | `B × 7 × 3 × 224 × 224` |
| 单图 patch tokens | 256；三张有效图共 768 |
| VLM hidden | 2048 |
| proprio | `B × 5 × 10`，hidden 1024 |
| action | `B × 10 × 2`，hidden 1024 |
| camera logits | `B × 5` |
| action logits | `B × 7` |
| 最大序列 | 788 + 5 + 10 |

`vision.config.num_image_tokens=768` 表示最终 3 张图的 token 总数，不是每张 768。

## 4. 训练与推理路径差别

Stage 1 使用真值 camera id 选择候选图，但仍预测 camera logits；Stage 2 动态选择。推理先编码两张固定前视，router 后只编码一个候选视角。Action 端先缓存 VLM/proprio KV，再做 10 次 Euler 流积分。
