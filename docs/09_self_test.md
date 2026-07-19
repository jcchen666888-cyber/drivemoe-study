# DriveMoE 详细自测

## A. 概念题

1. Drive-π0 与 DriveMoE 的差别是什么？
2. 为什么推理输入有 7 张图但只编码 3 张？
3. Vision Router 与 Action Router 分别在哪个计算阶段工作？
4. trajectory-level routing 为什么对 10 个 token 求均值？
5. shared expert 与 top-3 skill experts 如何组合？
6. open-loop 为什么不需要 CARLA？
7. 10 步 checkpoint 为什么不能和 20 步 baseline 直接比较？

<details><summary>答案</summary>

1. DriveMoE 在 Drive-π0 上增加动态视角选择和技能专家 Action MoE。
2. 两张固定前视先编码，router 只从其余五张选一张再编码。
3. Vision Router 在候选视觉 backbone 前；Action Router 在最后四个 action MoE 层的 FFN 前。
4. 让整条轨迹共享技能，而不是 waypoint 各自选择互相冲突的技能。
5. top-3 概率重归一化后加权求和，再加共享 expert 输出。
6. 输入来自记录数据，模型只做离线前向；CARLA 用于闭环反馈。
7. 时间覆盖、误差累计和指标分母不同。

</details>

## B. 手算题

### B1. 归一化

`z_1=0,z_99=10,z=7.5`，归一化结果为多少？答案：`0.5`。

### B2. Top-k

router 概率 `[0.1,0.2,0.3,0.15,0.05,0.12,0.08]`，top-3 为哪几个，重归一化权重是多少？答案：索引 2、1、3；总和 0.65，权重约 0.4615、0.3077、0.2308。

### B3. Flow

`σ=0.001,x0=2,x1=5,t=0.4`：`ψ_t=(1-0.999×0.4)×2+0.4×5=3.2008`；目标速度 `5-0.999×2=3.002`。

### B4. 全局 batch

2 GPU、每卡 8、global 128，需要累积 8 次；若不整除应修改配置而不是取整隐瞒。

## C. 自动自测

```powershell
python demo\minimal_drivemoe_loop.py --self-test
python scripts\verify_assets.py --hash
python scripts\prepare_mini_data.py
python scripts\audit_training_configs.py
python -m py_compile demo\minimal_drivemoe_loop.py scripts\*.py
```

教学 Demo 预期 `PASS: 5/5`。它不是神经网络重实现，官方推理才是 checkpoint 证据。

## D. 真实复现清单

- [ ] 权重与 3 个 archive 的字节数、SHA256 全通过；
- [ ] 三路线 15,109 个文件可读；
- [ ] 3 个 label 的帧数与 annotation 对齐；
- [ ] 快速转换与上游误差 < `2e-6`；
- [ ] tokenizer 来自已授权 PaliGemma repo；
- [ ] strict load 无差异；
- [ ] GPU bf16 前向产生 3×10×2；
- [ ] router 概率和轨迹 JSON 可回查；
- [ ] 能说明本次没有训练、没有 CARLA、没有论文指标。
