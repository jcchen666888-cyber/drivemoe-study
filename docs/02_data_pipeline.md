# Bench2Drive 数据流水线

## 1. 一条训练样本包含什么

DriveMoE 使用 5 帧历史状态与图像/标签，预测 10 个未来二维 waypoint。状态每帧 10 维：速度 1、加速度 3、角速度 3、相对航向 1、远导航点 2。

输入图像有 7 张：当前前视、前一帧前视、当前前左/前右/后/后左/后右。Vision Router 在后 5 个候选中选 1 个。

## 2. 世界坐标到 ego 坐标

当前自车位置为 `(x_e,y_e)`，航向为 `θ`，世界点 `p=(x,y)`：

$$
\begin{bmatrix}x'\\y'\end{bmatrix}
=
\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x-x_e\\y-y_e\end{bmatrix}.
$$

同一变换用于远导航点和未来轨迹。`window.py` 还会先把 CARLA 的 `theta` 减去 `π/2`。

## 3. 标签

Camera Router 五类为 front-left、front-right、back、back-left、back-right；原始 `NULL` 映射为 back（id 2）。Action Router 七类为 merging、parking-exit、overtaking、emergency-brake、giveway、traffic-sign、normal。

论文正文常写 5 或 6 类技能；发布代码明确加入 `PARKING_EXIT` 与 `NORMAL`，因此是 7 类。

## 4. 百分位归一化

对任意变量 `z`，代码用 1%/99% 统计量映射到：

$$
\tilde z=2\frac{z-z_{1\%}}{z_{99\%}-z_{1\%}}-1,
\qquad
z=\frac{\tilde z+1}{2}(z_{99\%}-z_{1\%})+z_{1\%}.
$$

checkpoint、训练和推理必须使用同一 `b2d_statistics.json`。

## 5. 上游完整预处理与最小预处理

完整流程先将每个 route 聚合为 episode，再用 Ray/TensorFlow 写每个窗口。它默认删除中间 `train.pkl/val.pkl`。本仓库的快速脚本只读 3 个指定 step，保持相同索引、坐标和 Tensor 字段；最大交叉验证误差为 `1.907e-6`。

退出码 0 不代表数据正确：相对路径解析到错误工作目录时，上游脚本会写两个空 pickle 并正常退出。必须断言 episode/sample 数量。
