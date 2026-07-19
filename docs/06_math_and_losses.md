# 条件流匹配与总损失推导

## 1. 条件概率路径

设噪声轨迹 `x0~N(0,I)`，真值归一化轨迹 `x1`，最小噪声 `σ=0.001`。发布代码定义：

$$
\psi_t(x_0,x_1)=\left[1-(1-\sigma)t\right]x_0+t x_1,
\quad t\in[0,1].
$$

对 `t` 求导：

$$
\frac{d\psi_t}{dt}=x_1-(1-\sigma)x_0.
$$

网络接收 `ψ_t`、图像/文本状态 `o` 与 `t`，预测速度 `v_θ`：

$$
\mathcal L_{FM}=\mathbb E\left[
\left\|v_\theta(\psi_t,o,t)-\bigl(x_1-(1-\sigma)x_0\bigr)\right\|_2^2
\right].
$$

训练的 `t` 默认由翻转的 Beta(1.5,1) 采样并缩放到 `[0,1-σ]`。

## 2. 推理 ODE

从高斯噪声 `a_0` 开始，发布配置用 10 步前向 Euler：

$$
a_{n+1}=\operatorname{clip}\left(
a_n+\Delta t\,v_\theta(a_n,o,t_n),-1,1\right),
\qquad \Delta t=0.1.
$$

VLM/proprio KV 只算一次；action tokens 随积分重复更新。

## 3. 总损失

发布代码的总损失为：

$$
\mathcal L=lambda_{FM}\mathcal L_{FM}
+\lambda_{cam}\mathcal L_{cam}^{focal}
+\lambda_{skill}\mathcal L_{skill}^{focal}.
$$

Stage 1 权重 `(1,10,10)`，Stage 2 `(1,5,5)`。这比论文中抽象交叉熵更具体，因为实现还使用类别 `α` 和 `γ=2`。

## 4. 米制误差

反归一化后，对 10×2 waypoint：

$$
MAE=\frac1{20}\sum_{t,d}|\hat a_{t,d}-a_{t,d}|,
\qquad
RMSE=\sqrt{\frac1{20}\sum_{t,d}(\hat a_{t,d}-a_{t,d})^2}.
$$

3 个样本的 RMSE 只用于检查输出尺度和可视化，不能代表论文 Avg. L2。
