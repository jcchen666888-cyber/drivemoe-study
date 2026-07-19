# Vision MoE：动态视角选择

## 1. Router

当前前视 embedding `e_t^front` 与导航目标 `g_t` 进入轻量 router：

$$
\mathbf p_t=\operatorname{softmax}
\left(R_{vision}(\mathbf e_t^{front},\mathbf g_t)\right),
\qquad \mathbf p_t\in\mathbb R^5.
$$

推理取 `argmax`，从前左、前右、后、后左、后右中选择 1 张。训练 Stage 2 使用 hard Gumbel-Softmax，使前向近似离散选择、反向仍能传梯度。

## 2. 为什么能省算力

如果全部 7 张图都过 SigLIP，视觉编码成本近似 7 份；发布推理只编码当前前视、历史前视、动态视角，共 3 份。router 本身要先获得前视特征，所以不是“零成本选择”。

## 3. 位置信息

选中候选图的 patch tokens 加上方向专属 embedding：

$$
\tilde{\mathbf z}^{v}=\mathbf z^v+\mathbf e_{view}^{v}.
$$

否则同一物体来自前左或后右的 token 在后续 Transformer 中难以区分几何方向。

## 4. 监督

论文写普通交叉熵；发布代码用带逆频率权重的 focal loss：

$$
\mathcal L_{cam}=-\alpha_y(1-p_y)^\gamma\log p_y,
\qquad \gamma=2.
$$

Stage 1 权重 10，Stage 2 权重 5。Camera 标签由未来轨迹、目标框和地图规则产生，并不等于人工逐帧主观点击。
