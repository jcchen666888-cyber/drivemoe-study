# Action MoE：技能专家路由

## 1. Trajectory-level Router

发布实现先在 10 个 action tokens 上求均值，再线性投影：

$$
\bar{\mathbf h}=\frac1T\sum_{t=1}^{T}\mathbf h_t,
\qquad
\mathbf r=\operatorname{softmax}(W_r\bar{\mathbf h}/|\tau|).
$$

因此一次轨迹共享同一组 expert，而不是每个 waypoint 各选一组。

## 2. Sparse top-3

设选择集合 `S=TopK(r,3)`，重新归一化：

$$
\hat r_k=\frac{r_k}{\sum_{j\in S}r_j},\quad k\in S.
$$

MoE 输出为：

$$
\mathbf y=\sum_{k\in S}\hat r_k E_k(\mathbf h)
+\sum_{m=1}^{M}E_m^{shared}(\mathbf h).
$$

发布配置是 7 个 skill experts、top-3、1 个 shared expert、最后 4 个 decoder 层为 MoE。

## 3. 技能标签

七类是 merging、parking-exit、overtaking、emergency-brake、giveway、traffic-sign、normal。发布 focal loss 同样按类别计数做逆频率加权。

## 4. 论文与代码差异

- 论文主文一处写 top-1/top-2；补充材料与发布配置为 top-3。
- 补充文字写 6 个 non-shared experts；发布代码 `num_skill_experts=7`。
- 因此复现以固定 commit 展开后的配置和 checkpoint 形状为准，论文用于解释设计。
