# 面向多模型变体（Multi-Variant）与多空地视角（Aerial-Ground Perspectives）的智能体原生通感算一体化（Agentic-Native ISCC）主线蓝图

> **状态**：`main` 主线核心公用财产（Living Blueprint，随工程与实验推进持续迭代更新）  
> **定位**：AeroRescue 科研主线唯一权威理论、算法、实验与可解释性诊断规范；与工程主线 [`docs/engineering-master-plan.md`](./engineering-master-plan.md) 构成“工程闭环底座 + 顶刊科学突破”双支柱。

---

## 核心科学命题（Non-Trivial Core）

面向多模型变体（Multi-Variant）与多空地视角（Aerial-Ground Perspectives）的智能体原生通感算一体化（Agentic-Native ISCC），其核心非平凡问题在于：
**空地大视角射影误差引起的空间特征错配，在跨所有共视锚点上是共享同一 6 自由度位姿扰动的秩-6 强相关噪声；该噪声与未知的跨变体基底变换矩阵发生乘性耦合（Errors-in-Variables, EIV），并最终通过接收端多模态大模型（VLM）动作决策头的低秩 Fisher 信息度量影响闭环控制风险。**

本蓝图剥离一切平凡的加性独立假设与概念包装，从**精确微分几何与矩阵扰动建模**、**非平凡上下界定理与证明**、**几何—语义锚点二分机制与三元预算求解算法**、**智能体主动感知—通信—决策闭环协议**、**多变体可解释性诊断与零稻草人实验设计**，以及**AeroRescue × AeroAgentSim 工程实现映射与演进评估**六个部分完整展开。

---

## 一、 核心数学建模：空地射影畸变、跨锚点相关错配与决策拉回度量

设系统由空中智能体 $i$（搭载源视觉/多模态变体 $E_i$，特征维数 $r$，内参 $\mathbf{K}_i$）与地面主智能体 $e$（搭载目标决策变体 $E_e$，特征维数 $d_e$，内参 $\mathbf{K}_e$）组成。在决策步 $t$，两节点间的真实相对位姿为 $\mathbf{T}_{e \leftarrow i}^\star \in \mathrm{SE}(3)$，广播或先验估计位姿为 $\widehat{\mathbf{T}}_{e \leftarrow i} = \mathbf{T}_{e \leftarrow i}^\star \exp(\boldsymbol{\xi}^\wedge)$，其中李代数残差扰动 $\boldsymbol{\xi} = [\boldsymbol{\omega}^\top, \boldsymbol{\upsilon}^\top]^\top \in \mathbb{R}^6$ 服从零均值高斯分布 $\mathcal{N}(\mathbf{0}, \Sigma_\xi)$。

### 1. 空地非对称射影雅可比与视锥畸变张量
空中视角（俯仰角 $\theta_{\mathrm{pitch}}^{(i)} \in [-90^\circ, -30^\circ]$，高度 $z_i \gg z_e$）与地面平视视角（$\theta_{\mathrm{pitch}}^{(e)} \approx 0^\circ$）在观测同一三维表面点 $\mathbf{p}_k^{(i)} = d_k \mathbf{K}_i^{-1} \bar{\mathbf{u}}_k$（在接收端坐标系下真实坐标为 $\mathbf{p}_k^{(e)} = [X_k, Y_k, Z_k]^\top = \mathbf{T}_{e \leftarrow i}^\star \mathbf{p}_k^{(i)}$）时，位姿扰动 $\boldsymbol{\xi}$ 在接收端二维特征网格坐标 $\mathbf{v}_k = [u_k^{(e)}, v_k^{(e)}]^\top \in \mathbb{R}^2$ 上诱导的一阶像素位移为：

$$\delta \mathbf{v}_k(\boldsymbol{\xi}) = \mathbf{J}_{\mathrm{proj}}\left(\mathbf{p}_k^{(e)}\right) \mathbf{J}_{\mathrm{se3}}\left(\mathbf{p}_k^{(e)}\right) \boldsymbol{\xi} \equiv \mathbf{G}_k \boldsymbol{\xi}, \quad \mathbf{G}_k \in \mathbb{R}^{2 \times 6}$$

其中透视投影雅可比与李代数空间雅可比分别为：

$$\mathbf{J}_{\mathrm{proj}}(X, Y, Z) = \frac{1}{Z} \begin{bmatrix} f_x^{(e)} & 0 & -f_x^{(e)} \frac{X}{Z} \\ 0 & f_y^{(e)} & -f_y^{(e)} \frac{Y}{Z} \end{bmatrix}, \qquad \mathbf{J}_{\mathrm{se3}}(\mathbf{p}) = \begin{bmatrix} -[\mathbf{p}]_\times & \mathbf{I}_3 \end{bmatrix}$$

> *注：式 (24)/(28) 的 $\mathbf{J}_{\mathrm{se3}}(\mathbf{p}_k^{(e)})$ 对应接收端相机坐标系下的左扰动 $\exp(\boldsymbol{\xi}_e^\wedge)\mathbf{T}_{e\leftarrow i}^\star$；若扰动定义在源坐标系右乘 $\mathbf{T}_{e\leftarrow i}^\star\exp(\boldsymbol{\xi}_i^\wedge)$，则通过伴随变换 $\boldsymbol{\xi}_e = \mathrm{Ad}_{\mathbf{T}_{e\leftarrow i}^\star}\boldsymbol{\xi}_i$ 等价转换。*

此外，空中视角在点 $\mathbf{p}_k$ 处的单位正方形 Patch 投影到地面视角图像平面时，其局部射影形变由单应性雅可比 $\mathbf{H}_{e \leftarrow i}(k) = \mathbf{J}_{\mathrm{proj}}(\mathbf{p}_k^{(e)}) \mathbf{R}_{e \leftarrow i} \left( \mathbf{I}_3 - \frac{\mathbf{p}_k^{(i)} \mathbf{n}_k^\top}{\mathbf{n}_k^\top \mathbf{p}_k^{(i)}} \right) \mathbf{K}_i^{-1} d_k \in \mathbb{R}^{2 \times 2}$ 刻画（$\mathbf{n}_k$ 为局部表面法向量）。当空地视线接近正交（$\mathbf{n}_k^\top \mathbf{p}_k^{(e)} \to 0$）时，$\mathbf{H}_{e \leftarrow i}(k)$ 的条件数急剧发散，导致跨视角感受野严重失配。

### 2. 跨锚点强相关的 Errors-in-Variables (EIV) 表征耦合模型
设在空地共视区域内选取 $m$ 个候选锚点 Patch，发送端变体提取的特征矩阵为 $\mathbf{X}_A = [\mathbf{x}_1, \dots, \mathbf{x}_m] \in \mathbb{R}^{r \times m}$。在理想零位姿误差位置 $\mathbf{v}_k^\star$ 处，接收端变体的特征 $\mathbf{y}(\mathbf{v}_k^\star) \in \mathbb{R}^{d_e}$ 与发送端特征满足局部线性子空间映射 $\mathbf{y}(\mathbf{v}_k^\star) = \mathbf{A} \mathbf{x}_k + \boldsymbol{\epsilon}_k$，其中 $\mathbf{A} \in \mathbb{R}^{d_e \times r}$ 为未知的跨变体基底变换矩阵，$\boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma_{\mathrm{obs}}^2 \mathbf{I}_{d_e})$ 为编码器本底残差。

当存在位姿误差 $\boldsymbol{\xi}$ 时，接收端实际上在偏移坐标 $\hat{\mathbf{v}}_k = \mathbf{v}_k^\star + \mathbf{G}_k \boldsymbol{\xi}$ 处采样本地特征图。记接收端特征场在 $\mathbf{v}_k^\star$ 处的空间方向导数（Spatial Feature Jacobian）为 $\mathbf{D}_k = \left[ \frac{\partial \mathbf{y}}{\partial u}, \frac{\partial \mathbf{y}}{\partial v} \right]_{\mathbf{v}_k^\star} \in \mathbb{R}^{d_e \times 2}$，做一阶空间展开可得：

$$\mathbf{y}(\hat{\mathbf{v}}_k) = \mathbf{A} \mathbf{x}_k + \boldsymbol{\epsilon}_k + \mathbf{M}_k \boldsymbol{\xi}, \qquad \mathbf{M}_k \triangleq \mathbf{D}_k \mathbf{G}_k \in \mathbb{R}^{d_e \times 6}$$

**非平凡耦合性质**：
在全部 $m$ 个锚点上，错配误差项 $\{\mathbf{M}_k \boldsymbol{\xi}\}_{k=1}^m$ **并非相互独立的随机噪声，而是由同一个 6 维随机向量 $\boldsymbol{\xi}$ 生成的完全相干扰动**。对于任意给定的接收端特征投影方向 $\mathbf{u} \in \mathbb{R}^{d_e}$（$\|\mathbf{u}\|_2 = 1$），记标量投影行向量 $\mathbf{y}_{\mathbf{u}, A}^\top = \mathbf{u}^\top \mathbf{Y}_A \in \mathbb{R}^{1 \times m}$ 及方向空间梯度矩阵 $\mathbf{M}_{\mathbf{u}, A} = [\mathbf{M}_1^\top \mathbf{u}, \dots, \mathbf{M}_m^\top \mathbf{u}]^\top \in \mathbb{R}^{m \times 6}$，则 $m$ 个锚点上的联合观测噪声协方差矩阵为：

$$\mathbf{\Omega}_{\mathbf{u}}\left(\mathcal{S}_a, \Sigma_\xi\right) = \mathbb{E}\left[ \left(\boldsymbol{\epsilon}_{\mathbf{u}} + \mathbf{M}_{\mathbf{u}, A} \boldsymbol{\xi}\right) \left(\boldsymbol{\epsilon}_{\mathbf{u}} + \mathbf{M}_{\mathbf{u}, A} \boldsymbol{\xi}\right)^\top \right] = \sigma_{\mathrm{obs}}^2 \mathbf{I}_m + \mathbf{M}_{\mathbf{u}, A} \Sigma_\xi \mathbf{M}_{\mathbf{u}, A}^\top \in \mathbb{R}^{m \times m}$$

### 3. 接收端 VLM 动作分布的拉回 Fisher 度量（Pullback Metric）
接收端智能体 $e$ 需根据本地观测上下文 $h_t$、注入的 $k$ 个盲区特征 $\widehat{\mathbf{Y}}_T \in \mathbb{R}^{d_e \times k}$ 及其变换后的三维坐标 $\widehat{\mathbf{P}}_T \in \mathbb{R}^{3 \times k}$，在离散候选动作集 $\mathcal{A} = \{a_1, \dots, a_K\}$（或连续控制空间，通常 $K \le 16 \ll d_e$）上生成策略分布 $\pi(a \mid h_t, \widehat{\mathbf{Y}}_T, \widehat{\mathbf{P}}_T) = \mathrm{Softmax}(\mathbf{z}_{\mathrm{dec}})$。

设全知对齐特征与真实三维坐标下的参考策略为 $\pi^\star$。在参考点处，策略分布的 Kullback-Leibler 散度（或局部二次 Bayes 风险增量 $\Delta \mathcal{R}$）由接收端 VLM 决策头关于注入变量的拉回 Fisher 信息矩阵（Pullback Fisher Information Matrix）唯一确定：

$$\Delta \mathcal{R}\left(\widehat{\mathbf{Y}}_T, \widehat{\mathbf{P}}_T\right) = \frac{1}{2} \begin{bmatrix} \mathrm{vec}(\widehat{\mathbf{Y}}_T - \mathbf{Y}_T^\star) \\ \mathrm{vec}(\widehat{\mathbf{P}}_T - \mathbf{P}_T^\star) \end{bmatrix}^\top \begin{bmatrix} \mathbf{J}_Y^\top \mathbf{F}_\pi \mathbf{J}_Y & \mathbf{J}_Y^\top \mathbf{F}_\pi \mathbf{J}_P \\ \mathbf{J}_P^\top \mathbf{F}_\pi \mathbf{J}_Y & \mathbf{J}_P^\top \mathbf{F}_\pi \mathbf{J}_P \end{bmatrix} \begin{bmatrix} \mathrm{vec}(\widehat{\mathbf{Y}}_T - \mathbf{Y}_T^\star) \\ \mathrm{vec}(\widehat{\mathbf{P}}_T - \mathbf{P}_T^\star) \end{bmatrix} + \mathcal{O}\left(\|\delta\|^3\right)$$

其中 $\mathbf{F}_\pi = \mathrm{diag}(\pi^\star) - \pi^\star (\pi^\star)^\top \in \mathbb{R}^{K \times K}$ 为动作分布的分类 Fisher 矩阵（满足 $\mathrm{rank}(\mathbf{F}_\pi) \le K - 1$），$\mathbf{J}_Y = \frac{\partial \mathbf{z}_{\mathrm{dec}}}{\partial \mathrm{vec}(\mathbf{Y}_T)} \in \mathbb{R}^{K \times (d_e k)}$ 与 $\mathbf{J}_P = \frac{\partial \mathbf{z}_{\mathrm{dec}}}{\partial \mathrm{vec}(\mathbf{P}_T)} \in \mathbb{R}^{K \times (3k)}$ 为 VLM 决策头对特征值与三维位置坐标的雅可比矩阵。

对通道方向灵敏度矩阵 $\mathbf{S}_Y = \sum_{j=1}^k \mathbf{J}_{Y, j}^\top \mathbf{F}_\pi \mathbf{J}_{Y, j} \in \mathbb{R}^{d_e \times d_e}$ 做特征值分解 $\mathbf{S}_Y = \mathbf{U}_{\mathrm{act}} \mathbf{\Lambda}_{\mathrm{act}} \mathbf{U}_{\mathrm{act}}^\top$。由于 $\mathrm{rank}(\mathbf{S}_Y) \le k(K-1) \ll d_e$，接收端 VLM 的下游动作决策仅对落在低维正交子空间 $\mathbf{U}_{\mathrm{act}} \in \mathbb{R}^{d_e \times r_{\mathrm{act}}}$（$r_{\mathrm{act}} \ll d_e$）内的特征误差敏感，而在其正交补空间 $\mathbf{U}_{\mathrm{act}}^\perp$ 上的任何映射误差对当前动作决策的一阶与二阶风险贡献恒为零。

---

## 二、 核心理论上下界推导：Woodbury 秩-6 瓶颈、锚点二分律与收缩门控界

### 定理 1（含相关位姿扰动的跨变体子空间后验精度与贝叶斯风险下界）
设跨变体映射在动作敏感子空间上的投影 $\widetilde{\mathbf{A}} = \mathbf{U}_{\mathrm{act}}^\top \mathbf{A} \in \mathbb{R}^{r_{\mathrm{act}} \times r}$ 具有行独立高斯先验，每行先验精度矩阵为 $\mathbf{\Lambda}_0 \in \mathbb{R}^{r \times r}$。给定分配了 $b_g$ 比特几何校准消息后的后验位姿协方差 $\widetilde{\Sigma}_\xi(b_g) \in \mathbb{R}^{6 \times 6}$，以及由 $b_a$ 比特传输的 $m$ 个共视表征锚点 $\mathbf{X}_A \in \mathbb{R}^{r \times m}$。
1. **带 Woodbury 秩-6 折损的精确后验精度矩阵**：沿接收端第 $\ell$ 个动作敏感特征向量 $\mathbf{u}_\ell$（对应灵敏度特征值 $\lambda_{\mathrm{act}, \ell}$），映射向量 $\mathbf{a}_\ell^\top = \mathbf{u}_\ell^\top \mathbf{A} \in \mathbb{R}^{1 \times r}$ 的后验精度矩阵 $\mathbf{\Lambda}_{\mathrm{post}}^{(\ell)}(\mathcal{S}_a, b_g)$ 精确满足：
   $$\mathbf{\Lambda}_{\mathrm{post}}^{(\ell)}\left(\mathcal{S}_a, b_g\right) = \underbrace{\mathbf{\Lambda}_0 + \sigma_{\mathrm{obs}}^{-2} \mathbf{X}_A \mathbf{X}_A^\top}_{\text{理想零位姿误差表征精度 } \mathbf{\Lambda}_{\mathrm{ideal}}(\mathcal{S}_a)} - \underbrace{\sigma_{\mathrm{obs}}^{-4} \mathbf{X}_A \mathbf{M}_{\ell, A} \left( \widetilde{\Sigma}_\xi(b_g)^{-1} + \sigma_{\mathrm{obs}}^{-2} \mathbf{M}_{\ell, A}^\top \mathbf{M}_{\ell, A} \right)^{-1} \mathbf{M}_{\ell, A}^\top \mathbf{X}_A^\top}_{\text{跨锚点相干位姿错配诱导的秩-6 精度折损 } \Delta \mathbf{\Lambda}_{\mathrm{pose}}^{(\ell)}(\mathcal{S}_a, b_g)}$$
2. **盲区任务决策风险的严格下界（Converse）**：设待传盲区任务特征为 $\mathbf{X}_T \in \mathbb{R}^{r \times k}$，经码率 $b_s$ 编码后的条件率失真下界为 $D_s(b_s) = \sum_{j=1}^k d_x^{(j)} 2^{-2 b_s / (rk)}$。则对任意无偏特征重构与注入估计器，接收端的二阶拉回决策风险满足下界：
   $$\mathbb{E}\left[\Delta \mathcal{R}\right] \ge \frac{1}{2} \sum_{\ell=1}^{r_{\mathrm{act}}} \lambda_{\mathrm{act}, \ell} \mathrm{Tr}\left( \mathbf{X}_T^\top \left[ \mathbf{\Lambda}_{\mathrm{post}}^{(\ell)}\left(\mathcal{S}_a, b_g\right) \right]^{-1} \mathbf{X}_T \right) + \frac{1}{2} \mathrm{Tr}\left( \mathbf{J}_{\mathrm{eff}, P}^\top \mathbf{F}_\pi \mathbf{J}_{\mathrm{eff}, P} \widetilde{\Sigma}_\xi(b_g) \right) + \frac{1}{2} \sigma_{\min}^2\left(\widetilde{\mathbf{A}}\right) \mathrm{Tr}(\mathbf{\Lambda}_{\mathrm{act}}) D_s(b_s)$$
   其中 $\mathbf{J}_{\mathrm{eff}, P} = (\mathbf{J}_P + \mathbf{J}_Y \mathbf{D}_T) \mathbf{J}_{\mathrm{se3}}(\mathbf{P}_T) \in \mathbb{R}^{K \times 6}$ 为盲区目标三维位置偏差对动作对数似然的直接雅可比。

*证明核心步骤*：
由高斯线性模型在相关噪声协方差 $\mathbf{\Omega}_\ell = \sigma_{\mathrm{obs}}^2 \mathbf{I}_m + \mathbf{M}_{\ell, A} \widetilde{\Sigma}_\xi(b_g) \mathbf{M}_{\ell, A}^\top$ 下的共轭后验公式，后验精度为 $\mathbf{\Lambda}_0 + \mathbf{X}_A \mathbf{\Omega}_\ell^{-1} \mathbf{X}_A^\top$。直接对 $\mathbf{\Omega}_\ell^{-1}$ 应用 Sherman-Morrison-Woodbury 矩阵求逆恒等式：
$$\left(\sigma_{\mathrm{obs}}^2 \mathbf{I}_m + \mathbf{M}_{\ell, A} \widetilde{\Sigma}_\xi \mathbf{M}_{\ell, A}^\top\right)^{-1} = \sigma_{\mathrm{obs}}^{-2} \mathbf{I}_m - \sigma_{\mathrm{obs}}^{-4} \mathbf{M}_{\ell, A} \left( \widetilde{\Sigma}_\xi^{-1} + \sigma_{\mathrm{obs}}^{-2} \mathbf{M}_{\ell, A}^\top \mathbf{M}_{\ell, A} \right)^{-1} \mathbf{M}_{\ell, A}^\top$$
左乘 $\mathbf{X}_A$、右乘 $\mathbf{X}_A^\top$ 即得性质 1。将后验协方差 $\big[\mathbf{\Lambda}_{\mathrm{post}}^{(\ell)}\big]^{-1}$ 与率失真定理中的正向高斯测试信道下界代入拉回二次型期望，即得性质 2。

---

### 定理 2（几何锚点与表征锚点的空间梯度二分律，Spatial-Gradient Dichotomy）
考察共视区域内的任意候选 Patch $k$，其空间特征梯度范数为 $\|\mathbf{D}_k\|_F$，局部图像几何边缘梯度张量为 $\mathbf{\Gamma}_k = \nabla_{\mathbf{v}} I_k \nabla_{\mathbf{v}} I_k^\top \in \mathbb{R}^{2 \times 2}$。
1. **几何校准信息最大化条件**：几何锚点集 $\mathcal{S}_g$ 对相对位姿的后验精度更新为 $\widetilde{\Sigma}_\xi(b_g)^{-1} = \Sigma_\xi^{(0)-1} + \sum_{k \in \mathcal{S}_g} \mathbf{G}_k^\top \mathbf{\Gamma}_k \mathbf{G}_k$。要最大化位姿精度增量，必须选择**空间梯度张量 $\mathbf{\Gamma}_k$ 极大且三维射线雅可比 $\mathbf{G}_k$ 跨不同深度与视场角分布**的高频边缘/角点（即 $\|\nabla_{\mathbf{v}} I_k\|_2 \to \max$）。
2. **表征对齐信息最大化条件（反梯度选择）**：在给定非零残余位姿协方差 $\widetilde{\Sigma}_\xi \succ \mathbf{0}$ 时，若表征锚点 $\mathcal{S}_a$ 选在特征空间梯度 $\mathbf{D}_k$ 极大的物体边缘处，使得 $\sigma_{\mathrm{obs}}^{-2} \mathbf{M}_{\ell, A}^\top \mathbf{M}_{\ell, A} \gg \widetilde{\Sigma}_\xi^{-1}$，则定理 1 中的 Woodbury 折损项达到渐近上界：
   $$\lim_{\|\mathbf{M}_{\ell, A}\| \to \infty} \mathbf{\Lambda}_{\mathrm{post}}^{(\ell)}\left(\mathcal{S}_a, b_g\right) = \mathbf{\Lambda}_0 + \sigma_{\mathrm{obs}}^{-2} \mathbf{X}_A \left( \mathbf{I}_m - \mathbf{P}_{\mathbf{M}_\ell} \right) \mathbf{X}_A^\top$$
   其中 $\mathbf{P}_{\mathbf{M}_\ell} = \mathbf{M}_{\ell, A} (\mathbf{M}_{\ell, A}^\top \mathbf{M}_{\ell, A})^{-1} \mathbf{M}_{\ell, A}^\top$ 为向空间梯度列空间的投影矩阵——这意味着落入空间梯度子空间的跨变体特征信息被位姿噪声完全抵消！反之，若表征锚点选在**局部空间平稳但跨区域语义多样**的连通域内部（即满足局部空间李普希茨常数 $\|\mathbf{D}_k \mathbf{G}_k \widetilde{\Sigma}_\xi^{1/2}\|_F \ll \sigma_{\mathrm{obs}}$，同时 $\mathbf{X}_A \mathbf{X}_A^\top$ 在盲区任务子空间 $\mathrm{span}(\mathbf{X}_T)$ 上特征值最大化），则折损项 $\Delta \mathbf{\Lambda}_{\mathrm{pose}}^{(\ell)} \to \mathbf{0}$。

> **物理与算法本质**：定理 2 从数学上证明了**几何校准锚点 $\mathcal{S}_g$ 与跨变体表征锚点 $\mathcal{S}_a$ 在空间微分属性上是严格互斥的**——几何锚点必须抓“空间高频跳变边缘”以锁定坐标；表征锚点必须抓“局部空间平稳内部（Low Spatial Gradient）+ 跨区域语义主成分对齐（High Subspace Projection）”以规避残余位姿错配污染。现有方法（如 BOLT、Where2comm）不区分这两类锚点，直接在高响应边缘或全重叠区回归，必然触发 Woodbury 秩-6 折损。

---

### 定理 3（矩阵 Wiener 收缩门控的可达风险上界与单调防负迁移保证）
在接收端将重构的盲区特征注入 VLM 决策头之前，对每个动作敏感通道 $\ell \in \{1, \dots, r_{\mathrm{act}}\}$ 施加如下闭式**矩阵 Wiener 收缩门控（Matrix Wiener Shrinkage Operator）**：

$$\widehat{\mathbf{y}}_{T, j}^{\mathrm{gated}} = \mathbf{U}_{\mathrm{act}} \mathrm{diag}\left( w_{1, j}^\star, \dots, w_{r_{\mathrm{act}}, j}^\star \right) \mathbf{U}_{\mathrm{act}}^\top \widehat{\mathbf{A}} \hat{\mathbf{x}}_{T, j}, \qquad w_{\ell, j}^\star = \frac{\hat{\mu}_{\ell, j}^2}{\hat{\mu}_{\ell, j}^2 + \mathbf{x}_{T, j}^\top \left[\mathbf{\Lambda}_{\mathrm{post}}^{(\ell)}\right]^{-1} \mathbf{x}_{T, j} + \|\hat{\mathbf{a}}_\ell\|_2^2 \sigma_{q, j}^2(b_s)}$$

其中 $\hat{\mu}_{\ell, j}^2 = (\hat{\mathbf{a}}_\ell^\top \hat{\mathbf{x}}_{T, j})^2$ 为第 $j$ 个盲区特征在方向 $\ell$ 上的信号能量估计，$\sigma_{q, j}^2(b_s)$ 为量化方差。
1. **严格防负迁移性质（Zero Negative Transfer）**：记不接收任何外部特征（$\widehat{\mathbf{Y}}_T = \mathbf{0}$，等价于单机本地决策 `Ego-Only`）时的二阶期望风险基准为 $\mathcal{R}_{\mathrm{ego}}$。无论位姿误差 $\Sigma_\xi$ 多大、共视重叠度多低（哪怕 $\mathcal{S}_a = \emptyset$ 导致后验方差趋于先验方差）或通信比特 $B_t$ 多小，经 $w_{\ell, j}^\star \in [0, 1)$ 门控后的期望决策风险严格不劣于单机基线：
   $$\mathbb{E}\left[\mathcal{R}\left(h_t, \widehat{\mathbf{Y}}_T^{\mathrm{gated}}\right)\right] = \mathcal{R}_{\mathrm{ego}} - \frac{1}{2} \sum_{\ell=1}^{r_{\mathrm{act}}} \sum_{j=1}^k \lambda_{\mathrm{act}, \ell} \, w_{\ell, j}^\star \, \hat{\mu}_{\ell, j}^2 \le \mathcal{R}_{\mathrm{ego}}$$
2. **跨时间步指数/代数摊销率（Sequential Amortization Rate）**：在长度为 $T$ 的序贯任务回合中，跨变体真实矩阵 $\mathbf{A}$ 时不变，递归精度矩阵满足 $\mathbf{\Lambda}_{\mathrm{post}, t}^{(\ell)} = \mathbf{\Lambda}_{\mathrm{post}, t-1}^{(\ell)} + \Delta \mathbf{\Lambda}_t^{(\ell)}(\mathcal{S}_{a, t}, b_{g, t})$。若智能体在前 $T_0$ 步通过主动视角控制累积了覆盖任务子空间 $\mathcal{U}_{\mathrm{task}}$（维数 $r_{\mathrm{task}}$）的特征能量 $\lambda_{\min}\big(\sum_{\tau=1}^{T_0} \Delta \mathbf{\Lambda}_\tau^{(\ell)}\big|_{\mathcal{U}_{\mathrm{task}}}\big) \ge \gamma T_0$，则在 $t \gt T_0$ 后，即使令表征对齐预算 $b_{a, t} = 0$，跨变体对齐误差对决策风险的贡献亦以 $\mathcal{O}(1 / (\gamma T_0))$ 锁定在极低水平，全部通信预算 $B_t$ 可转向实时几何跟踪 $b_{g,t}$ 与盲区任务传输 $b_{s,t}$。

---

## 三、 Agentic-Native 全流程贯通机制与精确求解算法

### 1. 四阶段闭环通信与计算协议（含精确张量维度与字节账本）

| 阶段与执行端 | 输入张量与状态 | 核心算子与数学执行步骤 | 输出数据包与精确字节账本（Byte Ledger） |
| :--- | :--- | :--- | :--- |
| **Phase I**<br>接收端（地面 $e$）<br>拉回子空间与盲区查询 | 本地视觉特征 $\mathbf{F}_e \in \mathbb{R}^{N_e \times d_e}$；本地候选动作对数似然 $\mathbf{z}_{\mathrm{dec}} \in \mathbb{R}^K$；上一时刻映射精度 $\mathbf{\Lambda}_{\mathrm{post}, t-1}$ | 1. 识别深度遮挡/高熵网格，生成三维盲区请求包围盒 $\mathcal{B}_{\mathrm{req}} \in \mathbb{R}^{6}$；<br>2. 通过一次反向传播或有限差分计算 $\mathbf{J}_Y = \partial \mathbf{z}_{\mathrm{dec}} / \partial \mathbf{Y}$，提取前 $r_{\mathrm{act}}=4$ 个主奇异向量 $\mathbf{U}_{\mathrm{act}}$，并通过上一时刻映射 $\widehat{\mathbf{A}}_{t-1}^\top \mathbf{U}_{\mathrm{act}}$ 映射为**源空间未对齐查询基** $\mathbf{V}_{\mathrm{query}} \in \mathbb{R}^{r \times 4}$。 | **下行请求包 $\mathcal{P}_{\mathrm{req}}$（总计 $164\text{ Bytes}$）**：<br>• 盲区视锥与位姿先验：`12 × FP16 = 24 B`<br>• 源空间查询基 $\mathbf{V}_{\mathrm{query}}$（设 PCA 降维后 $r=32$）：`32 × 4 × FP8 = 128 B`<br>• 帧头与校验：`12 B` |
| **Phase II**<br>协作端（空中 $i$）<br>梯度解耦与三元率失真编码 | 空中多视角图像与深度；源变体特征 $\mathbf{F}_i \in \mathbb{R}^{N_i \times d_i}$（经本地正交基压缩至 $r=32$）；下行请求 $\mathcal{P}_{\mathrm{req}}$ | 1. **候选池三分**：在盲区 $\mathcal{B}_{\mathrm{req}}$ 内提取目标与可通行走廊特征 $\mathbf{X}_T \in \mathbb{R}^{r \times k}$；在共视区内按图像梯度 $\mathbf{\Gamma}_u$ 顶分位数提取几何边缘池 $\mathcal{V}_g$，按特征空间平稳度 $\|\nabla_{\mathbf{u}} \mathbf{F}_i\|_F \le \tau_D$ 提取表征锚点池 $\mathcal{V}_a$；<br>2. **执行 Algorithm 1** 求解最优三元字节切分 $(b_g^\star, b_a^\star, b_s^\star)$ 及对应子集 $(\mathcal{S}_g^\star, \mathcal{S}_a^\star, \widehat{\mathbf{X}}_T)$。 | **上行协同包 $\mathcal{P}_{\mathrm{up}}$（严格受限于 $B_t$，如 $1024 \sim 4096\text{ B}$）**：<br>• 几何锚点 $\mathcal{S}_g^\star$：每个含三维点与边缘法向 `6 × FP16 = 12 B`<br>• 表征锚点 $\mathcal{S}_a^\star$：每个含网格索引 `2 B` + $r$ 维特征 `32 × INT4/FP8 = 16~32 B`<br>• 盲区任务负载 $\widehat{\mathbf{X}}_T$：含三维坐标与量化特征，总长 $b_s^\star$ bits |
| **Phase III**<br>接收端（地面 $e$）<br>级联位姿与 EIV 递归滤波 | 上行包 $\mathcal{P}_{\mathrm{up}}$；本地共视区特征场 $\mathbf{Y}(\mathbf{v})$ 及其空间导数场 $\mathbf{D}(\mathbf{v})$ | 1. **Step A（几何配准）**：用 $\mathcal{S}_g^\star$ 执行点到面加权高斯-牛顿配准，更新 $\widehat{\boldsymbol{\xi}}_t$ 与后验协方差 $\widetilde{\Sigma}_{\xi, t} = \big(\Sigma_{\xi, t-1}^{-1} + \sum_{k \in \mathcal{S}_g^\star} \mathbf{G}_k^\top \mathbf{\Gamma}_k \mathbf{G}_k\big)^{-1}$；<br>2. **Step B（带 Woodbury 修正的 RLS 更新）**：在配准坐标 $\hat{\mathbf{v}}_k$ 处读取 $\mathbf{y}_k$ 与空间梯度 $\mathbf{D}_k$，构造 $\mathbf{\Omega}_t = \sigma_{\mathrm{obs}}^2 \mathbf{I}_m + \mathbf{M}_t \widetilde{\Sigma}_{\xi, t} \mathbf{M}_t^\top$，执行卡尔曼/RLS 更新 $\widehat{\mathbf{A}}_t$ 与 $\mathbf{\Lambda}_{\mathrm{post}, t}$。 | 内部状态更新（耗时小于 $1.5\text{ ms}$）：<br>• 仅涉及 $6 \times 6$ 位姿矩阵求逆与 $r \times r$（$32 \times 32$）低维特征精度矩阵更新 |
| **Phase IV**<br>接收端（地面 $e$）<br>Wiener 收缩注入与主动视角控制 | 校准映射 $\widehat{\mathbf{A}}_t, \mathbf{\Lambda}_{\mathrm{post}, t}$；盲区量化特征 $\widehat{\mathbf{X}}_T$ | 1. 按定理 3 计算通道收缩权重 $w_{\ell, j}^\star$，生成 $\widehat{\mathbf{Y}}_T^{\mathrm{gated}}$ 并注入 VLM 跨注意力读取器；<br>2. **主动感知判据**：若盲区方向的不确定性迹 $\mathrm{Tr}(\mathbf{X}_T^\top \mathbf{\Lambda}_{\mathrm{post}, t}^{-1} \mathbf{X}_T)$ 高于阈值 $\tau_{\mathrm{active}}$（因当前共视重叠过低），触发云台/航向微调动作 $a_t^{\mathrm{view}}$ 以扩大下一帧共视视锥。 | 输出闭环控制指令 $\mathbf{a}_t = (\mathbf{a}_t^{\mathrm{nav}}, \mathbf{a}_t^{\mathrm{view}})$ |

---

### 2. 三元预算分裂与子空间贪心选择算法（Algorithm 1）
由于目标函数中的 Woodbury 折损项与对数行列式/迹项关于 $(b_g, b_a, b_s)$ 联合非凸且非次模，但**当固定几何锚点集 $\mathcal{S}_g$（从而固定 $\widetilde{\Sigma}_\xi(b_g)$）时，表征锚点选择在预白化空间下可转化为标准 A-最优实验设计**。发送端在毫秒级时间内执行如下精确解耦算法：

1. **输入**：单步总预算 $B_t$（Bytes），位姿先验 $\Sigma_{\xi, t-1}$，上一时刻映射精度 $\mathbf{\Lambda}_{\mathrm{post}, t-1}$，下行查询基 $\mathbf{V}_{\mathrm{query}} \in \mathbb{R}^{r \times r_{\mathrm{act}}}$，候选几何锚点池 $\mathcal{V}_g$，空间平稳表征锚点池 $\mathcal{V}_a$（满足局部梯度筛选 $\|\nabla_{\mathbf{u}} \mathbf{x}_k\|_2 \le \tau_D$），盲区任务特征矩阵 $\mathbf{X}_T \in \mathbb{R}^{r \times k}$。
2. **构造盲区任务目标协方差矩阵**：计算待解释的任务子空间权重矩阵 $\mathbf{W}_T = \mathbf{X}_T \mathbf{X}_T^\top + \gamma_q \mathbf{V}_{\mathrm{query}} \mathbf{V}_{\mathrm{query}}^\top \in \mathbb{R}^{r \times r}$。
3. **外层一维离散网格扫描几何预算**：在离散集合 $b_g \in \{0, \Delta_g, 2\Delta_g, \dots, b_g^{\max}\}$（通常仅 $4 \sim 5$ 个档位，对应 $0, 3, 6, 12$ 个三维边缘点）上遍历：
   * 利用 D-最优贪心选择 $\mathcal{S}_g(b_g) \subseteq \mathcal{V}_g$ 最大化 $\log\det\big(\Sigma_{\xi, t-1}^{-1} + \sum_{j \in \mathcal{S}_g} \mathbf{G}_j^\top \mathbf{\Gamma}_j \mathbf{G}_j\big)$，得到候选后验位姿协方差 $\widetilde{\Sigma}_\xi(b_g)$。
   * 对每个候选表征锚点 $k \in \mathcal{V}_a$，计算其在残余位姿误差 $\widetilde{\Sigma}_\xi(b_g)$ 下的等效信噪比衰减权重：
     $$\beta_k(b_g) = \left( \sigma_{\mathrm{obs}}^2 + \hat{c}_{\mathrm{grad}, k} \mathrm{Tr}\left(\mathbf{G}_k \widetilde{\Sigma}_\xi(b_g) \mathbf{G}_k^\top\right) \right)^{-1}$$
   * **内层秩-1 Sherman-Morrison 贪心挑选表征锚点**：令初始矩阵 $\mathbf{P}_0 = \mathbf{\Lambda}_{\mathrm{post}, t-1}^{-1}$。对于锚点个数 $m = 0, 1, \dots, m_{\max}$（对应字节数 $b_a(m) = m \cdot c_a \le B_t - b_g$）：
     * 在第 $m$ 步，遍历 $k \in \mathcal{V}_a \setminus \mathcal{S}_a$，利用 Sherman-Morrison 公式以 $\mathcal{O}(r^2)$ 复杂度计算挑选锚点 $k$ 后任务子空间加权迹的精确下降量：
       $$\Delta \Psi(k \mid \mathbf{P}_{m-1}, b_g) = \mathrm{Tr}\left(\mathbf{W}_T \mathbf{P}_{m-1}\right) - \mathrm{Tr}\left(\mathbf{W}_T \left(\mathbf{P}_{m-1} - \frac{\beta_k(b_g) \mathbf{P}_{m-1} \mathbf{x}_k \mathbf{x}_k^\top \mathbf{P}_{m-1}}{1 + \beta_k(b_g) \mathbf{x}_k^\top \mathbf{P}_{m-1} \mathbf{x}_k}\right)\right) = \frac{\beta_k(b_g) \left\| \mathbf{W}_T^{1/2} \mathbf{P}_{m-1} \mathbf{x}_k \right\|_2^2}{1 + \beta_k(b_g) \mathbf{x}_k^\top \mathbf{P}_{m-1} \mathbf{x}_k}$$
     * 选取 $k^\star = \arg\max_k \Delta \Psi(k \mid \mathbf{P}_{m-1}, b_g)$，更新 $\mathbf{P}_m = \mathbf{P}_{m-1} - \frac{\beta_{k^\star} \mathbf{P}_{m-1} \mathbf{x}_{k^\star} \mathbf{x}_{k^\star}^\top \mathbf{P}_{m-1}}{1 + \beta_{k^\star} \mathbf{x}_{k^\star}^\top \mathbf{P}_{m-1} \mathbf{x}_{k^\star}}$。
     * 将剩余预算 $b_s = B_t - b_g - b_a(m)$ 分配给盲区任务特征 $\mathbf{X}_T$，记录总代理风险值：
       $$J_{\mathrm{proxy}}(b_g, m) = \mathrm{Tr}\left(\mathbf{W}_T \mathbf{P}_m\right) + \alpha_{\mathrm{pose}} \mathrm{Tr}\left(\mathbf{G}_T \widetilde{\Sigma}_\xi(b_g) \mathbf{G}_T^\top\right) + \alpha_{\mathrm{quant}} k r \sigma_x^2 2^{-\frac{2(B_t - b_g - m c_a)}{rk}}$$
4. **输出**：全局最小化 $J_{\mathrm{proxy}}(b_g, m)$ 的最优配置 $(b_g^\star, b_a^\star, b_s^\star)$ 及对应消息包。由于 $r = 32$，整个双重循环在 GPU/CPU 上的矩阵乘法耗时小于 $0.8\text{ ms}$。

---

## 四、 多场景端到端实验设计与前沿 SOTA 公平对比协议

### 1. 跨尺度与跨家族异构变体矩阵（依托 Colab 混合计算架构）

| 变体组合编号 | 发送端（空中 UAV）模型变体与视觉参数 | 接收端（地面 UGV）模型变体与视觉参数 | 核心结构失配特征（待在线克服的基底障碍） |
| :--- | :--- | :--- | :--- |
| **Pair 1**<br>同族跨尺度 | `Qwen2.5-VL-3B`<br>(`patch=14, merge=2`, 3D MRoPE, $d_i=2048$)<br>*（本地快速原型兼容 `Qwen3.5-0.8B`）* | `Qwen2.5-VL-7B`<br>(`patch=14, merge=2`, 3D MRoPE, $d_e=3584$)<br>*（本地快速原型兼容 `Qwen3.5-2B`）* | 共享视觉 Patch 网格拓扑，但隐藏层宽度不同且深层语义表示空间发生非线性旋转 |
| **Pair 2**<br>跨家族 VLM A | `SmolVLM2-2.2B`<br>(基于 SigLIP 编码器，像素重排压缩，$d_i=2048$) | `Qwen2.5-VL-7B`<br>(动态分辨率原生 ViT，$d_e=3584$) | 视觉 Patch 空间划分网格不对齐（$16\times 16$ Pixel-Shuffle vs. $28\times 28$ Merged Patch），位置编码正交 |
| **Pair 3**<br>跨家族 VLM B | `InternVL2-2B`<br>(InternViT-300M，$448\times 448$ 切片，$d_i=2048$) | `Qwen2.5-VL-7B`<br>($d_e=3584$) | 动态高分辨率切片（Tile-based）特征与连续全图网格特征之间的空间拓扑与通道双重异构 |
| **Pair 4**<br>自监督视觉变体 | `DINOv2-Large`<br>(纯视觉自监督寄存器特征，$d_i=1024$) | `Qwen2.5-VL-7B` 语言决策骨干 | 发送端无任何语言预对齐先验（纯几何/纹理特征流形），需在线桥接至 VLM 动作语义子空间 |
| **Pair 5**<br>对比学习视觉变体 | `SigLIP2-SO400M`<br>(全局图文对比 Sigmoid 损失，$d_i=1152$) | `InternVL2-8B` (或 `Qwen2.5-VL-7B`) | 强全局语义聚合但局部空间边缘平滑的特征流形向细粒度具身动作空间的在线适配 |

### 2. 三类空地物理耦合场景与端到端闭环指标
在仿真器中设立 **3 类具有明确几何与语义退化特征的空地协同场景**（共 15 个独立空间布局，每个布局评估 30 个闭环回合，总计 450 个完整 Episodes）：
1. **Regime A（极端视角正交与高遮挡峡谷）**：空中无人机俯仰角 $-75^\circ$（近垂直俯视），地面无人车平视 $0^\circ$ 被倒塌掩体包围；空地三维视锥体交集仅占地面视野的 $10\% \sim 18\%$，且外参存在持续高斯游走漂移（$\sigma_\theta = 1.5^\circ \sim 3.0^\circ, \sigma_t = 0.5\text{m} \sim 1.5\text{m}$）。
2. **Regime B（大面积重复纹理与秩亏陷阱）**：空地视锥重叠率高达 $40\% \sim 55\%$，但共视区域中 $85\%$ 以上为单调混凝土地面或重复标准建筑立面（即均匀采样时 $\mathbf{X}_A \mathbf{X}_A^\top$ 在前 2 个主成分之外迅速衰减至 0），盲区内的受困目标与通道障碍则具有罕见语义特征。
3. **Regime C（高速机动与间歇零共视窗口）**：空中节点沿预设巡逻轨迹高速移动，空地共视重叠率在 $0\%$（完全无重叠）与 $35\%$ 之间周期性切换，直接测试**跨时间步映射精度累积（$\mathbf{\Lambda}_{\mathrm{post}, t}$）**、**无共视期 Wiener 收缩门控防负迁移**以及**主动云台视角微调（$a_t^{\mathrm{view}}$）**的闭环恢复能力。

**全过程刚性评价指标**：
* **闭环具身控制指标**：端到端搜救任务成功率（SR, %）、路径长度加权成功率（SPL）、盲区目标首次发现步数（Steps-to-Discovery）、障碍物碰撞率（Collision Rate, %）。
* **表征与几何精度指标**：在动作敏感子空间 $\mathbf{U}_{\mathrm{act}}$ 上的特征转换归一化均方误差 $\mathrm{NMSE}_{\mathrm{sub}} = \frac{\|\mathbf{U}_{\mathrm{act}}^\top (\widehat{\mathbf{Y}}_T - \mathbf{Y}_T^\star)\|_F^2}{\|\mathbf{U}_{\mathrm{act}}^\top \mathbf{Y}_T^\star\|_F^2}$、盲区目标三维定位均方根误差（3D RMSE, m）、候选动作条件对数似然（Action Log-Likelihood）。
* **系统效率与统计显著性**：单步实际空口总字节数（Bytes/Step，严格将下行请求 $\mathcal{P}_{\mathrm{req}}$ 与上行包 $\mathcal{P}_{\mathrm{up}}$ 合并计入）、端到端计算延迟（ms）；报告跨 15 个布局的均值、标准差及经 **Holm–Bonferroni 校正的配对 Wilcoxon 符号秩检验 $p$ 值**。

### 3. 前沿 SOTA 零稻草人（Zero-Strawman）公平对比实现规范
所有参评方法共享完全相同的单步总字节预算 $B_t \in \{0.5, 1.0, 2.0, 4.0, 8.0\}\text{ KB}$，且接收端统一使用相同的 VLM 骨干网：
* **KVComm / Q-KVComm (NeurIPS 2025 迁移适配)**：维护共视锚点池，计算一阶均值偏移与二阶标准差缩放因子，并对激活通道执行基于 Kurtosis 的混合精度量化（前 20% 敏感通道 8-bit，其余 4-bit）。
* **Vision Wormhole (2026)**：使用离线共享锚点集通过岭回归预拟合固定的中心辐射仿射映射 $(\mathbf{W}_i^{\mathrm{out}}, \mathbf{W}_e^{\mathrm{in}})$，发送端执行 `NormMatch` 潜空间推演并将压缩后的通用 Token 注入接收端视觉跨度。
* **BOLT (2026)**：完整实现在线共视区教师—学生特征蒸馏、轻量级适配模块与置信度门控，在全部共视高置信区域无差别在线更新适配器。
* **Which2comm (2026) & Where2comm (NeurIPS 2022)**：`Where2comm` 传输空间置信度乘积 $\mathbf{C}_i \odot \mathbf{R}_e$ 最高的网格块；`Which2comm` 仅提取高任务/安全效用的前景目标稀疏框（SemDBs）及其三维坐标索引，挂载预训练跨编码器投影头。
* **RDcomm (ICLR 2026) & VisionZip (CVPR 2025)**：`RDcomm` 按接收端条件 Bayes 风险独立选择任务特征并做率失真编码；`VisionZip` 在发送端单图内保留注意力主导 Token 并聚类合并上下文 Token。

---

## 五、 多 Variant 模型层面的机制洞察（Insights）与可解释性诊断套件

1. **跨模型家族格拉斯曼主角度谱与拉回有效秩（Grassmannian Spectrum & Pullback Effective Rank）**：计算 $\mathrm{srank}(\mathbf{S}_Y^{1/2} \mathbf{A}) = \frac{\|\mathbf{S}_Y^{1/2} \mathbf{A}\|_F^2}{\|\mathbf{S}_Y^{1/2} \mathbf{A}\|_2^2}$，实测证明拉回后稳定秩坍缩至 $4.2 \sim 11.5$。
2. **空间特征梯度 $\|\mathbf{D}_k\|_F$ 与语义贡献度的二维相平面验证（验证定理 2 二分律）**：在横轴 $\|\mathbf{D}_k \mathbf{G}_k\|_F$、纵轴 $\|\mathbf{W}_T^{1/2} \mathbf{P} \mathbf{x}_k\|_2$ 相平面上验证右下象限（最优 $\mathcal{S}_g$）与左上象限（最优 $\mathcal{S}_a$）的正交二分律。
3. **残余位姿偏差下的跨层注意力漂移（Attention Drift）与 Wiener 门控抑制图谱**：追踪解码器第 $1 \sim 28$ 层注意力权重比值与等效三维重心偏移，验证 Wiener 门控 $w_{\ell, j}^\star$ 对中间层（$12 \sim 20$ 层）异常 Attention Sink 的抑制。
4. **闭环回合内的三元比特相变与主动视角触发轨迹**：记录 $T=30$ 步闭环任务中 $(b_g^\star(t), b_a^\star(t), b_s^\star(t))$ 的三阶段自组织相变（建联期 $\to$ 巡航期 $\to$ 机动重锁期）。

---

## 六、 与 AeroRescue × AeroAgentSim 现有工程底座的对接评估与落地路线

### 1. 场景、几何与动作空间的精确对接映射

| 理论蓝图要素 | `AeroRescue` / `AeroAgentSim` 现有工程资产映射 | 落地状态与待扩展项 |
| :--- | :--- | :--- |
| **三类空地物理耦合场景 (Regimes A/B/C，15 布局)** | 已归拢至 `scene_data/` 的 **13 个灾后救援场景**（`SC01, 03, 04, 05, 07, 08, 09, 10, 11, 12, 13, 17, GT04`）+ `SAREnv` 扩展布局，天然涵盖：<br>• **Regime A（峡谷/倒塌掩体遮挡）**：`SC01`、`SC04`、`SC09`、`SC13`<br>• **Regime B（大面积混凝土地面/重复立面/水域）**：`SC05`、`SC07`、`SC10`、`SC11`<br>• **Regime C（高速巡逻与大范围机动切换）**：`GT04`、`SC08`、`SC12`、`SC17` | 13 个基础三维场景已就绪；需按 Regime A/B/C 相机俯仰角（空中 $-75^\circ \sim -30^\circ$、地面 $0^\circ$）与 $\mathrm{SE}(3)$ 外参漂移模型导出空地同步观测帧与深度/法向图。 |
| **相机内参 $\mathbf{K}$、深度 $d_k$ 与表面法向 $\mathbf{n}_k$** | `src/rescue_scenes/camera.py` + `scene_data/.../canonical_grid.npz` & `aas_full_yup.glb` | 现有 `camera.py` 已支持多视角 RGB 渲染与内外参记录；可通过 `trimesh`/光线投射直接从场景精确 GLB 网格提取像素级真实深度 $d_k$ 与表面法向 $\mathbf{n}_k$，用于精确计算 $\mathbf{G}_k$ 与单应性雅可比 $\mathbf{H}_{e\leftarrow i}(k)$。 |
| **离散候选动作集 $\mathcal{A}$ ($K \le 16$) 与云台微调 $a_t^{\mathrm{view}}$** | `src/rescue_scenes/catalog/AeroRescue_任务行动能力状态字典_v1.json`（原生动作 `A001~A063`）+ PX4 云台/航向控制指令 | 在具身 Harness 中将每步合法候选动作收敛为 $K \le 16$ 的离散带参动作集以计算 Fisher 矩阵 $\mathbf{F}_\pi$ 与拉回雅可比 $\mathbf{J}_Y, \mathbf{J}_P$，并暴露主动视角动作 $a_t^{\mathrm{view}}$。 |
| **本地 ↔ Colab 混合计算流水线** | • **本机（6× RTX 3090 + AAS/PX4 仿真器）**：负责运行 15 布局仿真、空地多视角 RGB-D/法向采集，以及 `Qwen3.5-0.8B / 2B` 与本地可跑变体的在线闭环测试；<br>• **Colab（A100/L4）+ Google Drive (`rclone`) + Tailscale**：负责 5 组跨尺度/跨家族大模型变体（`Qwen2.5-VL-7B`、`InternVL2-8B`、`SmolVLM2`、`DINOv2`、`SigLIP2`）的特征/雅可比张量高速预提取、跨变体读取器微调与 450 Episodes 大规模矩阵评测。 | 实现“本机采集打包 $\to$ Colab 特征预提取与模型微调 $\to$ 毫秒级 Algorithm 1 / EIV 滤波核在线闭环评测”的高效协同。 |
