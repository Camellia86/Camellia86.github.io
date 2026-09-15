+++
title = 'Itô MeanFlow 能否打破生成平均化？'
date = 2026-09-03
author = 'Zixiang Ni'
description = '从 pixel MeanFlow 到随机转移核的简明推导，以及 Itô 分支为何可能帮助单步生成器摆脱条件平均化。'
tags = ['Generative Modeling', 'MeanFlow', 'Stochastic Differential Equations']
draft = false
isCJKLanguage = true

[params]
subtitle = '从单个确定性终点到清晰分支的分布'
math = true
contentLanguage = 'zh-CN'
languageLabel = '中文'
translation = '/blog/ito-meanflow-generative-averaging'
+++

单步生成器将一条长轨迹压缩为一次网络求值。这种压缩能力很强，但确定性预测可能会把几种合理的后续演化合并为一个条件平均值。本文探讨一种可能的前进方向：保留 pixel MeanFlow（pMF）的有限时间视角，用 Itô 扩散替代确定性轨迹，并让网络摊销整个**转移核**的计算，而非仅仅预测单个终点。

由此得到的想法可以简单地表述为：

> **pMF 将一次生成旅程压缩为一次前向传播。Itô MeanFlow 则有可能将从同一状态出发的一族旅程压缩为一次随机前向传播。**

## 先给出答案

主要结论如下：

1. **pMF 学习有限时间输运。** 它将类图像的网络输出转换为平均速度，使噪声能够一步到达数据。
2. **确定性的条件目标可能导致平均化。** 当一个状态允许几种合理的未来时，平方损失回归会选择它们的条件均值。
3. **Itô 增量会产生新的分支。** 扩散项的条件均值为零、条件方差为正，因此轨迹在经过同一状态之后可以分离。
4. **MeanFlow 的随机对应形式是一个算子恒等式。** Itô 生成元描述局部漂移与扩散；转移算子描述它们在有限时间内的作用。
5. **完整转移核是保留多模态性的对象。** 随机输入网络可以采样出一个清晰的分支，同时与均值动力学、局部协方差以及双时间复合保持一致。

这启发了一个候选模型，我将其称为 **Branch MeanFlow**：

{{< math >}}
\boxed{
D_\theta(z,r,t,\xi)
\sim
\operatorname{Law}(Z_r\mid Z_t=z),
\qquad
\xi\sim\mathcal N(0,I).
}
{{< /math >}}

当前状态 $z$ 确定了共享的上下文。辅助变量 $\xi$ 选择其中一个可能的分支。

---

## 用四个方程理解 pMF

设干净图像为 $X\sim p_{\mathrm{data}}$，令 $\varepsilon\sim\mathcal N(0,I)$，并定义线性插值

{{< math >}}
Z_t=(1-t)X+t\varepsilon,
\qquad t\in[0,1].
{{< /math >}}

对于一个训练样本对，其速度为

{{< math >}}
Y=\varepsilon-X.
{{< /math >}}

一个仅观测 $(Z_t,t)$ 并以平方损失训练的网络，会识别出后验速度

{{< math >}}
\boxed{
v^\star(z,t)
=
\mathbb E[\varepsilon-X\mid Z_t=z].
}
{{< /math >}}

Flow Matching 学习这一瞬时场。MeanFlow 则学习整个区间上的平均速度。沿着经过 $Z_t=z$ 的概率流轨迹，定义

{{< math >}}
u^\star(z,r,t)
=
\frac{1}{t-r}
\int_r^t
v^\star(Z_s,s)\,ds.
{{< /math >}}

由此立即得到终点：

{{< math >}}
\boxed{
Z_r
=
z-(t-r)u^\star(z,r,t).
}
{{< /math >}}

pMF 将平均速度封装为一个类图像场

{{< math >}}
D^\star(z,r,t)
=
z-t\,u^\star(z,r,t).
{{< /math >}}

在单步边界 $r=0$、$t=1$ 且 $Z_1=\varepsilon$ 时，

{{< math >}}
D_\theta(\varepsilon,0,1)
\approx
Z_0.
{{< /math >}}

因此，实际流程为：

{{< math >}}
\boxed{
\text{噪声}
\longrightarrow
\text{类图像预测}
\longrightarrow
\text{平均输运}
\longrightarrow
\text{单步终点}.
}
{{< /math >}}

完整的 pMF 方法包含基于 JVP 的一致性、辅助目标、感知损失，以及精心采样的双时间对。对于本文所关注的问题，它最核心的贡献是对确定性终点进行双时间摊销。

---

## 生成平均化从何而来

假设在相同条件 $z$ 下，存在两个同样合理的清晰输出：

{{< math >}}
p(X\mid z)
=
\frac12\delta_{x_1}
+
\frac12\delta_{x_2}.
{{< /math >}}

平方损失下的最优确定性预测为

{{< math >}}
\boxed{
g^\star(z)
=
\mathbb E[X\mid z]
=
\frac{x_1+x_2}{2}.
}
{{< /math >}}

两个不同的模态变成了一个折中结果。在图像中，这可能表现为纹理混合或细节变模糊；在视频中，它可能表现为一个平均化的未来；在逆问题中，它可能把几种有效的重建结果隐藏在一个代表性答案背后。

pMF 已经从初始噪声中获得了全局多样性。尚待挖掘的机会在于**状态固定之后的条件多样性**。如果几种后续演化共享同一个中间表示，确定性终点就只会为该状态提供一种后续演化。随机转移则可以让这个状态再次分支。

---

## Itô 增量打开分支

用 Itô SDE 替代确定性 ODE：

{{< math >}}
\boxed{
dZ_s
=
b(Z_s,s)\,ds
+
\Sigma(Z_s,s)\,dW_s,
}
{{< /math >}}

并记

{{< math >}}
a(z,s)=\Sigma(z,s)\Sigma(z,s)^\top.
{{< /math >}}

这里相关的非光滑对象是**样本路径**。布朗路径连续，并且几乎必然处处不可微，因此通常的路径速度要让位于随机增量与二次变差。图像集合仍然可以被视为环境像素空间中的一个经验性软流形。经典 Itô 公式使用属于 $C^{1,2}$ 的网络或测试函数；SiLU 等光滑激活函数提供了一条自然的实现路径。

在 $[r,t]$ 上积分，得到

{{< math >}}
Z_t-Z_r
=
\int_r^t b(Z_s,s)\,ds
+
\int_r^t\Sigma(Z_s,s)\,dW_s.
{{< /math >}}

对于鞅增量

{{< math >}}
M_{r,t}
=
\int_r^t\Sigma(Z_s,s)\,dW_s,
{{< /math >}}

有

{{< math >}}
\mathbb E[M_{r,t}\mid\mathcal F_r]=0,
{{< /math >}}

而

{{< math >}}
\mathbb E[
M_{r,t}M_{r,t}^{\top}
\mid\mathcal F_r
]
=
\mathbb E\left[
\int_r^t a(Z_s,s)\,ds
\middle|\mathcal F_r
\right].
{{< /math >}}

均值增量遵循漂移，而各个样本则按照 $a$ 展开。这就是表示能力上的关键增益：**同一个起始状态可以在区间内获得新的随机性**。

### 保持 pMF 的边缘分布路径

概率流 ODE 按照下式演化密度 $p_t$：

{{< math >}}
\partial_t p_t
=
-\nabla\cdot(vp_t).
{{< /math >}}

SDE 则按照 Fokker–Planck 方程演化该密度：

{{< math >}}
\partial_t p_t
=
-\nabla\cdot(bp_t)
+
\frac12\nabla\cdot\nabla\cdot(ap_t).
{{< /math >}}

在密度光滑且为正的内部时刻，通过为扩散配上相容的漂移，SDE 可以与原始概率流 ODE 具有相同的边缘分布：

{{< math >}}
\boxed{
b_i
=
v_i
+
\frac{1}{2p_t}
\sum_j
\partial_{z_j}(a_{ij}p_t).
}
{{< /math >}}

等价地，

{{< math >}}
b
=
v
+
\frac12\nabla\cdot a
+
\frac12a\nabla\log p_t.
{{< /math >}}

这个公式带来了一个很有吸引力的设计空间：保持所期望的从噪声到数据的边缘分布旅程，同时选择 $a(z,t)$ 来控制条件分支沿途如何展开。扩散矩阵可以充当一张学得的**歧义图**。

---

## MeanFlow 的 Itô 版本

对于状态函数 $f$，Itô 公式给出

{{< math >}}
df(Z_s)
=
\left[
b\cdot\nabla f
+
\frac12a:\nabla^2f
\right]ds
+
\nabla f^\top\Sigma\,dW_s.
{{< /math >}}

定义 Itô 生成元

{{< math >}}
\boxed{
\mathcal L_s f
=
b(\cdot,s)\cdot\nabla f
+
\frac12a(\cdot,s):\nabla^2f.
}
{{< /math >}}

一阶项承载漂移。Hessian 项记录二次变差的系统性作用。鞅项则承载样本特有的分支信息。

对于马尔可夫扩散，定义有限时间转移算子

{{< math >}}
(P_{r,t}f)(z)
=
\mathbb E[f(Z_t)\mid Z_r=z].
{{< /math >}}

它在该区间上的平均生成元为

{{< math >}}
\boxed{
\overline{\mathcal L}_{r,t}f
=
\frac{P_{r,t}f-f}{t-r}.
}
{{< /math >}}

Dynkin 公式给出

{{< math >}}
\overline{\mathcal L}_{r,t}f
=
\frac1{t-r}
\int_r^t
P_{r,s}\mathcal L_s f\,ds.
{{< /math >}}

对右端点求导：

{{< math >}}
\boxed{
P_{r,t}\mathcal L_t f
=
\overline{\mathcal L}_{r,t}f
+
(t-r)\partial_t
\overline{\mathcal L}_{r,t}f.
}
{{< /math >}}

这与 MeanFlow 具有相同的有限时间结构：

{{< math >}}
\boxed{
\text{瞬时生成元}
=
\text{平均生成元}
+
\text{区间长度修正项}.
}
{{< /math >}}

MeanFlow 关心一个点沿着一条确定性曲线会走向哪里。Itô MeanFlow 关心一个可观测量在随机轨迹分布下如何变化。

### 条件均值这一特殊情形

令

{{< math >}}
M(z,r,t)
=
\mathbb E[Z_t\mid Z_r=z]
=
z+(t-r)u^+(z,r,t).
{{< /math >}}

后向 Kolmogorov 方程

{{< math >}}
\partial_r M+\mathcal L_rM=0,
{{< /math >}}

意味着

{{< math >}}
\boxed{
b(z,r)
=
u^+(z,r,t)
-
(t-r)
\left[
\partial_r u^+(z,r,t)
+
\mathcal L_r u^+(z,r,t)
\right].
}
{{< /math >}}

这是一个条件均值形式的 Itô–MeanFlow 恒等式。其生成元包含二阶修正项 $\tfrac12a:\nabla^2u^+$，但其输出仍然只是一个条件均值。完整的随机信息存在于转移核之中。

---

## 从均值终点到分支核

全协方差公式指出了分支所承载的信息：

{{< math >}}
\operatorname{Cov}(Z_t)
=
\operatorname{Cov}(
\mathbb E[Z_t\mid Z_r]
)
+
\mathbb E[
\operatorname{Cov}(Z_t\mid Z_r)
].
{{< /math >}}

均值终点承载第一项。随机转移还会承载第二项。

Itô 推导所启发的层次结构如下：

| 学习对象 | 保留的信息 | 生成行为 |
|---|---|---|
| 坐标均值 | 条件中心 | 一个代表性终点 |
| 线性与二次可观测量 | 均值与分支方向 | 局部校准的随机散布 |
| 丰富的可观测量族或完整转移核 | 条件分布 | 多个连贯的模态 |

坐标函数恢复漂移。二次函数揭示 $a$。足够丰富的测试函数集合能够刻画整个条件转移分布。

这促使我们考虑一个随机输入的有限时间采样器：

{{< math >}}
\boxed{
Z_t
=
G_\theta(z,r,t,\xi),
\qquad
\xi\sim\mathcal N(0,I),
}
{{< /math >}}

其中

{{< math >}}
G_\theta(z,r,t,\xi)
\sim
p(Z_t\mid Z_r=z).
{{< /math >}}

在反向生成时钟下，pMF 风格的边界变为

{{< math >}}
\boxed{
D_\theta(z,0,t,\xi)
\sim
K^\leftarrow_{t,0}(z,\cdot)
:=
\operatorname{Law}(Z_0\mid Z_t=z).
}
{{< /math >}}

一次调用仍然跨越整个区间。不同的 $\xi$ 值从同一条件下选择不同的终点。

### 一种候选训练方案

Branch MeanFlow 可以结合四类约束：

1. **均值动力学一致性。** 对 $\xi$ 求平均后遵循 MeanFlow 或 Itô–MeanFlow 恒等式，从而保留共享的输运骨架。
2. **局部协方差一致性。** 短区间内的分支协方差与扩散几何 $a(z,t)$ 相匹配。
3. **分布层面的匹配。** MMD、对抗特征匹配、条件流匹配或条件分数匹配，促使不同样本覆盖目标条件分布。
4. **双时间复合一致性。** 一次长距离随机跳跃与两次较短跳跃的复合在分布上保持一致，对应于 Chapman–Kolmogorov 方程。

pMF 的类图像参数化在这里仍然有用。每个分支都可以直接落入图像形状的输出空间，而感知特征在该空间中促进清晰结构的形成。漂移学习各分支的共性；扩散学习它们应当在哪里分离；$\xi$ 决定哪一个分支出现。

---

## 为什么这可能打破平均化

最有希望的作用，是对生成任务进行一种新的分解：

{{< math >}}
\boxed{
\text{共享结构}
+
\text{歧义几何}
+
\text{分支选择}
}
{{< /math >}}

漂移可以承载大尺度的语义运动。扩散矩阵可以打开与不确定纹理、姿态、颜色或未来运动相关的方向。潜在的分支变量可以将这些方向转化为各自清晰的样本。

这可能首先在固定观测下歧义显著的任务中发挥作用：

- 超分辨率、着色与图像复原；
- 逆问题的后验采样；
- 视频、语音与运动预测；
- 未来确实具有随机性的科学系统。

同样的机制也可能帮助无条件单步生成。一个中间状态可以重新展开为几种局部后续演化，恢复单个确定性回归目标倾向于压缩掉的高频细节与稀有模态。模型将在一次求值中同时完成长程输运与分支选择。

最清晰的早期实验，是将一个具有歧义的条件生成基准与四项测量配对：单个样本的清晰度、条件覆盖度、分支校准以及单步延迟。一个合成的双模态问题可以验证该机制；超分辨率或着色可以揭示该机制能否扩展到感知歧义。

如果这一想法有效，其核心进展将不止于给快速生成器添加噪声。它将把单步生成从**终点预测**转变为**转移分布采样**。

---

## 超越经典 Itô 扩散

转移核视角也指向更广泛的情形：

- 在光滑流形上，切向扩散与 Laplace–Beltrami 生成元可以塑造内禀分支。
- 在非光滑图像集合附近，弱生成元与学得的类图像势函数可以替代精确的几何约束。
- 跳跃过程可以通过 Lévy 生成元表示离散的分支变化。
- 非马尔可夫问题可以使用以历史为条件的转移核，使整个过去决定未来的分布。

在这些情形中，始终适用的想法是有限时间随机摊销：学习轨迹可能到达何处的分布，同时保留足够的一致性，使这些轨迹能够复合。

> **令人期待的图景是：pMF 提供单步输运骨架，而 Itô 动力学提供分支坐标。二者的结合可能使快速生成器对每个样本给出一个清晰答案，并在不同样本之间呈现完整的答案族。**

---

## 引用

如果本文对您的研究有帮助，请按以下方式引用：

> Zixiang Ni. “Can Itô MeanFlow Break Generative Averaging?” *Camellia86*, September 3, 2026. https://camellia86.github.io/blog/ito-meanflow-generative-averaging/

    @misc{ni2026itomeanflow,
      author = {Ni, Zixiang},
      title  = {Can Itô MeanFlow Break Generative Averaging?},
      year   = {2026},
      month  = sep,
      url    = {https://camellia86.github.io/blog/ito-meanflow-generative-averaging/}
    }

---

## 参考文献

- Yiyang Lu 等，[*One-step Latent-free Image Generation with Pixel Mean Flows*](https://arxiv.org/abs/2601.22158)，arXiv:2601.22158，2026。
- Zhengyang Geng 等，[*Mean Flows for One-step Generative Modeling*](https://arxiv.org/abs/2505.13447)，arXiv:2505.13447，2025。
- Michael S. Albergo、Nicholas M. Boffi 和 Eric Vanden-Eijnden，[*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://arxiv.org/abs/2303.08797)，JMLR，2025。
- Yang Song 等，[*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456)，ICLR，2021。
- Bernt Øksendal，*Stochastic Differential Equations: An Introduction with Applications*，Springer，第 6 版，2003。
