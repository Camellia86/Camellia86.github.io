+++
title = 'From Mean Velocity in pMF to Stochastic Branches'
date = 2026-09-02
description = 'A detailed path from pixel MeanFlow to an Itô-inspired model of stochastic generative branches.'
tags = ['Generative Modeling', 'MeanFlow', 'Stochastic Differential Equations']
draft = false

[params]
subtitle = 'How Itô Dynamics Could Move Beyond Generative Averaging'
math = true
+++

Recent research work led me to read *Pixel MeanFlow* (pMF) from Kaiming He and collaborators. The paper gave me several useful ideas and helped move my own research forward. I then explored an extension of pMF with stochastic increments and found a few intriguing connections. The first half of this post develops pMF in detail and from first principles. Readers who already know the method can jump directly to Section 8.

## Overview

This post has two parts:

- The first half reconstructs pMF, moving from Flow Matching and posterior velocity to the MeanFlow identity and one-step generation.
- The second half follows a more open-ended question: after adding Brownian increments to the trajectory, can a model move beyond conditional means and learn multiple generative branches from the same state?

> **Research thread.** The first half organizes pMF, MeanFlow, and the relevant implementation details. The second treats “Itô-MeanFlow” as a research direction that can continue to develop. The derivation already reveals a hierarchy connecting conditional means, stochastic branches, and full transition kernels, which suggests concrete training objectives and experiments.

---

## 0. The Main Conclusions First

The post can be summarized in six points:

1. The velocity of an individual training pair is $\varepsilon-X$, whereas the network can identify the conditional mean velocity $\mathbb E[\varepsilon-X\mid Z_t=z]$.
2. Flow Matching learns the velocity “right now”; MeanFlow learns the average velocity required to cross an entire time interval.
3. pMF first predicts a generalized denoising field that looks more like an image, then converts it back into velocity.
4. Itô integration describes stochastic futures that continue to branch after conditioning on the same current state. Path roughness, the geometry of the image set, and the smoothness of the network function belong to three separate levels.
5. Itô noise changes the quadratic variation and transition law of a path. A diffusion strength paired with a compatible drift can introduce new conditional randomness while preserving a prescribed family of marginals.
6. Brownian trajectories push MeanFlow from a pointwise identity toward amortized finite-time transitions: conditional means produce averaging, averaged generators retain local stochastic structure, and full transition kernels create a route beyond averaging.

---

> **Part I: A Detailed Guide to pMF**

A conventional diffusion model resembles a navigation system that guides noise toward an image through many small steps. pMF tries to compress the whole route into one instruction: “Go straight to the destination.”

Two aspects make this difficult:

1. It aims to use only **one network forward pass**, amortizing repeated model evaluations into the network.
2. It aims to generate directly in **pixel space**, without a pretrained latent tokenizer such as a VAE.

The most elegant idea in pMF fits into one sentence:

> **The network output and the training loss can live in different spaces.**

The network directly outputs a quantity that resembles a clean image—an $x$-prediction. The main dynamical loss follows the chain “image output → average velocity → instantaneous velocity” and is evaluated in velocity space. Because the direct output has image structure, perceptual losses can be added naturally during training.

---

## 1. Notation

The pMF paper uses $\mathbf x$ both for a clean image and for the generalized denoising field. To keep the two concepts separate, this post uses the following notation.

| Symbol | Mathematical definition | Intuition |
|---|---|---|
| $X$ | $X\sim p_{\mathrm{data}}$ | A real clean image |
| $\varepsilon$ | $\varepsilon\sim\mathcal N(0,I)$ | Gaussian noise |
| $t$ | $t\in[0,1]$ | Noise time; $t=0$ is data and $t=1$ is noise |
| $r$ | $0\le r\le t$ | Left endpoint of the average-velocity interval |
| $Z_t$ | $(1-t)X+t\varepsilon$ | Linear stochastic interpolation between data and noise |
| $Y$ | $\varepsilon-X$ | Sample-level velocity for one training pair |
| $v^\star(z,t)$ | $\mathbb E[Y\mid Z_t=z]$ | The marginal instantaneous velocity field identifiable by the network |
| $Z_\tau^{z,t}$ | ODE trajectory satisfying $Z_t^{z,t}=z$ | The probability-flow trajectory through the current state $z$ |
| $u^\star(z,r,t)$ | Average velocity over $[r,t]$ | The average velocity required to cross the whole interval in one step |
| $D^\star(z,r,t)$ | $z-tu^\star(z,r,t)$ | The generalized denoising image field in pMF |
| $D_\theta$ | $\operatorname{net}_\theta(z,r,t)$ | The image-like direct network output |
| $u_\theta$ | $(z-D_\theta)/t$ | Average velocity converted from the image output |
| $\mathcal D_t^q u$ | $\partial_tu+J_zu\,q$ | Total derivative along direction $q$ |
| $V_\theta$ | $u_\theta+(t-r)\operatorname{sg}[\mathcal D_t^{v_{\mathrm{jvp}}}u_\theta]$ | Instantaneous velocity reconstructed from the average velocity |
| $\operatorname{sg}$ | stop-gradient | The value participates in the computation while gradients stop at this branch |
| JVP | Jacobian-vector product | A directional derivative computed without constructing the full Jacobian |

The central distinction is between

{{< math >}}
Y=\varepsilon-X,
{{< /math >}}

which belongs to one random training pair, and

{{< math >}}
v^\star(z,t)=\mathbb E[Y\mid Z_t=z],
{{< /math >}}

which averages over all pairs capable of producing $z$ after observing $Z_t=z$. The network sees only $z$ and $t$, so squared loss drives its optimal prediction toward the latter.

---

## 2. Starting from Flow Matching: What Does the Network Fit?

First sample a real image and Gaussian noise. To keep the main argument clean, begin with the unit-variance setting:

{{< math >}}
X\sim p_{\mathrm{data}},
\qquad
\varepsilon\sim\mathcal N(0,I).
{{< /math >}}

This normalization is for exposition. The pMF v3 implementation scales the noise with resolution; for example, its $512\times512$ configuration uses $\sigma=2$, corresponding to covariance $4I$. When $\varepsilon\sim\mathcal N(0,\sigma^2I)$, the noise-variance terms in the Tweedie and score identities below acquire a factor of $\sigma^2$.

Interpolate linearly between data and noise:

{{< math >}}
\boxed{Z_t=(1-t)X+t\varepsilon}.
{{< /math >}}

Then

{{< math >}}
Z_0=X,
\qquad
Z_1=\varepsilon.
{{< /math >}}

Differentiating one pair with respect to $t$ gives

{{< math >}}
\frac{dZ_t}{dt}=\varepsilon-X.
{{< /math >}}

Thus the sample-level instantaneous velocity is

{{< math >}}
\boxed{Y=\varepsilon-X}.
{{< /math >}}

Standard Flow Matching uses the squared loss

{{< math >}}
\mathcal L_{\mathrm{FM}}
=
\mathbb E\left[
\left\|v_\theta(Z_t,t)-(\varepsilon-X)\right\|^2
\right].
{{< /math >}}

Many different $(X,\varepsilon)$ pairs can produce the same $z$. When a deterministic regressor encounters their conflicting sample-level velocities, squared loss aggregates them into a conditional mean.

The population optimum of squared loss is the conditional expectation:

{{< math >}}
\boxed{
v^\star(z,t)
=
\mathbb E[\varepsilon-X\mid Z_t=z]
}.
{{< /math >}}

More formally, the posterior satisfies

{{< math >}}
p(x,\varepsilon\mid z,t)
\propto
p_{\mathrm{data}}(x)p_{\mathrm{prior}}(\varepsilon)
\,\delta\!\left(z-(1-t)x-t\varepsilon\right),
{{< /math >}}

and therefore

{{< math >}}
v^\star(z,t)
=
\int(\varepsilon-x)\,
p(x,\varepsilon\mid z,t)
\,dx\,d\varepsilon.
{{< /math >}}

The MSE also admits a useful orthogonal decomposition:

{{< math >}}
\begin{aligned}
&\mathbb E\left[\|v_\theta(Z_t,t)-Y\|^2\right]\\
={}&
\mathbb E\left[\|v_\theta(Z_t,t)-v^\star(Z_t,t)\|^2\right]
+
\mathbb E\left[\|Y-v^\star(Z_t,t)\|^2\right].
\end{aligned}
{{< /math >}}

The second term is the posterior uncertainty that remains after conditioning on $Z_t$ and is independent of the model. The sample-level target is $Y$, while the population-level object identifiable under squared loss is the posterior mean field $v^\star$.

---

## 3. Why Does x-Prediction Work?

Starting from the linear interpolation,

{{< math >}}
\varepsilon-X=\frac{Z_t-X}{t}.
{{< /math >}}

The network can first predict a clean-image-like output $d_\theta(Z_t,t)$ and then convert it into a velocity:

{{< math >}}
v_\theta(Z_t,t)
=
\frac{Z_t-d_\theta(Z_t,t)}{t}.
{{< /math >}}

In that case,

{{< math >}}
v_\theta-(\varepsilon-X)
=
\frac{X-d_\theta}{t}.
{{< /math >}}

For fixed $z,t$, minimizing velocity MSE is therefore equivalent to minimizing image-reconstruction MSE weighted by $1/t^2$. Averaging over time yields a time-weighted objective, while ordinary pixel MSE uses a different weighting. Its optimum is

{{< math >}}
\boxed{
d^\star(z,t)=\mathbb E[X\mid Z_t=z]
}.
{{< /math >}}

This is the posterior interpretation of ordinary $x$-prediction: given the current noisy observation, the network returns the posterior mean. Whenever multiple plausible answers share the same condition, averaging enters at this point.

### 3.1 Relation to the Score

In the unit-variance special case $\varepsilon\sim\mathcal N(0,I)$, define

{{< math >}}
s_t(z)=\nabla_z\log p_t(z).
{{< /math >}}

Tweedie’s formula gives

{{< math >}}
\boxed{
\mathbb E[X\mid Z_t=z]
=
\frac{z+t^2s_t(z)}{1-t}
},
\qquad 0<t<1.
{{< /math >}}

It follows that

{{< math >}}
\boxed{
v^\star(z,t)
=
-\frac{z+t\,s_t(z)}{1-t}
}.
{{< /math >}}

Conversely, at the $r=t$ boundary of $x$-prediction, the denoising prediction recovers the score through

{{< math >}}
\boxed{
s_t(z)
=
\frac{(1-t)d^\star(z,t)-z}{t^2}
}.
{{< /math >}}

This relation becomes useful in the Itô extension because reverse-time SDEs typically require the score. As $t\to0$, the factor $1/t^2$ amplifies numerical error, so practical implementations benefit from truncation, reweighting, or a separate parameterization.

---


## 4. MeanFlow: How Can Average Velocity Replace Multistep Integration?

The posterior velocity field defines a probability-flow ODE:

{{< math >}}
\frac{dZ_\tau}{d\tau}
=
v^\star(Z_\tau,\tau).
{{< /math >}}

Fix the current state $Z_t^{z,t}=z$. Along the deterministic ODE trajectory through $z$, define the interval-averaged velocity

{{< math >}}
\boxed{
u^\star(z,r,t)
=
\frac{1}{t-r}
\int_r^t
v^\star(Z_\tau^{z,t},\tau)
\,d\tau
}.
{{< /math >}}

Because velocity integrated over time equals displacement,

{{< math >}}
(t-r)u^\star(z,r,t)
=
z-Z_r^{z,t},
{{< /math >}}

or equivalently,

{{< math >}}
\boxed{
Z_r^{z,t}
=
z-(t-r)u^\star(z,r,t)
}.
{{< /math >}}

Instantaneous velocity is like the speed shown on a car’s dashboard, while average velocity is the speed computed by a navigation system over the entire journey. Flow Matching learns the former. MeanFlow learns the latter, allowing the model to cross a whole interval at once.

### 4.1 The MeanFlow Identity

Starting from

{{< math >}}
(t-r)u^\star
=
\int_r^t v^\star(Z_\tau,\tau)d\tau,
{{< /math >}}

differentiate with respect to the upper endpoint $t$:

{{< math >}}
v^\star
=
u^\star+(t-r)\frac{d}{dt}u^\star.
{{< /math >}}

The field $u^\star$ depends on $t$ explicitly and also implicitly through the state $Z_t$, so

{{< math >}}
\frac{d}{dt}u^\star(Z_t,r,t)
=
\partial_tu^\star
+
J_zu^\star\,v^\star.
{{< /math >}}

Define the directional total derivative

{{< math >}}
\mathcal D_t^q u
=
\partial_tu+J_zu\,q.
{{< /math >}}

Along the theoretical trajectory, set $q=v^\star$. The material derivative is

{{< math >}}
\mathcal D_t^{v^\star}
=
\partial_t+v^\star\cdot\nabla_z.
{{< /math >}}

This yields the central MeanFlow identity:

{{< math >}}
\boxed{
v^\star(z,t)
=
u^\star(z,r,t)
+
(t-r)\mathcal D_t^{v^\star}u^\star(z,r,t)
}.
{{< /math >}}

Training data provide the sample-level instantaneous velocity $\varepsilon-X$, while the MeanFlow identity transfers this local supervision to average motion over an entire trajectory. It forms a bridge from an observable instantaneous target to finite-time transport.

At $r=t$, the interval collapses. Since the original definition contains $1/(t-r)$, the boundary value below is understood through continuous extension:

{{< math >}}
u^\star(z,t,t)=v^\star(z,t).
{{< /math >}}

MeanFlow therefore contains both short-interval Flow Matching and long-interval one-step transport.

---

## 5. pMF: Why Introduce a “Generalized Denoising Image Field”?

A pixel-space velocity mixes image and noise components, and its target can be substantially more dispersed than a clean image, with effective support closer to the full ambient space. Image-like targets, by contrast, exhibit the empirical low-dimensional structure of natural images. The two objects have the same tensor dimensions but potentially very different statistical complexity. pMF therefore dresses the average velocity in a more image-like parameterization:

{{< math >}}
\boxed{
D^\star(z,r,t)
=
z-t\,u^\star(z,r,t)
}.
{{< /math >}}

The coefficient here is $t$, whereas the trajectory displacement uses $t-r$. Thus, in general,

{{< math >}}
D^\star(z,r,t)\neq Z_r^{z,t}.
{{< /math >}}

Using

{{< math >}}
u^\star=\frac{z-Z_r}{t-r},
{{< /math >}}

we obtain

{{< math >}}
D^\star
=
\frac{tZ_r-rz}{t-r}.
{{< /math >}}

This object extrapolates the average motion over $[r,t]$ to time zero, producing an image-like generalized extrapolation. At its two boundaries, it connects a posterior mean to a trajectory endpoint.

### 5.1 The Two Boundaries

When $r=t$, average velocity reduces to instantaneous velocity:

{{< math >}}
D^\star(z,t,t)
=
z-tv^\star(z,t)
=
\mathbb E[X\mid Z_t=z].
{{< /math >}}

This is exactly the posterior mean associated with ordinary $x$-prediction.

When $r=0$,

{{< math >}}
D^\star(z,0,t)
=
z-tu^\star(z,0,t)
=
Z_0^{z,t}.
{{< /math >}}

Here it equals the deterministic endpoint obtained by running the probability-flow ODE from the current state back to the data end.

These are two conceptually distinct objects:

{{< math >}}
\mathbb E[X\mid Z_t=z]
\quad\text{and}\quad
Z_0^{z,t}.
{{< /math >}}

The first is the Bayesian posterior mean under the linear stochastic interpolation; the second is the endpoint of the deterministic ODE induced by the marginal velocity field. The boundaries $r=t$ and $r=0$ select these two objects respectively. They can coincide at the degenerate boundary $t=0$ or under additional special structure.

### 5.2 What Happens for 0 < r < t?

In the interior, $D^\star$ is a generalized image extrapolation. The pMF paper proposes a **generalized manifold hypothesis**: empirically, $D^\star(z,r,t)$ resembles a clean or slightly blurred image more closely than a velocity field and may therefore be easier for a pixel Transformer to model.

This is a modeling hypothesis motivated by experimental observations, and it leaves room for the soft-manifold and stochastic-branch perspectives developed later.

> **Terminology and scope.** This post retains the phrase “generalized manifold hypothesis” from pMF and interprets it statistically as a **low-effective-complexity, soft-manifold, or tubular-concentration hypothesis**. The values of $D^\star(z,r,t)$ look more like natural images and may concentrate near a low-complexity set. The immediate focus is the image-like structure and concentration of the network output. Tangent spaces, curvature, and intrinsic diffusion enter only when the analysis moves to a $C^k$ embedded submanifold.

This distinction also clarifies the range of applicability. The image set may be nonsmooth, have boundaries, or possess only topological or fractal low-complexity structure, while the empirical pMF parameterization can still be useful. A strict study of stochastic motion on a geometric manifold calls for the corresponding smooth structure, tangent spaces, curvature, and a Laplace–Beltrami operator.

---

## 6. How Does the Network Convert an Image Output Back into a Velocity Loss?

pMF lets the network directly output

{{< math >}}
D_\theta(z,r,t)=\operatorname{net}_\theta(z,r,t).
{{< /math >}}

This is the two-time notation used at the paper level. The released v3 backbone is conditioned explicitly on the interval $h=t-r$, while $t$ is used mainly outside the backbone to convert $D_\theta$ into $u_\theta$. The theoretical function signature describes the mathematical object; the backbone inputs reflect a particular implementation.

The output is converted into average velocity through

{{< math >}}
\boxed{
u_\theta(z,r,t)
=
\frac{z-D_\theta(z,r,t)}{t}
}.
{{< /math >}}

Direct use of this expression assumes $t>0$. At the $t=0$ boundary, one can use continuous extension, an explicit boundary condition, or numerical truncation.

Next compute a total derivative along the current trajectory direction:

{{< math >}}
\mathcal D_t^{v_{\mathrm{jvp}}}u_\theta
=
\partial_tu_\theta
+
J_zu_\theta\,v_{\mathrm{jvp}}.
{{< /math >}}

The simplified pseudocode in the paper uses

{{< math >}}
v_{\mathrm{jvp}}=u_\theta(z,t,t).
{{< /math >}}

The official implementation also includes an auxiliary instantaneous-velocity head and uses the conditional velocity $v_c$ as the state direction of the JVP during classifier-free guidance training. The central point remains the same: a JVP requires a direction along which the derivative is taken.

Define the composite instantaneous velocity

{{< math >}}
\boxed{
V_\theta
=
u_\theta
+
(t-r)\operatorname{sg}[\mathcal D_t^{v_{\mathrm{jvp}}}u_\theta]
}.
{{< /math >}}

The model is then trained against a velocity target:

{{< math >}}
\boxed{
\mathcal L_{\mathrm{pMF}}
=
\mathbb E_{t,r,X,\varepsilon}
\left[
\left\|V_\theta(Z_t,r,t)-(\varepsilon-X)\right\|^2
\right]
}.
{{< /math >}}

This is the main pMF pipeline:

{{< math >}}
\boxed{
\text{image-like output}
\;D_\theta
\longrightarrow
\text{average velocity}
\;u_\theta
\longrightarrow
\text{instantaneous velocity}
\;V_\theta
\longrightarrow
\text{velocity loss}
}.
{{< /math >}}

### 6.1 What Does the JVP Compute?

The JVP input direction can be written as

{{< math >}}
(v_{\mathrm{jvp}},0,1).
{{< /math >}}

Therefore,

{{< math >}}
\operatorname{JVP}
=
J_zu_\theta\,v_{\mathrm{jvp}}
+
\partial_tu_\theta.
{{< /math >}}

The zero in the middle means that $r$ remains fixed while differentiating with respect to the upper endpoint $t$. A JVP directly computes the derivative in a chosen direction and avoids constructing the enormous full Jacobian.

### 6.2 Why Does Stop-Gradient Deserve Separate Attention?

At the level of function values, the MeanFlow identity corresponds to

{{< math >}}
V_\theta=u_\theta+(t-r)\mathcal D_t^{v_{\mathrm{jvp}}}u_\theta.
{{< /math >}}

The implementation blocks parameter gradients through the JVP branch, so optimization uses a semi-gradient:

- The numerical value of the JVP enters the composite prediction $V_\theta$, while gradient computation treats it as a constant—a target-like term.
- Parameter gradients stop at the JVP branch.
- This avoids expensive, high-variance mixed parameter–input higher-order derivatives.

It is helpful to separate three objects:

1. the value-level MeanFlow identity;
2. the exact gradient of the full material-derivative loss;
3. the stop-gradient update used in pMF/iMF.

They describe the identity, the full gradient, and the practical optimization update respectively.

### 6.3 A Practical Benefit of Direct Image Output

The main dynamical supervision in pMF lives in velocity space, while training can also include perceptual losses such as LPIPS and ConvNeXt feature distances. Because the network directly outputs an image-like result, training can check both the dynamical accounting and the visual quality of that result.

In the ImageNet $256\times256$ B/16 ablation from the pMF paper, VGG-LPIPS improved the 1-NFE FID from 9.56 to 5.62; adding the ConvNeXt-V2 variant further improved it to 3.53. This is a practical advantage of $x$-prediction over a pure velocity parameterization.

Here, **latent-free** refers to a generative path independent of VAE/tokenizer encoding and decoding. Training can still use frozen pretrained feature networks to compute perceptual losses.

---

## 7. Why Can It Generate in One Step?

At generation time, choose

{{< math >}}
r=0,
\qquad
t=1,
\qquad
Z_1=\varepsilon.
{{< /math >}}

The network then directly outputs

{{< math >}}
D_\theta(\varepsilon,0,1).
{{< /math >}}

When the MeanFlow relation, boundary conditions, and network approximation are sufficiently accurate,

{{< math >}}
D_\theta(\varepsilon,0,1)
\approx
Z_0.
{{< /math >}}

A single forward pass therefore maps noise to the image end.

“One step” means that the endpoint of the entire ODE transport has been explicitly amortized into the network. A forward pass carries the result of the whole journey, far more than a single local Euler update.

### 7.1 What Does the Ideal Population Objective Require under Conceptual Unguided MSE?

The following orthogonal decomposition corresponds to the conceptual unguided MSE form of Eq. (12) in the paper: the target is $Y=\varepsilon-X$, and $V_\theta$ depends only on observable inputs. The released training objective additionally uses a predicted CFG target, truncation in $t$, adaptive residual normalization, an auxiliary $v$-head, and perceptual losses.

For any $V_\theta$, the velocity MSE can be written as

{{< math >}}
\mathcal L_{\mathrm{pMF}}
=
\mathbb E\left[
\|V_\theta(Z_t,r,t)-v^\star(Z_t,t)\|^2
\right]
+C,
{{< /math >}}

where

{{< math >}}
C
=
\mathbb E\left[
\|Y-v^\star(Z_t,t)\|^2
\right]
{{< /math >}}

is independent of the model.

Under infinite data, an appropriate function space, and ideal optimization, the target fixed point should satisfy

{{< math >}}
\boxed{
u^\star
+
(t-r)
\left(
\partial_tu^\star
+
J_zu^\star v^\star
\right)
=
v^\star
}.
{{< /math >}}

This population equation describes the structure of the ideal target. Practical training realizes a more specific optimization procedure through stop-gradient, an approximate JVP direction, CFG, an auxiliary velocity head, and adaptive weighting.

The equation also clarifies the boundary roles of $D_\theta$. At $r=t$,

{{< math >}}
D_\theta(z,r,t)=\mathbb E[X\mid Z_t=z].
{{< /math >}}

For $r<t$, $D_\theta$ becomes a two-time field constrained by the MeanFlow differential relation.

### 7.2 What Do the Original Ablations Establish?

The evidence in the pMF paper is revealing when interpreted within its experimental setting:

- At $64\times64$, where each patch has relatively low dimension, $x$-prediction and $u$-prediction obtain similar 1-NFE FIDs: 3.80 and 3.82.
- At $256\times256$, where the patch dimension rises to 768, $x$-prediction obtains 9.56 while $u$-prediction degrades to 164.89.
- In the ImageNet B/16 ablation, covering the full triangular region $0\le r\le t$ is essential. Sampling only $r=t$, only $r=0$, or only the two boundaries gives 194.53, 389.28, and 106.59 respectively; the full region gives 3.53.
- After scaling the model and training, pMF v3 reports class-conditional, pixel-space, 1-NFE FIDs of 2.22 at ImageNet $256\times256$ and 2.48 at $512\times512$.

Within the high-dimensional pixel-space ImageNet setting studied in the paper, these results show that image-like parameterization, training over the full two-time region, and perceptual losses are all crucial. They also provide a sharp starting point for exploring other datasets and conditional-generation tasks.

---

> **Part II: Can Itô Dynamics Unfold an “Average Answer” into Stochastic Branches?**


## 8. Why Introduce Itô Calculus, and Which Kind of Nonsmoothness Does It Address?

The motivation is conditional randomness. Original pMF obtains global diversity from the random initial noise $\varepsilon$. An SDE goes further by describing uncertainty that remains **after conditioning on the same current state or observation**. The same low-resolution image can support several sets of high-resolution details; the same history can lead to several futures; and a stochastic physical system can keep receiving fresh perturbations from the same macroscopic state.

Several distinct forms of “nonsmoothness” appear here:

| Object | Allowed to be nonsmooth? | Consequence for the formulas in this post |
|---|---|---|
| Sample path $s\mapsto Z_s(\omega)$ | Yes. Brownian sample paths are almost surely continuous and nowhere differentiable | This is the roughness handled by Itô calculus; semimartingale decomposition and quadratic variation replace ordinary path derivatives |
| The image set $M\subset\mathbb R^d$ in pMF | Yes. It may be a nonsmooth low-complexity set | The generalized manifold hypothesis remains an empirical parameterization motive; tangent spaces and curvature require smooth manifold structure |
| Ambient state space | Taken here to be the smooth space $\mathbb R^d$ | A process constrained exactly to a lower-dimensional manifold is usually formulated with at least $C^2$ geometric structure in classical manifold Itô theory |
| Fields such as $u_\theta,D_\theta,b,\Sigma$ | Smoothness follows the formula being used | The original MeanFlow JVP requires first derivatives; the Itô Hessian formula additionally uses second spatial derivatives |
| Marginal law $\mu_t$ | It may be singular | The weak Fokker–Planck identity covers general measures; a smooth positive density additionally supports $\nabla\log p_t$ |

> **Does MeanFlow allow these cases?** When the data or image set has nonsmooth structure, ordinary pMF/MeanFlow can still learn a field in the ambient space. ReLU-style networks support JVPs almost everywhere, while the classical derivation can be interpreted through first derivatives or weak derivatives. Once Brownian trajectories enter, the pointwise identity naturally lifts to an Itô generator, a conditional-mean identity, or a transition-kernel formulation.

The pointwise MeanFlow identity relies on a finite-variation ODE trajectory and the ordinary chain rule. Under standard regularity conditions, the ODE path is absolutely continuous and the first directional derivative of $u$ is meaningful along the path. Piecewise-smooth networks such as ReLU models admit JVPs almost everywhere in engineering practice, and a classical derivation can be understood through first or weak derivatives. After the Itô derivation introduces a Hessian trace, smooth parameterizations such as SiLU or softplus yield the cleanest formulas. Piecewise-linear activations lead instead toward Itô–Tanaka formulas, local time, and weak derivatives.

Brownian paths have quadratic variation, so finite-time stochastic increments take the place of pointwise velocity. The idea of “amortizing an entire interval with a two-time object” then expands into three levels: conditional means, transition operators, and full transition kernels.

### 8.1 Conditions behind the Classical Formulas

| Result used | Sufficient conditions used here |
|---|---|
| Classical Itô formula in the ambient space | $Z_t$ is a continuous semimartingale; $\Sigma(Z_t,t)$ is adapted and square-integrable; the test or network function lies in $C^{1,2}$ |
| Existence and uniqueness of a strong SDE solution | A common sufficient condition is that $b,\Sigma$ are locally Lipschitz and satisfy suitable growth or non-explosion conditions |
| Pointwise score and reverse drift | The interior-time density $p_t$ is sufficiently smooth and positive |
| Exact manifold constraint | $M$ is at least a $C^2$ embedded manifold; diffusion is tangent; drift satisfies a second-order constraint |
| Conditional-mean and generator identities | Markov structure and integrability of the required moments |
| A full one-step kernel corresponding to an Itô diffusion | Chapman–Kolmogorov consistency, stochastic continuity, and the correct small-time scaling of local moments |
| Nonsmooth network functions | Smooth parameterizations use the ordinary Hessian formula; piecewise-smooth parameterizations use generalized Itô–Tanaka or weak-derivative theory |

Standard Itô theory is built on semimartingale structure and a suitable quadratic variation. Fractional Brownian motion is also rough; when $H\ne\tfrac12$, the corresponding tools include rough paths, Young integration, and Malliavin calculus.

Under the classical setting above, replace the deterministic ODE

{{< math >}}
dZ_s=v(Z_s,s)ds
{{< /math >}}

with the Itô SDE

{{< math >}}
\boxed{
dZ_s
=
b(Z_s,s)ds
+
\Sigma(Z_s,s)dW_s
}.
{{< /math >}}

The drift $b$ controls the direction of the conditional mean. The stochastic diffusion term $\Sigma dW_s$ controls how trajectories spread after leaving the same state. Define

{{< math >}}
a(z,s)=\Sigma(z,s)\Sigma(z,s)^\top.
{{< /math >}}

An Itô integral can have zero conditional mean and positive conditional variance. Its mean can remain quiet while individual samples spread in different directions. Quadratic variation, transition laws, and conditional uncertainty enter the model through precisely this mechanism.

A one-dimensional example makes the second-order correction visible. If $dZ=dW$ and $f(Z)=Z^2$, Itô’s formula gives

{{< math >}}
df(Z)=2Z\,dW+dt.
{{< /math >}}

Although $dW$ has zero mean, applying the curved function $f$ creates a positive $dt$ term. Random fluctuations interacting with curvature produce a systematic change in the mean; this is the source of the second-order Itô correction.

### 8.2 Diffusion Strength and Drift Must Be Designed Together

The probability-flow ODE of pMF produces a family of marginal densities $p_t$:

{{< math >}}
\partial_t p_t
=
-\nabla\cdot(vp_t).
{{< /math >}}

The marginal density of an SDE satisfies the Fokker–Planck equation:

{{< math >}}
\partial_t p_t
=
-\nabla\cdot(bp_t)
+
\frac12\nabla\cdot\nabla\cdot(ap_t).
{{< /math >}}

Setting $b=v$ while adding a nonzero $\Sigma$ introduces an additional diffusion term and changes the marginal path. Preserving the original $p_t$ requires the drift to absorb this contribution.

A natural compatible drift that makes the SDE share the same family of marginals as the original probability-flow ODE is

{{< math >}}
\boxed{
b_i
=
v_i
+
\frac{1}{2p_t}
\sum_j\partial_{z_j}(a_{ij}p_t)
}.
{{< /math >}}

Division by $p_t$ and the score form below require a sufficiently smooth positive density, so this is best viewed as an interior-time formula for $t\in(0,1)$. If the data law is singular, discrete, or supported exactly on a lower-dimensional manifold at $t=0$, the endpoint can be handled through a limit, dequantization, or additional regularization.

When $\mu_t$ is singular with respect to the ambient Lebesgue measure, begin with the weak form:

{{< math >}}
\boxed{
\frac{d}{dt}\int \phi(z)\,\mu_t(dz)
=
\int \mathcal L_t\phi(z)\,\mu_t(dz)
},
\qquad
\phi\in C_c^\infty,
{{< /math >}}

where

{{< math >}}
\mathcal L_t\phi
=
b\cdot\nabla\phi
+
\frac12a:\nabla^2\phi.
{{< /math >}}

The weak form acts directly on measures and test functions. A sufficiently smooth density reduces it to the pointwise Fokker–Planck and score formulas.

Expanding the compatible drift gives

{{< math >}}
\boxed{
b
=
v
+
\frac12\nabla\cdot a
+
\frac12a\nabla\log p_t
}.
{{< /math >}}

When $a=a(t)$ is independent of the state, $\nabla\cdot a=0$, and the formula simplifies to

{{< math >}}
\boxed{
b
=
v
+
\frac12a(t)s_t
}.
{{< /math >}}

This is the key design principle: **diffusion strength and drift come as a pair.** Each choice of diffusion strength induces a compatible forward or reverse drift, allowing the model to preserve a target marginal path while shaping conditional randomness.

A vector field whose $p_t$-weighted divergence vanishes can also be added to $b$ without changing the Fokker–Planck equation. The expression above is the most direct and interpretable choice.

For a linear Gaussian interpolation, the $r=t$ denoising prediction indirectly yields the score for $0<t<1$. This creates a natural interface between pMF and a compatible SDE. That interface can support stochastic generalized denoising fields, branch-selection variables, and distribution-level training objectives.

### 8.3 Exact Manifolds and “Soft Manifolds” Form Two Different Branches

Suppose a smooth manifold is locally defined by constraints

{{< math >}}
F^\alpha(z)=0,
{{< /math >}}

and an Itô SDE starting from $Z_0\in M$ must remain on $M$. Each constraint requires at least

{{< math >}}
\boxed{
\nabla F^\alpha(z)^\top\Sigma(z,t)=0
}
{{< /math >}}

and

{{< math >}}
\boxed{
\nabla F^\alpha(z)^\top b(z,t)
+
\frac12a(z,t):\nabla^2F^\alpha(z)=0
}.
{{< /math >}}

The first condition keeps diffusion directions tangent to the manifold. The second makes the drift cancel the normal curvature displacement created by quadratic variation. The unit circle offers a simple example. Let $J$ denote a $90^\circ$ rotation. The Stratonovich equation

{{< math >}}
dZ_t=JZ_t\circ dW_t
{{< /math >}}

is equivalent to the Itô equation

{{< math >}}
dZ_t=-\frac12Z_t\,dt+JZ_t\,dW_t.
{{< /math >}}

Its sample paths are almost surely continuous and nowhere differentiable, while Itô’s formula gives $d\|Z_t\|^2=0$. A trajectory that starts on $S^1$ therefore remains on $S^1$. Rough motion can live stably on a smooth geometric space.

For a process constrained to a lower-dimensional manifold, densities are usually defined relative to the manifold volume measure; divergence, score, and the Laplacian then take their intrinsic forms. A full-dimensional $p_t(z)$ and $\nabla\log p_t(z)$ are appropriate when the process has a smooth ambient-space density.

pMF is closer to a different branch: states or network outputs need only remain concentrated near an image set most of the time. Let

{{< math >}}
\rho(z)=\operatorname{dist}(z,M)^2.
{{< /math >}}

Within a smooth tubular neighborhood of $M$, a testable heuristic condition is

{{< math >}}
\boxed{
\mathcal L_t\rho(z)
\le
-\lambda\rho(z)+\epsilon_t
}.
{{< /math >}}

Dynkin’s formula and Grönwall’s inequality yield

{{< math >}}
\mathbb E[\rho(Z_t)]
\le
e^{-\lambda(t-r)}\mathbb E[\rho(Z_r)]
+
\int_r^t e^{-\lambda(t-s)}\epsilon_s\,ds.
{{< /math >}}

This condition accommodates controlled normal diffusion and is independent of path differentiability. For a nonsmooth $M$, $\rho$ can be replaced by a smooth “image-likeness potential,” such as the reconstruction residual of a frozen autoencoder or a differentiable feature distance. The informal claim that outputs “stay near the image set” then becomes a statement that can be analyzed with a generator, tested experimentally, and developed into a training regularizer.

---

## 9. Why Does an “Average Increment Rate” Become Difficult on a Stochastic Path?

Define the pathwise average increment rate

{{< math >}}
\widehat U_{r,t}
=
\frac{Z_t-Z_r}{t-r}.
{{< /math >}}

Integrating the SDE gives

{{< math >}}
Z_t-Z_r
=
\int_r^t b(Z_s,s)ds
+
\int_r^t\Sigma(Z_s,s)dW_s.
{{< /math >}}

Therefore,

{{< math >}}
\boxed{
\widehat U_{r,t}
=
\frac1{t-r}\int_r^t b(Z_s,s)ds
+
\frac1{t-r}\int_r^t\Sigma(Z_s,s)dW_s
}.
{{< /math >}}

The first term is the average drift and the second is an average martingale increment. If $\Sigma$ is adapted and square-integrable, then conditional on $\mathcal F_r$,

{{< math >}}
\mathbb E\left[
\int_r^t\Sigma_s dW_s
\middle|\mathcal F_r
\right]=0,
{{< /math >}}

while its covariance is

{{< math >}}
\operatorname{Cov}\left[
\frac1{t-r}\int_r^t\Sigma_s dW_s
\middle|\mathcal F_r
\right]
=
\frac1{(t-r)^2}
\mathbb E\left[
\int_r^t a_sds
\middle|\mathcal F_r
\right].
{{< /math >}}

Let $\Delta=t-r$. For nondegenerate diffusion, Brownian displacement has probability order $O_p(\sqrt\Delta)$. Dividing by $\Delta$ makes the stochastic component of the average increment rate $O_p(\Delta^{-1/2})$, with variance $O(\Delta^{-1})$.

The $O(\Delta^{-1})$ statement refers specifically to the conditional variance of the averaged martingale term. The full $\widehat U_{r,t}$ also contains randomness in the drift and possible covariance terms.

The quantity is defined for $t>r$ and becomes singular as $t\downarrow r$. The finite deterministic MeanFlow boundary $u(z,t,t)=v(z,t)$ turns into a diverging stochastic increment rate. This reveals a counterintuitive fact:

> On a stochastic trajectory, a shorter interval produces a more volatile “path velocity.” Brownian sample paths are almost surely continuous and nowhere differentiable, so finite-time stochastic increments and quadratic variation replace ordinary curve velocity.

---

## 10. A Pathwise Analogy: Martingales Push the Learning Object toward Stochastic Branches

Fix $r$ and write

{{< math >}}
\widehat U_t
=
\frac{Z_t-Z_r}{t-r}.
{{< /math >}}

Because

{{< math >}}
(t-r)\widehat U_t=Z_t-Z_r,
{{< /math >}}

taking stochastic differentials on both sides gives

{{< math >}}
\widehat U_tdt
+
(t-r)d\widehat U_t
=
b_tdt+\Sigma_tdW_t.
{{< /math >}}

Thus,

{{< math >}}
\boxed{
d\widehat U_t
=
\frac{b_t-\widehat U_t}{t-r}dt
+
\frac{\Sigma_t}{t-r}dW_t
}.
{{< /math >}}

Equivalently,

{{< math >}}
\boxed{
b_tdt+\Sigma_tdW_t
=
\widehat U_tdt
+
(t-r)d\widehat U_t
}.
{{< /math >}}

This is an exact identity along a stochastic path. It separates the two ingredients required by a stochastic MeanFlow: the finite-variation component describes the average trend, and the martingale component carries branch information from the same starting point.

### 10.1 Applying Itô’s Formula to a Network Output

Let the network predict a deterministic state function $u_\theta(Z_t,r,t)$. Itô’s formula gives

{{< math >}}
\begin{aligned}
du_\theta
={}&
\left[
\partial_tu_\theta
+J_zu_\theta b
+\frac12a:\nabla_z^2u_\theta
\right]dt\\
&+J_zu_\theta\Sigma\,dW_t.
\end{aligned}
{{< /math >}}

Here,

{{< math >}}
a:\nabla^2u
=
\sum_{i,j}a_{ij}\,\partial_{ij}u
{{< /math >}}

is computed separately for every output component of the vector-valued function.

Applying the product rule to $(t-r)u_\theta$ gives

{{< math >}}
\begin{aligned}
d[(t-r)u_\theta]
={}&
\left[
u_\theta
+(t-r)
\left(
\partial_tu_\theta
+J_zu_\theta b
+\frac12a:\nabla_z^2u_\theta
\right)
\right]dt\\
&+(t-r)J_zu_\theta\Sigma\,dW_t.
\end{aligned}
{{< /math >}}

If a deterministic state function is required to match $dZ_t$ pathwise, matching the finite-variation and martingale coefficients yields

{{< math >}}
\boxed{
b
=
u
+
(t-r)
\left(
\partial_tu
+J_zu\,b
+\frac12a:\nabla_z^2u
\right)
}
{{< /math >}}

and

{{< math >}}
\boxed{
\Sigma
=
(t-r)J_zu\,\Sigma
}.
{{< /math >}}

The first equation adds an Itô Hessian correction to deterministic MeanFlow. The second asks the network Jacobian to carry the martingale coefficient as well. When $\Sigma$ is nondegenerate and $J_zu$ stays bounded as $t\downarrow r$, the equation drives the Jacobian toward a $1/(t-r)$ scale. This singular scaling suggests a useful division of labor: a deterministic state function can represent the average trend, while independent noise, latent variables, or a transition kernel can represent stochastic branches.

The increment $Z_t-Z_r$ depends on both $Z_r$ and the Brownian history inside the interval. Giving the network an additional random source $\xi$, a summary of the history, or a kernel-valued output makes that branch information explicit.

The design principle is therefore clear: **the Hessian correction adjusts the finite-variation component, while a random input or transition kernel carries the martingale component.** The learning object in stochastic MeanFlow expands from one deterministic field to “average trend + branching mechanism,” creating exactly the representational space needed to move beyond conditional averaging.

### 10.2 Transition Operators: What Remains of MeanFlow in the Stochastic Setting?

For a Markov Itô diffusion, define the two-time transition operator

{{< math >}}
(P_{r,t}f)(z)
=
\mathbb E[f(Z_t)\mid Z_r=z]
{{< /math >}}

and the instantaneous generator

{{< math >}}
\mathcal L_tf
=
b(\cdot,t)\cdot\nabla f
+
\frac12a(\cdot,t):\nabla^2f.
{{< /math >}}

The natural finite-time object along a stochastic path is the averaged generator

{{< math >}}
\boxed{
\overline{\mathcal L}_{r,t}f
=
\frac{P_{r,t}f-f}{t-r}
}.
{{< /math >}}

Dynkin’s formula gives

{{< math >}}
\boxed{
\overline{\mathcal L}_{r,t}f
=
\frac1{t-r}
\int_r^tP_{r,s}\mathcal L_sf\,ds
}.
{{< /math >}}

Fixing the left endpoint $r$ and differentiating with respect to the right endpoint $t$ yields

{{< math >}}
\boxed{
P_{r,t}\mathcal L_tf
=
\overline{\mathcal L}_{r,t}f
+
(t-r)\partial_t\overline{\mathcal L}_{r,t}f
}.
{{< /math >}}

Under suitable regularity,

{{< math >}}
\overline{\mathcal L}_{r,t}f
\longrightarrow
\mathcal L_rf,
\qquad t\downarrow r.
{{< /math >}}

The structure resembles the deterministic MeanFlow identity, but the learning object has changed: instead of asking where one point travels along a unique trajectory, it asks how the conditional expectation of an observable changes under a stochastic transition. This view gives a natural hierarchy:

- With coordinate functions $f_i(z)=z_i$, $\overline{\mathcal L}_{r,t}f_i$ is the conditional average increment rate, and its $t\downarrow r$ limit recovers the local drift.
- With quadratic functions $f_{ij}(z)=z_iz_j$, the generator contains $z_ib_j+z_jb_i+a_{ij}$; together with first moments and a small-time limit, this reveals local diffusion.
- Learning $P_{r,t}f$ for a sufficiently rich family of test functions amounts, in principle, to learning the full transition kernel.

{{< math >}}
\boxed{
\text{coordinate functions}
\rightarrow
\text{conditional mean/local drift};
\quad
\text{first- and second-order functions}
\rightarrow
\text{local drift/diffusion};
\quad
\text{rich test-function families}
\rightarrow
\text{full transition kernel}
}.
{{< /math >}}

A deterministic flow map obeys a two-time composition law; the stochastic counterpart is Chapman–Kolmogorov consistency. The operator identity is therefore a stochastic extension of the original MeanFlow idea and a concrete direction suggested by the Markov-generator viewpoint.

---

## 11. Two Learnable Stochastic Objectives


A standard SDE approach learns local objects $(b,a)$, or equivalently a generator, a score, or a reverse drift, and samples through multistep numerical integration. Following pMF’s goal of one-step amortization leads to two particularly clear levels: the conditional mean and the full transition kernel.

### 11.1 Objective A: Predict Only the Conditional Mean Endpoint

For a forward Markov SDE, define

{{< math >}}
u^+(z,r,t)
=
\mathbb E\left[
\frac{Z_t-Z_r}{t-r}
\middle|
Z_r=z
\right].
{{< /math >}}

Because the Itô integral has zero conditional mean,

{{< math >}}
u^+(z,r,t)
=
\frac1{t-r}
\mathbb E\left[
\int_r^t b(Z_s,s)ds
\middle|
Z_r=z
\right].
{{< /math >}}

Let

{{< math >}}
M(z,r,t)
=
\mathbb E[Z_t\mid Z_r=z]
=
z+(t-r)u^+(z,r,t).
{{< /math >}}

The function $M$ satisfies the backward Kolmogorov equation

{{< math >}}
\partial_rM+\mathcal L_rM=0,
\qquad
M(z,t,t)=z,
{{< /math >}}

with generator

{{< math >}}
\mathcal L_rf
=
b(z,r)\cdot\nabla_zf
+
\frac12a(z,r):\nabla_z^2f.
{{< /math >}}

Substituting $M=z+(t-r)u^+$ gives

{{< math >}}
\boxed{
b(z,r)
=
u^+(z,r,t)
-(t-r)
\left[
\partial_ru^+(z,r,t)
+\mathcal L_ru^+(z,r,t)
\right]
}.
{{< /math >}}

Expanded,

{{< math >}}
\boxed{
b
=
u^+
-(t-r)
\left[
\partial_ru^+
+J_zu^+b
+\frac12a:\nabla_z^2u^+
\right]
}.
{{< /math >}}

Through continuous extension, the conditional-mean formulation has a well-behaved boundary:

{{< math >}}
u^+(z,t,t)=b(z,t).
{{< /math >}}

Under the Markov and regularity assumptions above, this is an exact identity for the mean of a forward transition. I will call it the **conditional-mean Itô–MeanFlow identity**.

The derivative here is taken with respect to the left endpoint $r$. The sign difference reflects the direction of conditioning: this definition conditions on the starting state $Z_r=z$, while original pMF starts from the current right-end state $Z_t=z$ and follows an ODE leftward toward an endpoint. For generation, one can introduce a reverse clock and treat the noise end as the starting point of a new process.

The conditional-mean objective is simple, but it compresses the uncertainty generated by future Brownian increments from the same starting state into a single center. The law of total covariance isolates this lost conditional variance:

{{< math >}}
\operatorname{Cov}(Z_t)
=
\operatorname{Cov}(\mathbb E[Z_t\mid Z_r])
+
\mathbb E[\operatorname{Cov}(Z_t\mid Z_r)].
{{< /math >}}

A mean-only output discards the second term. Brownian motion starting at zero has endpoint law $\mathcal N(0,t)$, yet its conditional mean endpoint is always zero. Mean prediction turns an entire Gaussian cloud into a point—the cleanest mathematical miniature of generative averaging.

The initial noise in pMF still provides global diversity. Conditional-mean compression acts on the **residual randomness under the same condition or starting point**. If blur, compromise textures, or mode collapse arise because this information is averaged away, a stochastic transition kernel targets the exact location of the loss.

### 11.2 Objective B: Sample the Full Transition Law in One Step

To preserve stochasticity, the learning target can be expanded from a mean to the full conditional law

{{< math >}}
p(Z_t\mid Z_r=z).
{{< /math >}}

A simple approximation is

{{< math >}}
Z_t
=
M_\theta(z,r,t)
+
L_\theta(z,r,t)\xi,
\qquad
\xi\sim\mathcal N(0,I),
{{< /math >}}

which exactly represents a conditional Gaussian kernel. For non-Gaussian or multimodal transitions, use

{{< math >}}
Z_t=G_\theta(z,r,t,\xi),
{{< /math >}}

so that an auxiliary random variable $\xi$ directly parameterizes a conditional transition kernel.

The core idea of “stochastic one-step generation” is to amortize the sampler of an entire stochastic transition into one network call. A finite-time kernel carries much more global information than a single Euler–Maruyama update.

The map $G_\theta(z,r,t,\xi)$ first defines a stochastic output kernel. An identifiable distribution-level objective can make it stable and multimodally complete. Chapman–Kolmogorov or semigroup consistency connects kernels across different time spans into a Markov process. Stochastic continuity and the following local moment conditions further specialize that process toward an Itô diffusion:

{{< math >}}
\frac{\mathbb E[\Delta Z\mid Z_t=z]}{\Delta t}\to b(z,t),
\qquad
\frac{\operatorname{Cov}(\Delta Z\mid Z_t=z)}{\Delta t}\to a(z,t),
{{< /math >}}

with higher-order jump moments vanishing at the appropriate rates. Retaining higher-order jump moments leads naturally to a general Markov kernel or jump process. Using a multistep diffusion model inside $G_\theta$ yields a stronger but more expensive hierarchical sampler. A single network call, a few stochastic correction steps, and internal multistep sampling form a continuous spectrum between speed and expressivity.

---

## 12. From Noise to Data: The Reverse-Time SDE

Suppose the forward process is

{{< math >}}
dZ_t
=
b(Z_t,t)dt
+
\Sigma(Z_t,t)dW_t,
{{< /math >}}

with $a=\Sigma\Sigma^\top$ and marginal density $p_t$. When time runs backward from $T$ to $0$, the reverse-time drift is

{{< math >}}
b_{\mathrm{back}}
=
b
-\nabla\cdot a
-a\nabla\log p_t,
{{< /math >}}

where $dt<0$.

Alternatively, introduce a forward-running reverse clock $s=T-t$ and define $Y_s=Z_{T-s}$. Its drift is

{{< math >}}
\boxed{
\bar b(y,s)
=
-b(y,T-s)
+(\nabla\cdot a)(y,T-s)
+a(y,T-s)\nabla_y\log p_{T-s}(y)
}.
{{< /math >}}

When the diffusion matrix depends only on time, $\nabla\cdot a=0$.

Deriving a reverse SDE from a given forward SDE brings in the score or equivalent reverse-drift information. A model that directly learns the reverse transition kernel can absorb this information implicitly into the kernel. Keeping the time direction and the sign of $dt$ explicit makes the forward and reverse drift conventions consistent.

---


## 13. How Could the Itô Extension Break Generative Averaging?

This is the most exciting part. Under a multimodal conditional distribution, squared loss drives a deterministic predictor toward the conditional mean:

{{< math >}}
g^\star(z)=\mathbb E[X\mid Z=z].
{{< /math >}}

Suppose the same $z$ admits two equally plausible images $x_1,x_2$:

{{< math >}}
p(X\mid z)
=
\frac12\delta_{x_1}
+
\frac12\delta_{x_2}.
{{< /math >}}

Deterministic MSE regression gives

{{< math >}}
g^\star(z)=\frac{x_1+x_2}{2},
{{< /math >}}

so two sharp branches merge into a compromise in pixel or feature space. Mixed textures in image restoration, averaged detail in super-resolution, and blurred futures in video prediction all echo this simple model.

An Itô extension supports a different representation:

{{< math >}}
\boxed{
(z,\xi)
\longmapsto
X^{(\xi)},
\qquad
X^{(\xi)}\sim p(X\mid z)
}.
{{< /math >}}

The condition $z$ stays fixed while the fresh random variable $\xi$ selects a branch. A conditional mean retains only the center of mass; a transition kernel retains the whole distribution. The diffusion matrix $a(z,t)$ can further describe which directions are most ambiguous and most worth expanding. In this view, the Itô term gives the model a set of **conditional branch coordinates**.

Original pMF already receives global diversity from its initial noise, yet a one-layer network can still encounter path convergence and conditional ambiguity at intermediate states. A stochastic kernel allows the same intermediate state to split again. This creates an appealing possibility even for unconditional one-step generation: high-frequency textures, local structures, and rare modes flattened by a one-step MSE objective could be unfolded again. The direction is especially relevant when model capacity is ample but samples still look averaged.

In principle, a sufficiently expressive deterministic transport can map continuous noise to a complex data distribution. Branch MeanFlow offers a different **factorization**: the drift learns a shared generative backbone, while $\xi$ and the diffusion learn conditional ambiguity and local branching. Separating these roles can reduce target conflict inside a one-step network and give high-probability structure, rare modes, and fine detail more suitable representations. This optimization-level reorganization may matter more than another increase in model width.

### 13.1 The Derivation Suggests Three Levels of Escape from Averaging

The operator view above decomposes the averaging problem into three levels:

{{< math >}}
\boxed{
\text{coordinate functions}
\rightarrow
\text{conditional mean};
\quad
\text{first- and second-order test functions}
\rightarrow
\text{mean and diffusion directions};
\quad
\text{rich test-function families}
\rightarrow
\text{full conditional distribution}
}.
{{< /math >}}

Matching coordinate functions alone keeps the model at the mean level. Adding quadratic functions reveals the conditional covariance $a$ and tells the model how branches should open. Matching a sufficiently rich family of nonlinear observables can recover the full transition kernel. For example, in a perceptual feature space $\phi(x)$, consider

{{< math >}}
f_\omega(x)
=
\exp\!\left(i\omega^\top\phi(x)\right).
{{< /math >}}

Matching $P_{r,t}f_\omega$ over many frequencies $\omega$ amounts to matching the characteristic function of the conditional feature distribution. MMD, energy distance, adversarial feature matching, conditional score matching, and conditional flow matching can all live at this level. MeanFlow provides the two-time amortization structure; a distribution-level loss unfolds an “average answer” into a family of answers.

### 13.2 Imagining a Branch MeanFlow

Extend the deterministic generalized denoising field to

{{< math >}}
D_\theta(z,r,t,\xi),
\qquad
\xi\sim\mathcal N(0,I),
{{< /math >}}

and define

{{< math >}}
u_\theta(z,r,t,\xi)
=
\frac{z-D_\theta(z,r,t,\xi)}{t}.
{{< /math >}}

At the one-step generation boundary $r=0$, each value of $\xi$ selects an image branch:

{{< math >}}
D_\theta(z,0,t,\xi)
\sim
K^{\leftarrow}_{t,0}(z,\cdot)
:=
\operatorname{Law}(Z_0\mid Z_t=z).
{{< /math >}}

Call this provisional idea **Branch MeanFlow**. One possible training recipe contains four parts:

1. **Mean-dynamics consistency.** The prediction averaged over $\xi$ continues to satisfy a MeanFlow or generator constraint, preserving the global transport direction.
2. **Local-covariance consistency.** The covariance across branches induced by different values of $\xi$ matches $a(z,t)$, directing randomness toward the largest conditional ambiguities.
3. **Distribution-level matching.** The full conditional law is matched in pixel, perceptual, or discriminator feature space so that each sample lands on a sharp mode.
4. **Two-time composition consistency.** One long jump and two short jumps agree in distribution, allowing stochastic branches to persist coherently across time.

The image-like parameterization of pMF may be especially valuable here. Each stochastic branch directly outputs an image-like object, so LPIPS, ConvNeXt feature distances, and soft-manifold potentials can pull every branch toward a clear-image region. Drift advances the sample along the main generative direction; diffusion expands along multi-solution directions. Together, they can turn one blurry average into several individually sharp candidates.

A more ambitious version learns $a(z,t)$ itself and aligns it with local principal components of the conditional residual. Ambiguous textures receive stronger diffusion, structurally determined regions receive weaker diffusion, and randomness contracts into several clean branches near the data endpoint. This state-dependent diffusion acts like an **ambiguity map** and could become the key component that breaks averaging.

### 13.3 Where Would the Breakthrough Be Easiest to See?

| Setting | What stochastic branches could change |
|---|---|
| Image restoration, super-resolution, and colorization | Generate several sets of sharp details from the same observation, placing mutually exclusive textures on separate branches |
| Posterior sampling and general inverse problems | Produce a candidate set directly from $p(x\mid y)$ and reveal posterior coverage |
| Medical and scientific imaging | Provide several structurally plausible explanations together with their empirical frequencies |
| Speech, video, and motion prediction | Unfold one history into several coherent futures, each with a sharp motion trajectory |
| Stochastic physical systems | Use drift for average dynamics and diffusion for real perturbations and branching |
| Unconditional one-step image generation | Continue expanding local modes around a fixed initial-noise or intermediate state, restoring textures and rare samples flattened by one-step regression |

---

## 14. What Does This Direction Add to Original pMF?

| Dimension | Original pMF | Candidate Itô / Branch MeanFlow |
|---|---|---|
| Dynamics | Deterministic probability-flow ODE | Drift and diffusion jointly define a stochastic transition |
| Endpoint from one start | One endpoint | A conditional family of endpoints indexed by $\xi$ |
| Main learning object | Average velocity / generalized denoising endpoint | Conditional mean, local covariance, and full kernel |
| Mathematical operator | First-order material derivative | Second-order generator and two-time transition operator |
| Branch information | Encoded globally by the initial noise | Encoded jointly by initial noise and randomness along the path |
| One-step inference | Native 1-NFE | A direct kernel supports stochastic 1-NFE; a few corrector steps provide stronger refinement |
| Main strength | Fast deterministic endpoint transport | Multimodality under a fixed condition, ambiguity directions, and conditional coverage |
| Current stage | pMF has ImageNet results | This post develops a derivational skeleton ready for training-objective and validation studies |

Four potential gains stand out:

1. **Move beyond the conditional mean.** One observation can produce several individually sharp results, and the generator represents a family of answers directly.
2. **Give randomness a direction.** The matrix $a(z,t)$ describes local ambiguity geometry, opening branches along uncertain textures, poses, and future motions.
3. **Upgrade one-step generation into one-step stochastic transition.** The network amortizes the whole kernel, performing long-range transport and branch selection in one forward pass.
4. **Create a controllable quality–coverage spectrum.** Diffusion strength, branch dimension, and a small number of corrector steps offer continuous control over sharpness, coverage, and latency.

### 14.1 What Will Determine Success?

- **Distribution-level supervision:** a conditional-mean loss controls the main direction, while characteristic functions, MMD, adversarial objectives, or conditional flow matching promote mode coverage.
- **Structured diffusion:** $a(z,t)$ should align with conditional ambiguity directions and be trained jointly with a compatible drift.
- **Stable branch variables:** $\xi$ should preserve semantic identity over long time spans so that one branch develops coherently.
- **Kernel composition:** distributional agreement between one long jump and several short jumps supplies two-time supervision.
- **Image-likeness constraints:** perceptual losses and soft-manifold potentials keep every branch near a sharp-image region.
- **Efficient second-order estimation:** Hessian-vector products, Hutchinson estimators, or operator distillation can bring second-order costs into a trainable range.

A probability-flow ODE, a mean-only kernel, a full stochastic kernel, and a few stochastic corrector steps form a natural experimental spectrum. Comparing them can reveal whether generative averaging originates primarily from the conditional-mean objective, one-step capacity, trajectory convergence, or weak coverage of rare modes in the training objective.

---

## 15. Summary

The core of original pMF is a chain connecting image output, average velocity, and dynamical supervision:

{{< math >}}
\boxed{
\text{easy-to-model image output}
\xrightarrow{D_\theta\to u_\theta\to V_\theta}
\text{dynamically consistent velocity supervision}
}.
{{< /math >}}

Statistically, the network learns

{{< math >}}
\boxed{
v^\star(z,t)
=
\mathbb E[\varepsilon-X\mid Z_t=z]
}.
{{< /math >}}

Dynamically, MeanFlow compresses average motion over an entire ODE interval into a two-time field. At $r=0,t=1$, one forward pass can therefore predict the endpoint.

After adding Itô stochastic increments, the drift/score correction that preserves the same marginal path is

{{< math >}}
\boxed{
b
=
v
+
\frac{1}{2p_t}\nabla\cdot(ap_t)
}.
{{< /math >}}

The Itô generator additionally introduces the second-order drift correction

{{< math >}}
\boxed{
\frac12a:\nabla^2u
}
{{< /math >}}

and the product differential $d[(t-r)u]$ contains the martingale term

{{< math >}}
\boxed{
(t-r)J_zu\,\Sigma\,dW
}.
{{< /math >}}

The learning object now separates into three levels. A standard route learns local $(b,a)$ or a score/reverse drift. A mean-only route outputs a conditional mean. A fully stochastic route learns a transition kernel. The first captures local stochastic dynamics, the second still averages, and the third preserves the full branch structure under the same condition.

The central vision is:

> **pMF compresses one deterministic generative journey into one forward pass. Itô / Branch MeanFlow aims to compress an entire family of possible journeys from the same starting point into one stochastic forward pass. If this direction succeeds, a one-step model can move from predicting an average answer to sampling a sharp family of answers.**

### 15.1 Beyond Classical Euclidean Itô Diffusions

The derivation in this post covers the standard setting: a continuous Markov semimartingale in a finite-dimensional Euclidean space. Beyond this setting, the idea of amortizing finite-time transitions can remain useful while the differential tools change with the object:

- **Diffusions on smooth manifolds:** use tangent diffusion, curvature corrections, intrinsic scores, and the Laplace–Beltrami generator.
- **Nonsmooth or singular support:** use weak generators, Dirichlet forms, or transition kernels; the ambient Hessian formula applies to smooth fields.
- **Jump processes:** the generator acquires a Lévy jump-integral term; averaged generators and transition operators still apply, while higher-order small-time moments retain jump information.
- **Non-semimartingale drivers such as fractional Brownian motion and rough paths:** use rough-path theory, Young integration, or Skorokhod/Malliavin tools.
- **Non-Markovian dynamics with memory:** augment the state with history, latent variables, or paths. The learning object then becomes a path-conditional kernel in place of $p(Z_t\mid Z_r)$, with a corresponding generalization of Chapman–Kolmogorov consistency.

Each direction extends outward from Itô-pMF and can grow into a research program of its own. They share one picture:

> **ODE MeanFlow amortizes a point map induced by a deterministic trajectory. Itô / Branch MeanFlow amortizes a Markov transition operator or conditional transition kernel. Rough sample paths provide stochastic branches, while the state space, SDE coefficients, test functions, and two-time consistency shape those branches. The most exciting outcome would be for every sample to land on a sharp mode and for repeated samples to cover the full space of answers.**

---

## Acknowledgments

Thanks to the GPT editing team.

---

## References and Implementations

- Yiyang Lu et al., [*One-step Latent-free Image Generation with Pixel Mean Flows*](https://arxiv.org/abs/2601.22158), arXiv:2601.22158v3, 2026.
- Zhengyang Geng et al., [*Mean Flows for One-step Generative Modeling*](https://arxiv.org/abs/2505.13447), arXiv:2505.13447, 2025.
- Zhengyang Geng et al., [*Improved Mean Flows: On the Challenges of Fastforward Generative Models*](https://arxiv.org/abs/2512.02012), arXiv:2512.02012, 2025/2026.
- Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, [*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://arxiv.org/abs/2303.08797), JMLR 2025 / arXiv:2303.08797.
- Yang Song et al., [*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456), ICLR 2021.
- Bernt Øksendal, *Stochastic Differential Equations: An Introduction with Applications*, Springer, 6th ed., 2003.
- Elton P. Hsu, *Stochastic Analysis on Manifolds*, American Mathematical Society, 2002.
- Stewart N. Ethier and Thomas G. Kurtz, *Markov Processes: Characterization and Convergence*, Wiley, 1986.
- Peter K. Friz and Martin Hairer, *A Course on Rough Paths*, Springer, 2nd ed., 2020.
- Official pMF implementation: [Lyy-iiis/pMF](https://github.com/Lyy-iiis/pMF).


