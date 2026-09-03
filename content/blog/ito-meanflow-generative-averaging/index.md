+++
title = 'Can Itô MeanFlow Break Generative Averaging?'
date = 2026-09-03
author = 'Zixiang Ni'
description = 'A concise derivation from pixel MeanFlow to stochastic transition kernels, and why Itô branches may help one-step generators escape conditional averaging.'
tags = ['Generative Modeling', 'MeanFlow', 'Stochastic Differential Equations']
draft = false

[params]
subtitle = 'From one deterministic endpoint to a distribution of sharp branches'
math = true
+++

One-step generators compress a long trajectory into one network evaluation. That compression is powerful, yet a deterministic prediction can merge several plausible continuations into one conditional average. This note develops a possible way forward: retain the finite-time viewpoint of pixel MeanFlow (pMF), replace deterministic trajectories with Itô diffusions, and let the network amortize an entire **transition kernel** instead of a single endpoint.

The resulting idea is simple to state:

> **pMF compresses one generative journey into one forward pass. Itô MeanFlow could compress a family of journeys from the same state into one stochastic forward pass.**

## The Answer Up Front

The main conclusions are:

1. **pMF learns finite-time transport.** It converts an image-like network output into an average velocity that can carry noise to data in one step.
2. **A deterministic conditional target can average.** When one state admits several plausible futures, squared regression selects their conditional mean.
3. **An Itô increment creates fresh branches.** The diffusion term has zero conditional mean and positive conditional variance, so trajectories can separate after passing through the same state.
4. **The stochastic analogue of MeanFlow is an operator identity.** The Itô generator describes local drift and diffusion; the transition operator describes their finite-time effect.
5. **The full transition kernel is the object that preserves multimodality.** A random-input network can sample a sharp branch while remaining consistent with mean dynamics, local covariance, and two-time composition.

This suggests a candidate model, which I will call **Branch MeanFlow**:

{{< math >}}
\boxed{
D_\theta(z,r,t,\xi)
\sim
\operatorname{Law}(Z_r\mid Z_t=z),
\qquad
\xi\sim\mathcal N(0,I).
}
{{< /math >}}

The current state $z$ fixes the shared context. The auxiliary variable $\xi$ selects one possible branch.

---

## pMF in Four Equations

Let a clean image be $X\sim p_{\mathrm{data}}$, let $\varepsilon\sim\mathcal N(0,I)$, and define the linear interpolation

{{< math >}}
Z_t=(1-t)X+t\varepsilon,
\qquad t\in[0,1].
{{< /math >}}

For one training pair, the velocity is

{{< math >}}
Y=\varepsilon-X.
{{< /math >}}

A network that observes only $(Z_t,t)$ and is trained with squared loss identifies the posterior velocity

{{< math >}}
\boxed{
v^\star(z,t)
=
\mathbb E[\varepsilon-X\mid Z_t=z].
}
{{< /math >}}

Flow Matching learns this instantaneous field. MeanFlow instead learns the average velocity over a whole interval. Along the probability-flow trajectory through $Z_t=z$, define

{{< math >}}
u^\star(z,r,t)
=
\frac{1}{t-r}
\int_r^t
v^\star(Z_s,s)\,ds.
{{< /math >}}

The endpoint follows immediately:

{{< math >}}
\boxed{
Z_r
=
z-(t-r)u^\star(z,r,t).
}
{{< /math >}}

pMF packages the average velocity into an image-like field

{{< math >}}
D^\star(z,r,t)
=
z-t\,u^\star(z,r,t).
{{< /math >}}

At the one-step boundary $r=0$, $t=1$, and $Z_1=\varepsilon$,

{{< math >}}
D_\theta(\varepsilon,0,1)
\approx
Z_0.
{{< /math >}}

The practical flow is therefore:

{{< math >}}
\boxed{
\text{noise}
\longrightarrow
\text{image-like prediction}
\longrightarrow
\text{average transport}
\longrightarrow
\text{one-step endpoint}.
}
{{< /math >}}

The full pMF method contains JVP-based consistency, auxiliary objectives, perceptual losses, and carefully sampled two-time pairs. For the present question, its essential contribution is the two-time amortization of a deterministic endpoint.

---

## Where Generative Averaging Enters

Suppose the same condition $z$ supports two equally plausible sharp outputs:

{{< math >}}
p(X\mid z)
=
\frac12\delta_{x_1}
+
\frac12\delta_{x_2}.
{{< /math >}}

The optimal deterministic squared-loss prediction is

{{< math >}}
\boxed{
g^\star(z)
=
\mathbb E[X\mid z]
=
\frac{x_1+x_2}{2}.
}
{{< /math >}}

Two distinct modes become one compromise. In images this can appear as blended textures or softened detail; in video it can appear as an averaged future; in inverse problems it can hide several valid reconstructions behind one representative answer.

pMF already receives global diversity from the initial noise. The remaining opportunity lies in **conditional diversity after a state has been fixed**. If several continuations share the same intermediate representation, a deterministic endpoint gives that state one continuation. A stochastic transition can let the state branch again.

---

## Itô Increments Open Branches

Replace the deterministic ODE with an Itô SDE:

{{< math >}}
\boxed{
dZ_s
=
b(Z_s,s)\,ds
+
\Sigma(Z_s,s)\,dW_s,
}
{{< /math >}}

and write

{{< math >}}
a(z,s)=\Sigma(z,s)\Sigma(z,s)^\top.
{{< /math >}}

The relevant nonsmooth object is the **sample path**. Brownian paths are continuous and almost surely nowhere differentiable, so ordinary path velocity gives way to stochastic increments and quadratic variation. The image set may still be treated as an empirical soft manifold in the ambient pixel space. Classical Itô formulas use a network or test function in $C^{1,2}$; smooth activations such as SiLU give a natural implementation route.

Integrating over $[r,t]$ yields

{{< math >}}
Z_t-Z_r
=
\int_r^t b(Z_s,s)\,ds
+
\int_r^t\Sigma(Z_s,s)\,dW_s.
{{< /math >}}

For the martingale increment

{{< math >}}
M_{r,t}
=
\int_r^t\Sigma(Z_s,s)\,dW_s,
{{< /math >}}

we have

{{< math >}}
\mathbb E[M_{r,t}\mid\mathcal F_r]=0,
{{< /math >}}

while

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

The mean increment follows the drift, while individual samples fan out according to $a$. This is the key representational gain: **the same starting state can acquire new randomness during the interval**.

### Preserving the pMF Marginal Path

The probability-flow ODE evolves a density $p_t$ according to

{{< math >}}
\partial_t p_t
=
-\nabla\cdot(vp_t).
{{< /math >}}

The SDE evolves it according to the Fokker–Planck equation

{{< math >}}
\partial_t p_t
=
-\nabla\cdot(bp_t)
+
\frac12\nabla\cdot\nabla\cdot(ap_t).
{{< /math >}}

At interior times with a smooth positive density, the SDE can share the same marginals as the original probability-flow ODE by pairing the diffusion with the compatible drift

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

Equivalently,

{{< math >}}
b
=
v
+
\frac12\nabla\cdot a
+
\frac12a\nabla\log p_t.
{{< /math >}}

This formula creates an appealing design space: keep the desired marginal journey from noise to data, while choosing $a(z,t)$ to control how conditional branches open along that journey. The diffusion matrix can act as a learned **ambiguity map**.

---

## The Itô Version of MeanFlow

For a state function $f$, Itô’s formula gives

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

Define the Itô generator

{{< math >}}
\boxed{
\mathcal L_s f
=
b(\cdot,s)\cdot\nabla f
+
\frac12a(\cdot,s):\nabla^2f.
}
{{< /math >}}

The first-order term carries drift. The Hessian term records the systematic effect of quadratic variation. The martingale term carries sample-specific branch information.

For a Markov diffusion, define the finite-time transition operator

{{< math >}}
(P_{r,t}f)(z)
=
\mathbb E[f(Z_t)\mid Z_r=z].
{{< /math >}}

Its average generator over the interval is

{{< math >}}
\boxed{
\overline{\mathcal L}_{r,t}f
=
\frac{P_{r,t}f-f}{t-r}.
}
{{< /math >}}

Dynkin’s formula gives

{{< math >}}
\overline{\mathcal L}_{r,t}f
=
\frac1{t-r}
\int_r^t
P_{r,s}\mathcal L_s f\,ds.
{{< /math >}}

Differentiate with respect to the right endpoint:

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

This has the same finite-time structure as MeanFlow:

{{< math >}}
\boxed{
\text{instantaneous generator}
=
\text{average generator}
+
\text{interval-length correction}.
}
{{< /math >}}

MeanFlow asks where a point travels along one deterministic curve. Itô MeanFlow asks how an observable changes under a distribution of stochastic trajectories.

### The Conditional-Mean Special Case

Let

{{< math >}}
M(z,r,t)
=
\mathbb E[Z_t\mid Z_r=z]
=
z+(t-r)u^+(z,r,t).
{{< /math >}}

The backward Kolmogorov equation,

{{< math >}}
\partial_r M+\mathcal L_rM=0,
{{< /math >}}

implies

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

This is a conditional-mean Itô–MeanFlow identity. Its generator contains the second-order correction $\tfrac12a:\nabla^2u^+$, yet its output remains one conditional mean. The full stochastic information lives in the transition kernel.

---

## From a Mean Endpoint to a Branch Kernel

The law of total covariance identifies the information carried by the branches:

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

A mean endpoint carries the first term. A stochastic transition also carries the second.

The hierarchy suggested by the Itô derivation is:

| Learned object | Information retained | Generative behavior |
|---|---|---|
| Coordinate means | Conditional center | One representative endpoint |
| Linear and quadratic observables | Mean and branch directions | Locally calibrated stochastic spread |
| A rich family of observables or the full kernel | Conditional distribution | Multiple coherent modes |

Coordinate functions recover the drift. Quadratic functions expose $a$. A sufficiently rich set of test functions characterizes the entire conditional transition law.

This motivates a random-input finite-time sampler:

{{< math >}}
\boxed{
Z_t
=
G_\theta(z,r,t,\xi),
\qquad
\xi\sim\mathcal N(0,I),
}
{{< /math >}}

with

{{< math >}}
G_\theta(z,r,t,\xi)
\sim
p(Z_t\mid Z_r=z).
{{< /math >}}

Under the reverse generative clock, the pMF-style boundary becomes

{{< math >}}
\boxed{
D_\theta(z,0,t,\xi)
\sim
K^\leftarrow_{t,0}(z,\cdot)
:=
\operatorname{Law}(Z_0\mid Z_t=z).
}
{{< /math >}}

One call still crosses the whole interval. Different values of $\xi$ select different endpoints from the same condition.

### A Candidate Training Recipe

Branch MeanFlow could combine four constraints:

1. **Mean-dynamics consistency.** Averaging over $\xi$ follows the MeanFlow or Itô–MeanFlow identity, preserving the shared transport backbone.
2. **Local-covariance consistency.** Short-interval branch covariance matches the diffusion geometry $a(z,t)$.
3. **Distribution-level matching.** MMD, adversarial feature matching, conditional flow matching, or conditional score matching encourages distinct samples to cover the target conditional law.
4. **Two-time composition consistency.** A long stochastic jump agrees in distribution with two composed shorter jumps, mirroring the Chapman–Kolmogorov equation.

The image-like parameterization of pMF remains useful here. Each branch can land directly in an image-shaped output space, where perceptual features encourage sharp structure. Drift learns what the branches share; diffusion learns where they should separate; $\xi$ chooses which branch appears.

---

## Why This Could Break Averaging

The most promising effect is a new factorization of the generative task:

{{< math >}}
\boxed{
\text{shared structure}
+
\text{ambiguity geometry}
+
\text{branch selection}
}
{{< /math >}}

The drift can carry large-scale semantic motion. The diffusion matrix can open directions associated with uncertain texture, pose, color, or future motion. The latent branch variable can turn those directions into individually sharp samples.

This could matter first in tasks where ambiguity is visible under a fixed observation:

- super-resolution, colorization, and image restoration;
- posterior sampling for inverse problems;
- video, speech, and motion prediction;
- scientific systems with genuinely stochastic futures.

The same mechanism may also help unconditional one-step generation. An intermediate state can reopen into several local continuations, restoring high-frequency detail and rare modes that a single deterministic regression target tends to compress. The model would perform both long-range transport and branch selection in one evaluation.

The clearest early experiment would pair an ambiguous conditional benchmark with four measurements: individual-sample sharpness, conditional coverage, branch calibration, and one-step latency. A synthetic two-mode problem can verify the mechanism; super-resolution or colorization can reveal whether the mechanism scales to perceptual ambiguity.

If the idea works, the central advance would go beyond adding noise to a fast generator. It would turn one-step generation from **endpoint prediction** into **transition-law sampling**.

---

## Beyond Classical Itô Diffusions

The transition-kernel viewpoint also points toward broader settings:

- On a smooth manifold, tangent diffusion and the Laplace–Beltrami generator can shape intrinsic branches.
- Near a nonsmooth image set, weak generators and learned image-likeness potentials can replace exact geometric constraints.
- Jump processes can represent discrete branch changes through a Lévy generator.
- Non-Markovian problems can use history-conditioned kernels, allowing an entire past to determine the distribution of futures.

Across these cases, the durable idea is finite-time stochastic amortization: learn the law of where trajectories can go, together with enough consistency to make those trajectories compose.

> **The hopeful picture is that pMF supplies the one-step transport skeleton, while Itô dynamics supplies branch coordinates. Their combination could let a fast generator produce one sharp answer per sample and the full family of answers across samples.**

---

## Citation

If this article is useful in your research, please cite it as:

> Zixiang Ni. “Can Itô MeanFlow Break Generative Averaging?” *Camellia86*, September 3, 2026. https://camellia86.github.io/blog/ito-meanflow-generative-averaging/

    @misc{ni2026itomeanflow,
      author = {Ni, Zixiang},
      title  = {Can Itô MeanFlow Break Generative Averaging?},
      year   = {2026},
      month  = sep,
      url    = {https://camellia86.github.io/blog/ito-meanflow-generative-averaging/}
    }

---

## References

- Yiyang Lu et al., [*One-step Latent-free Image Generation with Pixel Mean Flows*](https://arxiv.org/abs/2601.22158), arXiv:2601.22158, 2026.
- Zhengyang Geng et al., [*Mean Flows for One-step Generative Modeling*](https://arxiv.org/abs/2505.13447), arXiv:2505.13447, 2025.
- Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden, [*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://arxiv.org/abs/2303.08797), JMLR, 2025.
- Yang Song et al., [*Score-Based Generative Modeling through Stochastic Differential Equations*](https://arxiv.org/abs/2011.13456), ICLR, 2021.
- Bernt Øksendal, *Stochastic Differential Equations: An Introduction with Applications*, Springer, 6th ed., 2003.
