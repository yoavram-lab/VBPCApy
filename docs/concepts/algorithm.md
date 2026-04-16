# Algorithm Overview

VBPCApy implements the Variational Bayesian PCA (VB-PCA) algorithm described by
Ilin and Raiko (2010), with extensions for missing data, sparse masks, and
Automatic Relevance Determination (ARD).

## Generative model

VB-PCA assumes the following generative process for an observed data matrix
$X \in \mathbb{R}^{D \times N}$ ($D$ features, $N$ samples):

$$
X = A S + \mu \mathbf{1}^T + \varepsilon
$$

where:

- $A \in \mathbb{R}^{D \times K}$ is the **loading matrix** ($K$ latent components),
- $S \in \mathbb{R}^{K \times N}$ is the **score matrix** (latent representations),
- $\mu \in \mathbb{R}^{D}$ is the **bias** (per-feature mean), and
- $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$ is isotropic noise with variance $V = \sigma^2$.

## Variational inference

Rather than computing the exact posterior $p(A, S, \mu \mid X)$, VB-PCA
approximates it with a factorised Gaussian:

$$
q(A, S, \mu) = \prod_i q(A_i) \prod_j q(S_j) \, q(\mu)
$$

The algorithm maximises the **Evidence Lower Bound (ELBO)**, equivalently
minimising the **variational free energy** (negative ELBO), by alternating
between:

1. **E-step (scores):** update each $q(S_j)$ given current loadings and noise.
2. **E-step (loadings):** update each $q(A_i)$ given current scores and noise.
3. **M-step (noise):** update the noise variance $V$.
4. **Bias update:** update $q(\mu)$ if `bias=True`.

Each update has a closed-form Gaussian solution. The posterior covariances
$\text{Av}_i$ (per-row loading covariance) and $\text{Sv}_j$
(per-column score covariance) are maintained throughout.

## Automatic Relevance Determination (ARD)

ARD places a hierarchical prior on the loading columns:

$$
p(A_{\cdot k}) = \mathcal{N}(0, V_{a,k}^{-1} I)
$$

where $V_a = (V_{a,1}, \dots, V_{a,K})$ are per-component precisions. Components
with large $V_{a,k}$ are effectively pruned — their loadings shrink toward zero,
providing automatic model complexity control.

### ARD-related parameters

| Parameter | Description |
|-----------|-------------|
| `hp_va`, `hp_vb` | Shape and rate of the Gamma hyperprior on $V_a$ |
| `hp_v` | Hyperprior on the noise precision |
| `niter_broadprior` | Number of warmup iterations under a broad (uninformative) prior before ARD engages |
| `va_init` | Initial value for the broad prior variance |

During the first `niter_broadprior` iterations, $V_a$ is held at a large value
(`va_init`) so the model can find reasonable loadings before ARD shrinkage begins.

## Missing data handling

When entries of $X$ are unobserved, VB-PCA restricts the likelihood terms to
observed entries only. Each update equation sums only over the observed subset,
and the posterior covariances adapt to the per-observation pattern of missingness.

For data with shared missingness patterns (many columns missing the same set of
rows), VBPCApy identifies unique patterns and reuses the covariance factorisation
across columns sharing a pattern, reducing computation.

## PCA rotation

After convergence, the latent space can be rotated to a PCA-like orientation where:

1. Score dimensions are uncorrelated (diagonal covariance).
2. Components are sorted by decreasing explained variance.

This is a post-hoc orthogonal rotation that does not change the model fit — it
only reorients $A$ and $S$ for interpretability. Controlled by the
`rotate2pca` option (enabled by default).

## Cost function

The cost reported by `model.cost_` is the **negative ELBO** (variational free
energy). It includes:

- The expected log-likelihood over observed entries.
- KL divergences for $q(A)$, $q(S)$, and $q(\mu)$ against their priors.
- Entropy terms for the posterior covariances.

A decreasing cost indicates improving model fit. The cost is used as a
model-selection metric in [`select_n_components`](../api/model-selection.md).

## References

> Ilin, A., & Raiko, T. (2010). Practical Approaches to Principal Component
> Analysis in the Presence of Missing Values. *Journal of Machine Learning
> Research*, 11, 1957–2000.
