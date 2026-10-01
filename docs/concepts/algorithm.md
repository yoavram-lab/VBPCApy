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
p(A_{\cdot k}) = \mathcal{N}(0, V_{a,k} I)
$$

where $V_a = (V_{a,1}, \dots, V_{a,K})$ are per-component prior variances.
After the warm-up described below, each is re-estimated every iteration as

$$
V_{a,k} = \frac{\lVert a_k \rVert^2 + \operatorname{tr}\Sigma_{a_k} + 2\,\text{hp\_va}}
{(p + 2\,\text{hp\_vb}) / f},
$$

with $p$ features and $f$ the observed fraction of entries. Components whose
$V_{a,k}$ shrinks toward zero are effectively pruned: their loadings are pulled
to zero, providing automatic model complexity control.

### ARD-related parameters

| Parameter | Role |
|-----------|------|
| `hp_va` | Added to the numerator of the $V_a$ update. Small values let unused components shrink toward zero (strong pruning); larger values put a floor of about $2\,\text{hp\_va} f / p$ under every $V_{a,k}$. Also enters the bias prior variance. |
| `hp_vb` | Added to the denominator; larger values shrink every component's prior variance. |
| `hp_v` | Hyperprior term in the noise variance update, $V = (\text{residual} + 2\,\text{hp\_v}) / (n_{\text{obs}} + 2\,\text{hp\_v})$. |
| `niter_broadprior` | Number of warm-up iterations with $V_a$ held at `va_init` before ARD updates start; convergence checks also wait for it. |
| `va_init` | Initial (broad) prior variance for the loadings and bias. |

During the first `niter_broadprior` iterations, $V_a$ is held at a large value
(`va_init`) so the model can find reasonable loadings before ARD shrinkage begins.
`recommend_config()` sets `niter_broadprior=0` and larger `hp_va`/`hp_vb` than
the core defaults, which weakens pruning.

### Inspecting pruning

After fitting, `VBPCA` exposes:

- `prior_variances_`: the final $V_a$, and `bias_prior_variance_`;
- `component_relevance_`: each returned component's share of reconstruction
  energy, $\lVert a_k \rVert \lVert s_k \rVert$ normalized to sum to one;
- `effective_rank(threshold=0.01)`: the number of components above a relevance
  threshold;
- `prior_trace_`: $V_a$ at every iteration when fitted with
  `record_prior_trace=True`.

$V_a$ is last updated before the PCA rotation in each iteration, so with
`rotate2pca` its entries need not line up with the returned components; use
`component_relevance_` for per-component statements.

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
