# VB-PCA Convergence Paper — Finalized Ablation, Analysis & Proof Plan

## Paper thesis

> VB-PCA has a closed-form ELBO that guarantees monotonic improvement and
> provably converges at a characterizable linear rate—yet the reference
> implementation ships with the ELBO disabled, a broken RMS criterion, and
> a warmup period that wastes 90%+ of iterations. We prove convergence
> rates under different data regimes, diagnose the defaults, and construct
> a composite Bayesian convergence metric—a learned weighted combination
> of ELBO improvement, subspace angle, and smoothed RMS—tuned to maximize
> posterior coverage and rank-selection accuracy. The resulting Pareto
> front of wall time vs. accuracy reveals that enabling the ELBO + tuned
> priors + lowered warmup yields 3–10× faster convergence with no
> accuracy loss, potentially making full-covariance VB tractable at scales
> previously reserved for diagonal approximations.

---

## Part I — Theoretical Results (Proofs)

### Proof 1: ELBO monotonic improvement

**Statement.** Each coordinate-ascent update (scores → loadings → bias →
noise → hyperpriors) yields $\mathcal{L}(\theta^{(t+1)}) \geq
\mathcal{L}(\theta^{(t)})$, with equality iff $\theta^{(t)}$ is a
fixed point of the CAVI operator.

**Sketch.**
Each conditional update maximises $\mathcal{L}$ over the corresponding
factor while holding the others fixed. For conjugate-exponential
models every conditional optimum is available in closed form
(Bishop §10.2.1). The ELBO decomposes as:

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z})] - \mathbb{E}_q[\log q(\mathbf{Z})]$$

where $\mathbf{Z} = \{\mathbf{A}, \mathbf{S}, \boldsymbol{\mu}, \sigma^2, \mathbf{v}_a\}$.
Each coordinate step replaces one factor $q_j$ with $q_j^* = \arg\max_{q_j} \mathcal{L}$,
which is the exponential-family form $\log q_j^* = \mathbb{E}_{q_{-j}}[\log p(\mathbf{X}, \mathbf{Z})] + \text{const}$.
Since the feasible set (all distributions over $\mathbf{Z}_j$) is
convex and $\mathcal{L}$ is concave in each $q_j$, the replacement
can only increase (or maintain) $\mathcal{L}$.

**What's needed.** Write out the six update blocks (S, A, μ, V, v_a, v_μ)
from `_full_update.py` and verify each matches the CAVI optimum.
Cross-reference against `cf_full.m` / `_cost.py` to confirm the
implemented cost is the actual ELBO.

**Novelty.** This is textbook for generic CAVI, but nobody has written it
down explicitly for the Ilin & Raiko (2010) model with all six blocks,
optional bias, missing-data masks, and ARD hyperpriors.

---

### Proof 2: Convergence guarantee

**Statement.** The ELBO sequence $\{\mathcal{L}^{(t)}\}$ converges to a
finite limit. The parameter sequence $\{\theta^{(t)}\}$ has at least one
accumulation point, and every accumulation point is a stationary point
of $\mathcal{L}$.

**Sketch.**
- Monotonicity (Proof 1) + bounded above by $\log p(\mathbf{X})$
  → monotone convergence theorem → $\mathcal{L}^{(t)} \to \mathcal{L}^*$.
- The parameter space is compact after adding the ARD constraints
  ($v_a > 0$, $\sigma^2 > 0$, bounded loadings from the ARD prior).
  Bolzano–Weierstrass → accumulation point exists.
- Continuity of $\mathcal{L}$ + each coordinate update being a
  strict maximiser → any accumulation point satisfies the
  block-coordinate optimality conditions (Tseng 2001, Proposition 2).

**What's needed.** Verify compactness of the effective parameter space
under ARD (the prior precision $1/v_a$ can push $v_a \to 0^+$, so we
need the $\epsilon$-floor in the code: `_EPS_VAR`). Cite Tseng (2001)
for the block-coordinate convergence theorem.

---

### Proof 3: Asymptotic linear convergence rate

**Statement.** Near a fixed point $\theta^*$, the CAVI operator
$T: \theta^{(t)} \mapsto \theta^{(t+1)}$ is a contraction with
spectral radius $\rho = \rho(J_T(\theta^*)) < 1$, where $J_T$ is
the Jacobian of the CAVI map. The convergence rate satisfies:

$$\|\theta^{(t)} - \theta^*\| \leq C \cdot \rho^t$$

and $\rho$ depends on the data regime through:

$$\rho \approx 1 - \frac{\lambda_{\min}(\mathbf{A}^\top \mathbf{A})}{\lambda_{\max}(\mathbf{A}^\top \mathbf{A}) + n\sigma^2 / \text{tr}(\text{diag}(\mathbf{v}_a)^{-1})}$$

**Sketch.**
For Gaussian CAVI, each conditional is a linear function of the
sufficient statistics of the other factors. The composite CAVI map
$T = T_6 \circ T_5 \circ \cdots \circ T_1$ is therefore a linear
operator near the fixed point. Its spectral radius determines the
asymptotic contraction rate.

For the scores update:
$\mathbb{E}[\mathbf{s}_j | \text{rest}] = (\mathbf{A}^\top \text{diag}(\mathbf{m}_j) \mathbf{A}/\sigma^2 + \mathbf{I})^{-1} \mathbf{A}^\top \text{diag}(\mathbf{m}_j) \mathbf{x}_j / \sigma^2$

where $\mathbf{m}_j$ is the observation mask for column $j$.
Under complete data ($\mathbf{m}_j = \mathbf{1}$), this simplifies and
the rate depends on the condition number $\kappa(\mathbf{A}^\top\mathbf{A})$
and the SNR $\|\mathbf{A}\|^2 / \sigma^2$.

**What's needed.**
1. Derive $J_T$ by differentiating each of the six update equations
   with respect to the natural parameters.
2. Show $\rho(J_T) < 1$ under mild conditions (SNR > 0, $n > k$).
3. Characterise how $\rho$ varies with:
   - SNR (signal strength relative to noise)
   - Missing fraction (weakens coupling → changes eigenstructure of $J_T$)
   - Overspecification ($k > k_{\text{true}}$: ARD must kill components → slow)
   - Condition number of true loadings
4. Numerical verification: fit $\log\|\theta^{(t)} - \theta^*\|$ vs. $t$,
   extract empirical $\rho$, compare to analytical prediction.

**Novelty.** This is the core theoretical contribution. Convergence
*rate* of CAVI for PCA has not been characterised in the literature.
The closest is Ghahramani & Beal (2000) and Ilin & Raiko (2010), who
discuss convergence qualitatively but give no rate bounds.

---

### Proof 4: RMS oscillation is a measurement artefact

**Statement.** The 2-cycle oscillation in raw RMS under complete data is
caused by evaluating the reconstruction error at an asymmetric point in
the coordinate cycle (after loadings update but before bias update), not
by divergence of the algorithm. The ELBO is monotonically non-decreasing
throughout.

**Sketch.**
- At iteration $t$, RMS is computed using $\mu^{(t-1)}$ (not yet updated)
  but $\mathbf{A}^{(t)}, \mathbf{S}^{(t)}$ (already updated).
- The reconstruction $\hat{\mathbf{X}} = \mathbf{A}^{(t)} \mathbf{S}^{(t)} + \boldsymbol{\mu}^{(t-1)} \mathbf{1}^\top$ uses a stale mean.
- After the bias update, $\mu^{(t)}$ adjusts to compensate, but RMS has already been recorded.
- This creates an alternating over/under-correction visible as period-2 oscillation.
- Pre-centering eliminates it because $\mu \approx 0$ and the stale-mean effect vanishes.
- The ELBO, computed from all current parameters including covariances, does not oscillate.

**What's needed.** Show formally that the ELBO uses all current
parameters (confirmed in `_append_cost_value` → `compute_full_cost`),
while `_recompute_rms` uses the reconstruction before the bias update
(confirmed in `_iteration_step` ordering). This is diagnostic, not a
deep theorem—but it resolves a confusing empirical observation that
could undermine confidence in the algorithm.

---

### Proof 5: k-fold CV consistency via probe-set holdout

**Statement.** The probe-set mechanism (prms) is a single-fold holdout
estimator of reconstruction error. Under mild regularity conditions
(finite variance, i.i.d. observation pattern), the k-fold average of
prms converges to the expected reconstruction error as $k \to \infty$,
providing a consistent model selection criterion.

**Sketch.**
- `_ensure_metric_opts` samples a random 10% holdout from observed entries
- Entries are masked out (NaN/removed) so the fit never sees them
- `_recompute_rms` evaluates $\text{RMSE}_{\text{probe}} = \sqrt{\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} (x_{ij} - \hat{x}_{ij})^2}$
- k-fold: partition observed entries into $K$ folds, fit $K$ times, average probe RMSE
- By the law of large numbers, the average converges to $\mathbb{E}[\text{RMSE}]$
- For model selection: choose $k^* = \arg\min_k \overline{\text{RMSE}}_K(k)$

**What's needed.** This is mostly a framing contribution—showing that
the existing prms infrastructure is a single fold of CV, and that
wrapping it in a $K$-fold loop gives a consistent estimator. The key
subtlety is that the observation mask changes between folds (different
entries are held out), so the effective data regime varies per fold.
Under MCAR this is benign; under MNAR the folds may not be exchangeable.

---

## Part II — Ablations & Empirical Analyses

### Analysis 1: RMS computation timing (Tier 1)

**Question.** Does evaluating RMS after the *full* coordinate cycle
(including bias+noise) eliminate oscillation?

| Factor | Levels |
|--------|--------|
| RMS timing | T0 (current), T1 (end of cycle), T2 (pre+post average), T3 (after loadings only) |
| Pre-centering | R0 (none+bias), R1 (center, no bias), R2 (center+bias) |
| Data geometry | D2 (50×100×5) |
| Missingness | M0 (complete), M2 (MCAR 30%) |

**Runs.** 4 × 3 × 1 × 2 = **24**

**Implementation.** Monkey-patch `_iteration_step` to move the
`_recompute_rms` call. Record both raw RMS and ELBO per iteration.
Measure oscillation amplitude in last 20 iterations.

**Expected outcome.** T1 eliminates oscillation (all parameters
current); T0 preserves it (stale μ). This empirically confirms Proof 4.

**Blocked by.** Nothing.

---

### Analysis 2: ELBO vs. angle vs. RMS as stopping criterion (Tier 2)

**Question.** How many iterations does each criterion need? Which fires
first? Do they agree on when to stop?

| Factor | Levels |
|--------|--------|
| Criterion | C0 (raw RMS), C1 (smoothed RMS), C2 (angle), C3 (ELBO abs), C4 (ELBO rel), C5 (ELBO curvature) |
| Data geometry | D1–D4 |
| Missingness | M0–M3 |
| Priors | P0 (legacy default) |

**Runs.** 6 × 4 × 4 × 1 = **96** (run all criteria simultaneously per
fit, report which would fire first)

**Implementation.** Enable `cfstop=[10, 1e-6, 1e-3]` for all runs.
Post-hoc evaluate all 6 criteria on the recorded traces. Report
iteration-to-fire for each criterion.

**Key metrics.**
- Iterations to convergence (per criterion)
- Final accuracy (angle to ground truth, RMSE)
- Agreement matrix: how often do criteria C_i and C_j stop within 5 iters of each other?

**Expected findings.**
- ELBO fires earliest on dense data (angle also fast)
- RMS criterion wastes iterations due to oscillation on dense data
- On missing data, angle stalls but ELBO and smoothed RMS continue improving
- ELBO curvature (C5) detects the transition to linear regime

**Blocked by.** Nothing (cfstop exists, just needs to be enabled).

---

### Analysis 3: Prior tuning for coverage and rank accuracy (Tier 3)

**Question.** What prior settings (hp_va, hp_vb, hp_v) *optimize*
posterior coverage and rank-selection accuracy, and how does this
interact with convergence speed?

| Factor | Levels |
|--------|--------|
| Prior config | P0–P6 (7 levels, see design matrix) |
| Data geometry | D2 (50×100×5), D3 (100×200×10) |
| Missingness | M0, M2 (MCAR 30%), M4 (MCAR 75%) |
| Criterion | C3 (ELBO abs), C2 (angle) |

**Runs.** 7 × 2 × 3 × 2 = **84**

**Key metrics — the three objectives.**
1. **Rank accuracy**: $|\hat{k}_{\text{ARD}} - k_{\text{true}}|$ — how well ARD recovers the true rank
2. **Coverage**: empirical posterior coverage at 90% nominal level on held-out entries
3. **Wall time**: iterations to convergence × per-iter cost (convergence speed)

**Additional diagnostics (scored against known truth, not training data).**
- Convergence rate (empirical $\rho$ from log-linear fit)
- Final RMSE on training data
- RMSE on holdout entries
- CRPS on holdout entries (via `scoringrules.crps_gaussian`)
- Log predictive density on holdout
- PIT KS statistic (calibration beyond coverage)
- Mean interval width (sharpness)
- Effective rank $\|A\|_F^2 / \|A\|_2^2$
- Loadings Frobenius error after Procrustes alignment
- Noise variance recovery $|\hat{\sigma}^2 - \sigma^2_{\text{true}}|$

**Analysis approach — multi-objective optimization.**

This is not just sensitivity analysis. We are solving:
$$\min_{\mathbf{h}} \; \bigl( \text{wall\_time}(\mathbf{h}), \; |\hat{k} - k_{\text{true}}|(\mathbf{h}), \; |0.90 - \text{coverage}(\mathbf{h})| \bigr)$$
where $\mathbf{h} = (\text{hp\_va}, \text{hp\_vb}, \text{hp\_v})$.

1. Fit a Gaussian process surrogate for each objective over the 84-run grid
2. Extract the 3D Pareto front (wall time, rank error, coverage gap)
3. Identify the "knee" — the prior setting that gives the best coverage and
   rank accuracy before wall time explodes
4. Report regime-conditional optima: the best prior differs by missingness level

**Expected findings.**
- Near-uninformative priors (P1) → fastest convergence but slightly worse coverage
  (posterior too wide → coverage paradoxically *above* nominal; or too narrow from
  underestimated variance)
- Moderate priors (P2) → best coverage-accuracy tradeoff
- Strong priors (P3–P4) → best rank accuracy (strong ARD) but slow and over-regularised
- The Pareto knee is regime-dependent: complete data favours weaker priors;
  high-missingness data needs moderate priors to stabilise ARD
- Asymmetric priors (P5, P6) → pathological; reported but not recommended

**Blocked by.** Issue #50 (expose hp_va/hp_vb/hp_v).

---

### Analysis 4: niter_broadprior profiling (Tier 4)

**Question.** What is the minimum safe warmup period before ARD activates?

| Factor | Levels |
|--------|--------|
| niter_broadprior | B0 (0), B1 (5), B2 (10), B3 (50), B4 (100) |
| Data geometry | D1–D4 |
| Missingness | M0–M4 |
| Priors | P0, P1 |

**Runs.** 5 × 4 × 5 × 2 = **200**

**Key metrics.**
- Convergence rate
- Rank recovery (compare ARD-pruned rank to true rank)
- Whether algorithm destabilises at low B values under high missingness
- Wall-clock time to convergence

**Expected findings.**
- B0 (no warmup) is safe for complete data but may cause premature pruning under >50% missing
- B2 (10 iters) is safe across all tested regimes
- B4 (100 iters) wastes 90+ iterations universally

**Implication.** Recommend B2 (10) as new default, with B1 (5) for
complete-data shortcuts and B3 (50) as conservative fallback for
extreme missingness.

**Blocked by.** Issues #50, #52.

---

### Analysis 5: Full convergence characterisation (Tier 5)

**Question.** How does convergence rate vary across the full factorial
design? Is it predictable from data characteristics?

| Factor | Levels |
|--------|--------|
| Data geometry | D1–D8 (8 levels) |
| Missingness | M0–M6 (7 levels) |
| Prior config | P0–P4 (5 levels) |
| Criterion | C3 (ELBO abs), C5 (ELBO curvature), C7 (composite) |
| Algorithm | A0 (VB), A1 (MAP) |
| Pre-centering | R0, R1 |
| SNR | N0–N3 (4 levels) |
| Condition number κ(A) | K0–K2 (3 levels) |
| Overspecification | O0–O1 (2 levels) |

**Runs.** 8 × 7 × 5 × 3 × 2 × 2 × 4 × 3 × 2 = 80,640 configs × 3 seeds = **241,920**
(subsample via Latin hypercube over continuous factors if HPC budget constrains)

**Key metrics (per run).** Full endpoint battery (see design matrix):
- Empirical $\rho$ (spectral radius from log-linear fit of ELBO improvement)
- Iterations to convergence (each criterion)
- All quality metrics: coverage, CRPS, log pred density, PIT KS, rank recovery,
  effective rank, pp-eigentest p-value, loadings Frobenius, noise var error
- Wall-clock time

**Analyses on the grid.**
1. **ANOVA / regression**: $\log(\rho) \sim \text{geometry} + \text{missingness} + \text{priors} + \text{SNR} + \text{\kappa(A)} + \text{overspec} + \text{interactions}$
2. **Regime map**: heatmap of median iterations-to-convergence across (geometry × missingness)
3. **Rate prediction**: can $\rho$ be predicted from observable quantities (effective SNR, condition number, missing fraction)?
4. **VB vs. MAP**: does removing posterior variances speed convergence? By how much?
5. **Criterion agreement**: Sankey diagram showing which criterion fires first across regimes

**Expected contribution.** First systematic characterisation of VB-PCA
convergence rates across data regimes. The regression model for $\rho$
connects theory (Proof 3) to empirical reality.

**Blocked by.** Tiers 1–4 results (to select best defaults), all Phase 1 issues.

---

### Analysis 6: Scalability frontier (Tier 6)

**Question.** At what $(n, p)$ does full-covariance VB become slower
*to converge* than diagonal VB (VBPCAd)?

| Factor | Levels |
|--------|--------|
| Data geometry | D5 (500×200), D6 (1000×100), D7 (200×500), D8 (5000×500) |
| Missingness | M0, M2, M4 |

**Runs.** 4 × 3 = **12** configs, time-profiled.

**Key metrics.**
- wall-clock per iteration (full cov vs. diagonal)
- iterations to convergence (full cov vs. diagonal)
- total wall-clock to convergence (the product)
- final accuracy (full cov vs. diagonal)

**Expected contribution.** If full-covariance VB converges in 10–20
iterations (as Tiers 1–2 suggest for dense data), the per-iteration
overhead of maintaining $k \times k$ covariance matrices may be offset
by fewer iterations, making full VB tractable at scales where diagonal
was assumed necessary.

**Blocked by.** Tier 5; issue #16 (VBPCAd variant) for direct comparison.
Can do partial analysis (full-cov only) without #16.

---

### Analysis 7: k-fold cross-validation (Tier 2.5 — parallel track)

**Question.** Does k-fold CV via the existing prms mechanism improve
model selection over single holdout?

| Factor | Levels |
|--------|--------|
| Folds | 1 (current), 5, 10 |
| Data geometry | D1–D4 |
| Missingness | M0–M4 |
| True rank | 3, 5, 10 |

**Runs.** 3 × 4 × 5 × 3 = **180** (each k-fold config requires $K$ refits)

**Implementation.** Wrap `select_n_components` in a fold-rotation loop:
1. Partition observed entries into $K$ disjoint folds
2. For each fold, set `xprobe` to that fold's entries, mask them from training
3. Collect final prms per fold per candidate $k$
4. Average across folds, select $k^* = \arg\min_k \overline{\text{prms}}_K(k)$

**Key metrics.**
- Rank recovery accuracy (vs. single holdout, vs. ELBO-based selection)
- Variance of selected rank across seeds
- Computational cost ($K$× more fits)

**Expected findings.**
- 5-fold CV reduces variance of rank estimate vs. single holdout
- Diminishing returns from 5→10 folds
- Under MNAR, fold exchangeability breaks down → CV less reliable
- ELBO-based selection (no holdout needed) may match CV accuracy at 1/K the cost

**Blocked by.** Nothing (prms infrastructure exists). Can implement as a
utility function wrapping `select_n_components`.

---

### Analysis 8: Composite Bayesian convergence metric & Pareto front (Tier 7)

**Question.** Can we construct a single convergence criterion — a
learned weighted combination of ELBO, angle, and RMS — that is tuned
to maximize downstream objectives (coverage, rank accuracy), and what
does the Pareto front of wall time vs. accuracy look like?

**Core idea.** Each individual stopping criterion (ELBO plateau, angle
threshold, RMS plateau, ELBO curvature) stops at a different iteration.
Stopping earlier saves compute but may sacrifice accuracy/coverage.
Stopping later improves accuracy but wastes compute. The optimal
stopping point depends on *what you care about*.

**The composite metric.**

Define a per-iteration convergence score:
$$\mathcal{C}^{(t)} = w_1 \cdot \tilde{\Delta}\mathcal{L}^{(t)} + w_2 \cdot \alpha^{(t)} + w_3 \cdot \tilde{\Delta}\text{RMS}^{(t)} + w_4 \cdot \kappa^{(t)}$$

where:
- $\tilde{\Delta}\mathcal{L}^{(t)}$ = normalised ELBO improvement (relative to first improvement)
- $\alpha^{(t)}$ = subspace angle between $\mathbf{A}^{(t)}$ and $\mathbf{A}^{(t-1)}$
- $\tilde{\Delta}\text{RMS}^{(t)}$ = smoothed RMS relative change (2-pt MA)
- $\kappa^{(t)}$ = ELBO curvature (second difference)

Stop when $\mathcal{C}^{(t)} < \epsilon$ for $\tau$ consecutive iterations.

The weights $\mathbf{w} = (w_1, w_2, w_3, w_4)$ and thresholds $(\epsilon, \tau)$
are learned via Bayesian optimisation to minimise a *scalarised* objective:

$$\mathcal{J}(\mathbf{w}, \epsilon, \tau) = \lambda_1 \cdot |\hat{k} - k_{\text{true}}| + \lambda_2 \cdot |0.90 - \text{coverage}| + \lambda_3 \cdot \log(\text{iters}) + \lambda_4 \cdot \text{CRPS} + \lambda_5 \cdot D_{\text{KS}}(\text{PIT})$$

where $\lambda_1, \ldots, \lambda_5$ trace out the Pareto front as they vary.
CRPS and PIT KS-statistic are strictly proper scores that separate calibration
from sharpness more cleanly than coverage alone.

**The Pareto front.**

For each point in the Tier 5 grid, we have full per-iteration traces.
Post-hoc, for any candidate $(\mathbf{w}, \epsilon, \tau)$, we can compute
when that composite criterion would have fired and what the resulting
accuracy/coverage/wall-time would have been. This means:

1. **No additional fits needed** — just replay the recorded traces
2. **Full Pareto front** scanned by varying $\lambda_1 : \cdots : \lambda_5$
3. **Regime-conditional fronts**: the Pareto shape differs by missingness level

**Implementation.** Pareto extraction via `pymoo` from the existing grid (no
surrogate needed — we have all runs). Composite weight learning via `optuna`
multi-objective Bayesian optimization.

**Visualisation.** 3D Pareto surface (exact analogy to poster.tex
cost-vs-skill):
- x-axis: wall time (iterations to composite-stop)
- y-axis: rank error ($|\hat{k} - k_{\text{true}}|$)
- z-axis: coverage gap ($|0.90 - \text{coverage}|$)
- colour: data regime (complete / 30% MCAR / 75% MCAR)

Project onto 2D:
- wall time vs. rank accuracy (main figure)
- wall time vs. coverage
- rank accuracy vs. coverage

**Stacking variant (Bayesian model averaging).**

In addition to the composite *criterion*, we can also stack *models*:
- For each stopping criterion $C_m$, get model $\hat{\theta}_{C_m}$
- Yao et al. (2018) stacking weights from LOO predictive densities
  via `arviz.weight_models(method='stacking')`:
  $\mathbf{w}^* = \arg\max_{\mathbf{w} \in \Delta_M} \sum_i \log \sum_m w_m p(x_i | \hat{\theta}_{C_m})$
- The stacked ensemble inherits strengths of each criterion:
  ELBO gives fast stopping on dense data, angle gives precision on
  low-rank, smoothed RMS is robust under missingness

**Stacking targets (multiple objectives, separately optimised).**
- LOO reconstruction RMSE → optimises accuracy
- Posterior predictive log-likelihood → optimises calibration (`scipy.stats.norm.logpdf`)
- CRPS → optimises sharpness + calibration (`scoringrules.crps_gaussian`)
- PIT uniformity → full distributional calibration (`scipy.stats.kstest`)
- Coverage at 90% nominal → directly optimises coverage

**Connection to Analysis 3 (priors).**

The composite metric weights and the prior hyperparameters are *jointly*
tunable. The full optimisation is:
$$\min_{\mathbf{h}, \mathbf{w}, \epsilon, \tau, v_a^{(0)}, f_{\text{probe}}} \; \mathcal{J}(\mathbf{h}, \mathbf{w}, \epsilon, \tau, v_a^{(0)}, f_{\text{probe}})$$

where $\mathbf{h} = (\text{hp\_va}, \text{hp\_vb}, \text{hp\_v})$, the stopping
threshold $\epsilon$ and patience $\tau$ are from Factors 12–13, the initial
broad prior $v_a^{(0)}$ is Factor 14, and probe fraction $f_{\text{probe}}$ is
Factor 15. This gives a **recommended default configuration**: specific values
for all tunable knobs that sit at the Pareto knee for the most common data regimes.

**Expected contributions.**
1. First composite convergence metric for VB-PCA (or any CAVI model) tuned to downstream objectives
2. Pareto front mapping the wall-time/accuracy/coverage tradeoff — practitioners pick their operating point
3. Regime-conditional recommendations: different optimal configurations for complete vs. sparse data
4. Bayesian stacking across stopping criteria is novel for PCA model averaging
5. Joint prior + stopping + warmup tuning provides a single "recommended defaults" configuration

**Python packages consumed at this tier.**
- `scoringrules` — CRPS, log score (proper scoring primitives)
- `arviz` — Yao (2018) stacking via `az.weight_models()`; LOO-PIT via `az.plot_loo_pit()`
- `pymoo` — Pareto front extraction from existing grid (no surrogate needed)
- `optuna` — composite metric weight learning (lightweight multi-objective BO)
- `netcal` — reliability diagrams for supplementary calibration figures

**Follow-on — pp-eigentest (conditional, separate sweep):**
pp-eigentest builds its own composite grid for rank test fidelity,
conditioned on VBPCApy's tuned defaults from this sweep. The VBPCApy
sweep updates defaults first; pp-eigentest then optimises its own
objectives (rank significance power, type-I error control) taking
the tuned VBPCApy as a fixed upstream dependency.

**What we build (genuine gap — no package exists).**
- Trace recorder (~100 lines): standardised per-iteration recording format for post-hoc replay
- Composite criterion replay (~200 lines): for any $(\mathbf{w}, \epsilon, \tau)$, compute when it fires on a saved trace

**Blocked by.** Tier 5 traces (no additional fits; all computation is post-hoc replay).

---

## Part III — Dependency Graph & Execution Order

```
Phase 0: Proofs 1–4 (pen-and-paper, parallel with everything)
         Analysis 1 (24 runs, no code changes)
         Analysis 2 (96 runs, enable cfstop)

Phase 1: Code changes
         #50 — expose hp_va/hp_vb/hp_v
         #52 — store angle in learning curves
         #34 — get_params/set_params for sklearn
         k-fold CV utility function

Phase 2: Analysis 3 (84 runs, needs #50) — prior Pareto front
         Analysis 4 (200 runs, needs #50 + #52)
         Analysis 7 (180 runs, needs k-fold utility)
         Proof 5 (k-fold consistency, parallel with Analysis 7)

Phase 3: Analysis 5 (16,800 runs on HPC, needs Phase 1–2 results)
         → records full per-iteration traces for ALL criteria simultaneously

Phase 4: Analysis 8 — composite metric + Pareto front (post-hoc on Tier 5 traces)
         → learns (w₁, w₂, w₃, w₄, ε, τ) via Bayesian optimisation
         → jointly tunes priors (from Analysis 3) + stopping weights
         → maps the 3D Pareto surface: wall time × rank error × coverage gap
         → extracts regime-conditional recommended defaults
         Analysis 6 (12 configs, needs Tier 5)
         Proof 3 numerical verification (compare empirical ρ to analytical)
```

**Critical path:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4

**Key insight for Phase 4:** Analysis 8 requires NO additional model
fits. The 16,800 runs from Tier 5 record full per-iteration traces
(ELBO, angle, RMS, noise_var, cost per iteration). For any candidate
composite metric $(\mathbf{w}, \epsilon, \tau)$, we replay the traces
post-hoc and compute when it would have stopped and what accuracy/
coverage would result. This makes the Bayesian optimisation over
$(\mathbf{w}, \epsilon, \tau)$ cheap — each objective evaluation is a
numpy operation on pre-recorded arrays, not a model fit.

**Parallelism:**
- Proofs 1–4 are pen-and-paper; can proceed in parallel with all code work
- Analyses 1 & 2 require zero code changes; can run immediately
- Analysis 7 (k-fold) is independent of the prior/broadprior analyses
- Analysis 8 (composite metric + Pareto) requires only the recorded traces from Tier 5

---

## Part IV — Paper Outline (sections → analyses/proofs)

| Section | Content | Sources |
|---------|---------|---------|
| §1 Introduction | VB-PCA has provable convergence nobody exploited; we construct a tuned composite metric and map the Pareto front | — |
| §2 Background | Ilin & Raiko model, CAVI, ELBO | Ilin & Raiko (2010) |
| §3.1 ELBO monotonicity | **Proof 1** | §3 Theory |
| §3.2 Convergence guarantee | **Proof 2** | §3 Theory |
| §3.3 Linear convergence rate | **Proof 3** (core theorem: $\rho$ depends on SNR, missingness, κ) | §3 Theory |
| §3.4 RMS oscillation diagnosis | **Proof 4** + **Analysis 1** | §3 Theory + §4 Empirics |
| §3.5 k-fold CV consistency | **Proof 5** | §3 Theory |
| §4.1 Stopping criteria comparison | **Analysis 2** — which criterion fires first/best per regime | §4 Empirics |
| §4.2 Prior tuning | **Analysis 3** — multi-objective optimisation of (hp_va, hp_vb, hp_v) for coverage, rank accuracy, wall time | §4 Empirics |
| §4.3 Warmup period | **Analysis 4** — minimum safe niter_broadprior | §4 Empirics |
| §4.4 Full convergence characterisation | **Analysis 5** — main empirical result: convergence rate map, regression model for $\rho$ | §4 Empirics |
| §4.5 Scalability frontier | **Analysis 6** — full-cov vs. diagonal crossover | §4 Empirics |
| §5.1 Composite Bayesian convergence metric | **Analysis 8** — learned weighted combination of ELBO/angle/RMS tuned to downstream objectives | §5 The Pareto Front |
| §5.2 Pareto front: wall time vs. accuracy vs. coverage | **Analysis 8** — 3D Pareto surface, regime-conditional projections | §5 The Pareto Front |
| §5.3 Joint prior + stopping optimisation | **Analysis 3** + **Analysis 8** — recommended defaults at the Pareto knee | §5 The Pareto Front |
| §5.4 Bayesian stacking across criteria | **Analysis 8** stacking variant — model averaging for PCA | §5 The Pareto Front |
| §5.5 k-fold CV for rank selection | **Analysis 7** — k-fold via prms infrastructure | §5 The Pareto Front |
| §6 Recommended defaults | Synthesis: specific values for priors, composite weights, warmup, thresholds per regime | §6 Discussion |
| §7 Software | VBPCApy design, API | JOSS overlap |

---

## Part V — What each proof/analysis gives the paper

| Item | Contribution type | Novelty |
|------|-------------------|---------|
| Proof 1 | Theoretical | Explicit verification for Ilin & Raiko model (6 blocks + ARD + missing masks) |
| Proof 2 | Theoretical | First explicit convergence guarantee for VB-PCA (not just generic CAVI) |
| Proof 3 | Theoretical + Empirical | **Core theorem**: characterisable linear rate $\rho$ depending on data regime |
| Proof 4 | Diagnostic | Resolves a confusing empirical observation; demonstrates ELBO superiority |
| Proof 5 | Methodological | k-fold CV from existing infrastructure; exchangeability under MCAR |
| Analysis 1 | Empirical (diagnostic) | Pinpoints RMS timing as root cause, not algorithmic divergence |
| Analysis 2 | Empirical (prescriptive) | ELBO is the theoretically and empirically best stopping criterion |
| Analysis 3 | **Multi-objective optimisation** | Priors tuned for coverage + rank accuracy + wall time; Pareto knee per regime |
| Analysis 4 | Empirical (prescriptive) | niter_broadprior=10 is safe; 100 wastes 90% of compute |
| Analysis 5 | **Main empirical result** | First convergence rate map across the full VB-PCA design space |
| Analysis 6 | Empirical (practical) | Fast convergence may shift the full-cov/diagonal tradeoff boundary |
| Analysis 7 | Empirical (model selection) | k-fold via prms improves rank selection reliability |
| Analysis 8 | **Core methodological result** | Composite Bayesian convergence metric tuned to downstream objectives; Pareto front of wall time vs. accuracy vs. coverage; joint prior+stopping optimisation; Bayesian stacking across criteria |

### The three headline results

1. **Proof 3 + Analysis 5**: VB-PCA converges at a linear rate $\rho$ that
   is characterisable from observable data properties (SNR, missingness,
   condition number). First such result for any VB-PCA variant.

2. **Analysis 8 (composite metric + Pareto front)**: A learned convergence
   criterion $\mathcal{C}^{(t)} = \mathbf{w}^\top \mathbf{f}^{(t)}$ that
   is tuned to maximise coverage and rank accuracy. The Pareto front shows
   practitioners exactly how much accuracy they trade for speed, and
   provides regime-specific recommended operating points.

3. **Analysis 3 + Analysis 8 (joint optimisation)**: The priors and the
   stopping rule are not independent knobs — they should be tuned jointly.
   The recommended default configuration (specific hp values + composite
   weights + warmup period) sits at the Pareto knee across the most common
   data regimes, yielding 3–10× faster convergence than legacy defaults
   with no loss in coverage or rank accuracy.

---

## Immediate next actions

1. **Run Analysis 1** (24 runs, zero code changes, ~10 min)
2. **Run Analysis 2** (96 runs, just enable cfstop, ~30 min)
3. **Start Proofs 1–2** (pen-and-paper / LaTeX, verify against `_cost.py` and `_full_update.py`)
4. **Implement #50** (expose priors — unblocks prior tuning Analysis 3)
5. **Implement #52** (store angle in lc — unblocks Analysis 4 and full trace recording for Analysis 8)
6. **Write k-fold CV wrapper** (unblocks Analysis 7)
7. **Design Tier 5 trace format** — see specification below

---

## Tier 5 trace format specification

Every run in Tiers 2+ must record a standardised trace for post-hoc replay
in Analysis 8. Saved as `.npz` per run; collected into a single Parquet table
per tier for analysis.

### Per-iteration arrays (length = n_iter)

```
rms[t]          — raw RMS
rms_smooth[t]   — 2-point moving average RMS
angle[t]        — subspace angle (requires #52)
cost[t]         — ELBO/cost (requires cfstop enabled)
noise_var[t]    — noise variance
delta_cost[t]   — ELBO decrease per iteration
delta2_cost[t]  — curvature: second difference of ELBO
```

### Per-run endpoint scalars

**Quality metrics (scored against known truth):**
```
rank_recovered     — ARD-pruned rank
rank_exact         — binary: k_recovered == k_true
effective_rank     — ||A||_F^2 / ||A||_2^2
coverage_90        — posterior interval coverage at 90% on holdout
mean_interval_width — sharpness at 90% on holdout
CRPS_holdout       — scoringrules.crps_gaussian(obs, mu, sigma)
log_pred_density   — mean log predictive density on holdout
PIT_ks_stat        — scipy.stats.kstest(pits, 'uniform') statistic
RMSE_holdout       — reconstruction error on held-out entries
angle_final        — subspace angle to A_true
loadings_frobenius — Frobenius error after Procrustes alignment
noise_var_error    — |σ̂² − σ²_true|
rms_final          — final reconstruction error (training set)
```

**Wall-time metrics:**
```
n_iter             — total iterations run
wall_time          — seconds to convergence
n_warmup_wasted    — iters in broad-prior phase after angle converged
```

**Convergence diagnostics:**
```
convergence_rate   — empirical ρ̂ from log-linear fit
oscillation_amplitude — period-2 RMS amplitude in last 20 iters
elbo_mono_violations  — count of iters where ELBO decreased
```

### Per-run metadata (design factors)

```
geometry_id, missingness_id, prior_id, criterion_id, algorithm_id,
precentering_id, broadprior_id, snr_id, kappa_id, overspec_id,
threshold_id, patience_id, init_prior_id, probe_fraction_id, seed
```

### Python dependencies for endpoint computation

```
scoringrules    — CRPS (crps_gaussian)
scipy.stats     — log predictive density (norm.logpdf), PIT (kstest)
scipy.linalg    — Procrustes alignment (orthogonal_procrustes),
                  subspace angles (subspace_angles)
arviz           — stacking weights (weight_models), LOO-PIT
pymoo           — Pareto front extraction
optuna          — composite weight learning
netcal          — reliability diagrams (supplementary)
```
