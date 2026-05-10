# 02 — Mathematics

## 1. Setup and Notation

| Symbol | Meaning |
|---|---|
| `x ∈ [0,1]^(C×H×W)` | Current adversarial example (image) |
| `x_0` | Original clean image |
| `ε` | L∞ perturbation budget (e.g. 0.05) |
| `f_T` | Target model (black box) |
| `f_S` | Surrogate model (white box) |
| `L(x, y)` | Cross-entropy loss at image x, true label y |
| `g_S = ∂L_S/∂x` | Surrogate gradient — free, unlimited, biased |
| `g_T ≈ ĝ_T` | Target gradient — expensive, unbiased estimate |
| `η ∈ [0,1]` | Blending weight (our key variable) |
| `q` | Number of NES query pairs per step |
| `{g_1,...,g_q}` | Individual NES gradient draw estimates |
| `ĝ = (1/q)Σg_i` | Mean NES gradient estimate |

---

## 2. The Attack Update Rule

At each step t, the adversarial example is updated by:

```
g_blend(η) = η · g_S  +  (1 - η) · ĝ_T

m_t = μ · m_{t-1}  +  g_blend(η) / ||g_blend(η)||_1    (momentum)

x_{t+1} = Π_{x_0, ε} [ x_t + α · sign(m_t) ]
```

where `Π_{x_0,ε}` clips to the L∞ ball of radius ε around x_0 and to [0,1].

This is MI-FGSM with a blended gradient source. The question is entirely: **how do we choose η at each step t?**

---

## 3. NES Gradient Estimation

Natural Evolution Strategies estimates the gradient by sampling random directions:

```
Draw u_k ~ N(0, I^D),  k = 1,...,q   (antithetic: also use -u_k)

ĝ_T ≈ (1/qσ) Σ_{k=1}^{q} u_k · [f_T(x + σu_k) - f_T(x - σu_k)]
```

**Why antithetic pairs?** Using both `u_k` and `-u_k` halves the variance of the estimator at no extra query cost (each pair costs 2 queries but gives 2 correlated draws that cancel opposite-direction noise).

**Rao-Blackwell baseline subtraction:** Replace `f_T(x ± σu_k)` with `f_T(x ± σu_k) - b` where `b = mean of all 2q evaluations`. This is the optimal constant that minimises estimator variance while keeping it unbiased (proof: the baseline is orthogonal to the gradient in expectation).

Each of the 2q per-draw gradient contributions `{g_i}` is:
```
g_i = u_i · (f_pos_i - baseline) / (2σ)   for the positive direction
     -u_i · (f_neg_i - baseline) / (2σ)   for the negative direction
```

These 2q vectors are the "posterior draws" fed to the ELPD blender.

---

## 4. The Power Likelihood Framework (from Gower et al.)

In the original biostatistics paper, the joint model is:

```
p(θ | D_E, D_O) ∝ p(D_E | θ) · p(D_O | θ)^η · p(θ)
```

- `D_E` = small, unbiased RCT data (our: target query draws)
- `D_O` = large, biased observational data (our: surrogate gradients)
- `p(D_O | θ)^η` = the observational likelihood *raised to the power η*

When η=0: observational data is completely ignored → only RCT.  
When η=1: full combination.  
When 0 < η < 1: partial trust in the biased data.

The **optimal η** is found by maximising the Expected Log Predictive Density (ELPD) evaluated on `D_E`:

```
η* = argmax_η  ELPD(η, D_E)
```

ELPD measures: "if I fit the combined model with this η, how well does it predict new unbiased observations?"

---

## 5. ELPD — What It Is and Why We Use It

ELPD (Expected Log Predictive Density) is a Bayesian model comparison criterion. For a model M with parameter η:

```
ELPD(η) = E_{ỹ} [log p(ỹ | y_observed, η)]
```

In plain terms: the expected log probability of a new observation under the model. Higher ELPD = better predictive model.

**WAIC (Watanabe-Akaike Information Criterion)** approximates ELPD without needing new data:

```
ELPD_WAIC = lppd - p_WAIC
lppd = Σ_i log(mean_θ [p(y_i | θ)])    ← log pointwise predictive density
p_WAIC = Σ_i Var_θ [log p(y_i | θ)]    ← effective number of parameters (overfitting penalty)
```

**LOO-CV (Leave-One-Out Cross-Validation)** approximates ELPD by:

```
ELPD_LOO = Σ_i log p(y_i | y_{-i})
```

"For each observation y_i, fit the model on all *other* observations, then score y_i."  
LOO-CV is the gold standard because it directly measures out-of-sample predictive accuracy.

---

## 6. Our Adaptation: LOO Cosine ELPD

### The Translation

We map the statistical framework to gradient space:

| Statistics | Our Setting |
|---|---|
| "New observation ỹ" | A new NES gradient draw g_i |
| "Model parameters θ" | The blending weight η |
| "Predictive distribution p(ỹ | θ)" | How well g_blend(η) predicts g_i |
| "Fitting on y_{-i}" | Using the mean of the other q-1 draws |

### The Formula

For each held-out draw `g_i`, the LOO mean of the remaining `q-1` draws is:

```
ĝ_loo_i = (q·ĝ - g_i) / (q - 1)
```

This identity avoids recomputing q separate means — it's O(1) given the overall mean ĝ.

The blended LOO prediction at weight η:

```
μ_LOO_i(η) = η · g_S  +  (1 - η) · ĝ_loo_i
```

The score for how well this predicts the held-out draw:

```
score_i(η) = cos(μ_LOO_i(η), g_i)
           = (μ_LOO_i(η) · g_i) / (||μ_LOO_i(η)|| · ||g_i||)
```

We use cosine similarity (not MSE or log-likelihood) because:
1. We care about **direction**, not magnitude — the attack step size is controlled by α and sign(), so only the gradient direction matters.
2. Cosine similarity is scale-invariant — immune to the σ calibration problems that plagued earlier MSE-based approaches (see `04_bugs_and_fixes.md`).

The ELPD score across all q draws:

```
ELPD(η) = mean_i[score_i(η)] - T · std_i[score_i(η)]
```

The second term is the WAIC-style penalty: `std_i[score_i]` is high when the surrogate is helpful for some draws but harmful for others — genuine uncertainty. T (temperature) controls how strongly we penalise inconsistency.

### The Analytic Structure

Let `Δg = g_S - ĝ_loo_i` (how far the surrogate is from the LOO mean). Then:

```
μ_LOO_i(η) = ĝ_loo_i + η · Δg_i
```

This is a **line segment in gradient space**, parameterised by η, starting at the LOO mean (η=0) and ending at the surrogate (η=1).

The cosine score `cos(μ_LOO_i(η), g_i)` is a smooth function of η with a unique maximum in (0,1) whenever:
- g_S has a component aligned with g_i that is NOT already captured by ĝ_loo_i
- i.e., when the surrogate provides *new directional information*

When g_S ≈ -g_i (fully opposed), the score decreases monotonically → η*=0.  
When g_S ≈ g_i (perfectly aligned), the score peaks at η*=1.  
When g_S has partial alignment, the peak is at an interior η* ∈ (0,1).

---

## 7. η Selection: Grid Search

We evaluate ELPD(η) at `n_grid = 21` equally spaced points in [0, 1]:

```
η_grid = {0.00, 0.05, 0.10, ..., 0.95, 1.00}

η* = argmax_{η ∈ η_grid} ELPD(η)
```

The grid search is fully vectorised: the `(q, n_grid, D)` tensor of LOO predictions is computed in a single matrix operation.

**Why not use a differentiable optimiser to find η* continuously?** The ELPD surface over η is smooth but not convex. The surrogate and target gradients are not from the same distribution, so gradient-based η selection can get trapped. Grid search with 21 points finds the optimum reliably for this 1D problem with negligible cost (21 cosine dot products of D-dimensional vectors).

---

## 8. EMA Smoothing

The raw η* from a single step is noisy — a single outlier NES draw can shift it considerably. We apply exponential moving average smoothing:

```
η_smooth_t = α · η*_t  +  (1 - α) · η_smooth_{t-1}
```

with α = 0.3 (configurable). This means the current estimate gets 30% weight and history gets 70%. The effect is a smooth η trajectory that adapts to genuine alignment changes without reacting to individual noisy draws.

---

## 9. The Full Algorithm

```
Input: x_0, y, f_T, f_S, ε, α_step, T_attack, q, σ, η_grid

Initialise: x ← x_0, m ← 0, η_ema ← 0.5

For t = 1 to T_attack:
    1. g_S ← ∂L_S(x, y)/∂x                         [free, white-box]
    
    2. {g_i}_{i=1}^{2q} ← NES(f_T, x, σ, q)        [costs 2q queries]
       ĝ_T ← mean_i(g_i)
    
    3. Normalise: g_S ← g_S/||g_S||_∞
                  g_i ← g_i/||g_i||_∞  for each i
    
    4. For each η in η_grid:
         ĝ_loo_i(η) ← (q·ĝ_T - g_i)/(q-1)          [LOO mean]
         μ_i(η) ← η·g_S + (1-η)·ĝ_loo_i(η)
         score_i(η) ← cos(μ_i(η), g_i)
       ELPD(η) ← mean_i[score_i(η)] - T·std_i[score_i(η)]
    
    5. η* ← argmax_η ELPD(η)
       η_ema ← 0.3·η* + 0.7·η_ema                    [EMA smooth]
    
    6. g_blend ← η_ema·g_S + (1-η_ema)·ĝ_T            [in raw space]
    
    7. m ← μ·m + g_blend/||g_blend||_1                [momentum]
    
    8. x ← clip(x + α_step·sign(m), x_0±ε, [0,1])    [PGD step]
    
    9. If f_T(x) ≠ y: return x (success)              [early stop]

Return x (failure)
```

**Total queries per image:** `T_attack × 2q`  
At T=100, q=20: **4,000 queries** vs NES-only's 4,000, but with the surrogate accelerating convergence so the attack often succeeds in 20–40 steps → **800–1,600 effective queries**.

---

## 10. Why Previous ELPD Formulations Failed

We went through three failed attempts before arriving at LOO cosine. Full details in `04_bugs_and_fixes.md`, but the mathematical reasons are:

1. **Gaussian log-likelihood with fixed σ:** `log_const = -D/2·log(2πσ²)` dominated the score. With σ=0.001 and D=150528, this term is ≈ -10^7 and identical for all η. Surface flat.

2. **Negative MSE:** `-mean_i[||g_i - μ(η)||²]` is always maximised at η=0 because g_mean is the empirical centroid of the draws by definition. MSE is minimised by the centroid, so any blend with the surrogate increases it. Surface monotonically decreasing.

3. **Improvement formula:** `MSE(η=0) - MSE(η)` involves `r0_i = g_i - g_mean`, which sums to zero (by definition of the mean). So `mean_i[r0_i · Δg] = 0` for any Δg. Surface identically zero at η=0 and decreasing for η>0.

4. **LOO cosine (working):** The LOO mean `ĝ_loo_i ≠ g_i`, so the cosine score is genuinely informative. The surrogate can improve or worsen the prediction non-trivially.
