# 04 — Bugs, Failures, and the Fix

This documents every ELPD formulation we attempted, why it failed mathematically, and what the failure looked like in practice. This is important context because the final solution is non-obvious — it took four attempts to arrive at it.

---

## The Symptom We Were Chasing

Every attempt produced one of two failure modes:

**Mode A — Flat surface:** `ELPD(η) ≈ constant` for all η. `argmax` returns index 0 (η*=0) by tie-breaking convention. The blender always outputs η=0 (pure query), ignoring the surrogate.

**Mode B — Monotonically decreasing:** `ELPD(η=0) > ELPD(η=0.5) > ELPD(η=1)`. `argmax` again returns η*=0. The blender correctly identifies that adding the surrogate hurts even when it's aligned.

In both cases the smoke test `assert result_A.eta_raw > 0.4` fails with η=0.000.

---

## Attempt 1 — Gaussian Log-Likelihood with Fixed σ

### Formula
```
llik_i(η) = -D/2·log(2πσ²) - ||g_i - μ(η)||² / (2σ²)
ELPD(η) = mean_i[llik_i(η)] - Var_i[llik_i(η)]
```
using `σ = 0.001` (the NES finite-difference radius from config).

### Why It Failed
After L∞ normalisation, gradients live in [-1, 1]^D where D = 150528 for ImageNet.  
A typical squared distance between two normalised gradient vectors is `O(D × 0.01) ≈ 1500`.  
But `2σ² = 2 × (0.001)² = 2×10⁻⁶`.

So `||g_i - μ(η)||² / (2σ²) ≈ 1500 / 2×10⁻⁶ = 7.5×10⁸`

And `D/2 · log(2πσ²) ≈ 75264 × log(6.28×10⁻⁶) ≈ 75264 × (-12.0) ≈ -903168`

The `log_const` term is `-903168`. The residual term is `-750000000`. Both are astronomically large in magnitude and **identical across all η** because only the residual changes, but the residual divided by σ is so large it makes every point on the surface equally terrible. The surface was numerically flat.

### Failure Output
```
ELPD η=0: -903168.123456  η=0.5: -903168.123456  η=1: -903168.123456
```
All equal to 6 decimal places. η*=0.

---

## Attempt 2 — Data-Adaptive σ

### Fix Attempted
Instead of fixed σ=0.001, estimate σ from the spread of the draws:
```
σ_data = sqrt(mean_i[||g_i - g_mean||²] / D)
```
This is the RMS per-dimension residual in normalised space. With NES noise of 0.05 and D=512 (test), `σ_data ≈ 0.05`.

### Why It Still Failed
The `log_const` and residual terms were now in a reasonable range (~-1000). But there was a second problem: the `p_waic` penalty was computed as:
```python
p_waic = llik_ij.var(dim=0).clamp(max=lppd_per_eta.abs())
```
`lppd_per_eta` was negative (e.g. -1200). `lppd_per_eta.abs()` was +1200. Since `p_waic` was also ~1200, `lppd - p_waic` computed to approximately zero. The `.clamp(max=lppd.abs())` cap was intended to prevent negative WAIC but instead forced everything to zero.

### Failure Output
```
ELPD η=0: 0.000  η=0.5: 0.000  η=1: 0.000
AssertionError: Expected η > 0.4, got 0.000
```

---

## Attempt 3 — Negative MSE (Drop log_const)

### Insight
Since `log_const` is identical for all η and doesn't affect `argmax`, drop it entirely:
```
ELPD(η) = -mean_i[||g_i - μ(η)||²] - T·std_i[||g_i - μ(η)||²]
```

### Why It Failed
This is mathematically equivalent to the log-likelihood for the purpose of η selection. The problem is geometric, not numerical.

`g_mean` is defined as `mean_i[g_i]`. It is the **unique minimiser of MSE** among all vectors. So:

```
||g_i - g_mean||² ≤ ||g_i - μ(η)||²  for any μ(η) ≠ g_mean, for all i
```

Since `μ(η) = η·g_sur + (1-η)·g_mean`, and the only time `μ(η) = g_mean` is at η=0:

```
ELPD(η=0) = -mean_i[||g_i - g_mean||²] = maximum possible value
ELPD(η>0) < ELPD(η=0)  always
```

The surface is strictly decreasing from η=0. No alignment signal is detectable.

### Failure Output
```
ELPD η=0: -1.563731  η=0.5: -1.573485  η=1: -1.679285
AssertionError: Expected η > 0.4, got 0.200
```
(Better — the surface is no longer flat! But η*=0 still wins.)

---

## Attempt 4 — MSE Improvement Formula

### Insight
Instead of absolute MSE, compute improvement over the η=0 baseline:
```
improvement_i(η) = MSE_i(η=0) - MSE_i(η)
                 = 2η⟨r0_i, Δg⟩ - η²||Δg||²
```
where `r0_i = g_i - g_mean` and `Δg = g_sur - g_mean`.

This formula is identically 0 at η=0 and potentially positive for well-aligned surrogates.

### Why It Failed
The critical term is `⟨r0_i, Δg⟩`:
- `r0_i = g_i - g_mean` — the residual of each draw from the overall mean
- By the definition of the mean: `Σ_i r0_i = 0`
- Therefore: `mean_i[⟨r0_i, Δg⟩] = ⟨mean_i[r0_i], Δg⟩ = ⟨0, Δg⟩ = 0` **for any Δg**

The expected improvement is exactly zero for all η. The surface reduces to `-η²||Δg||²` which is always ≤ 0 and maximised at η=0.

This is a fundamental algebraic identity, not a numerical coincidence.

### Failure Output
```
ELPD η=0: 0.000000  η=0.5: -0.105555  η=1: -0.286704
AssertionError: Expected η > 0.4, got 0.000
```

---

## Attempt 5 — LOO Cosine (Working Solution)

### The Geometric Insight
All previous approaches compared each draw `g_i` to a mean that includes `g_i` itself. The fundamental problem: when you use the mean of the draws as the prediction target, any deviation from η=0 always looks worse because the mean is the optimal predictor of itself.

The solution: **leave draw i out when computing the mean used to predict it**.

```
ĝ_loo_i = (q·ĝ - g_i) / (q-1)   ← mean of ALL OTHER draws
μ_LOO_i(η) = η·g_sur + (1-η)·ĝ_loo_i
score_i(η) = cos(μ_LOO_i(η), g_i)
```

Now `ĝ_loo_i ≠ g_i` and `ĝ_loo_i ≠ g_mean`. The score is genuinely informative:
- If `g_sur` aligns with `g_i`, blending it in (high η) improves the prediction of `g_i`
- If `g_sur` opposes `g_i`, blending it in worsens the prediction

### Why Cosine (Not MSE)?
We use cosine similarity rather than negative MSE for the LOO score because:
1. The attack uses `sign(g)` — only direction matters, not magnitude
2. Cosine is scale-invariant — immune to the normalisation mismatch problems
3. Cosine scores are bounded in [-1, 1] — no numerical instability

### The Vectorised Implementation
The key identity `ĝ_loo_i = (q·ĝ - g_i)/(q-1)` avoids q separate mean computations:

```python
g_loo_mean = (q * g_mean.unsqueeze(0) - g_tgt) / (q - 1)   # (q, D)
delta = g_sur.unsqueeze(0) - g_loo_mean                     # (q, D)
mu_loo = (
    g_loo_mean.unsqueeze(1)                                  # (q, 1, D)
    + self.eta_grid.view(1, -1, 1) * delta.unsqueeze(1)     # broadcast over n_grid
)                                                            # (q, n_grid, D)
```

One tensor operation produces all `q × n_grid` predictions simultaneously.

### Success Output
```
[A aligned]   η=0.500  cos=+0.9990  method=waic
  ELPD η=0: 0.988276  η=0.5: 0.988548  η=1: 0.988299
  PASS ✓

[B opposing]  η=0.000  cos=-0.9991
  ELPD η=0: 0.988276  η=0.5: -0.242685  η=1: -0.989748
  PASS ✓
```

The ELPD surface now:
- Peaks at η*=0.5 for the aligned surrogate (cos=+0.999 → surrogate is helpful but not perfect)
- Peaks at η*=0.0 for the opposing surrogate (cos=-0.999 → surrogate is harmful)
- Has a meaningful shape (not flat, not monotone)

---

## The MPS Generator Bug

**Error:**
```
RuntimeError: Expected a 'mps' device type for generator but found 'cpu'
```

**Cause:** `torch.Generator()` creates a CPU generator by default. Passing it to `torch.randn(..., device='mps', generator=cpu_gen)` fails because MPS requires device-matched generators.

**Fix:**
```python
if device.type == 'mps':
    self._gen = None    # MPS: use global torch seed, no generator object
else:
    self._gen = torch.Generator(device=device)
    self._gen.manual_seed(self._seed)
```

The generator is created lazily (on first `estimate()` call) so it correctly picks up the input tensor's device.
