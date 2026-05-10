# 03 — Code Walkthrough

## Project Structure

```
adversarial_ml/
├── configs/
│   └── config.yaml              ← all hyperparameters, one source of truth
├── src/
│   ├── attacks/
│   │   ├── elpd_blender.py      ← THE core: LOO cosine η selection
│   │   ├── query_estimator.py   ← NES / SPSA gradient estimation
│   │   ├── elpd_attack.py       ← per-image attack loop
│   │   └── baselines.py         ← MI-FGSM, DI-FGSM, NES-only, static hybrid, Square
│   ├── models/
│   │   └── model_loader.py      ← TargetModel (black box) + SurrogateModel (white box)
│   └── utils/
│       ├── config_loader.py     ← YAML → dot-accessible Python namespace
│       └── logger.py            ← W&B + CSV dual logger
├── notebooks/
│   ├── phase1_scaffold_and_validation.ipynb
│   └── experiment.ipynb
├── results/
│   ├── logs/                    ← CSV outputs, config snapshots
│   └── plots/                   ← ELPD surface, Query-vs-ASR, η trajectory
└── explain/                     ← this folder
```

---

## `configs/config.yaml`

The single source of truth for every hyperparameter. Key sections:

```yaml
experiment:
  device: "mps"       # Apple Silicon GPU; change to "cuda" for AutoDL
  seed: 42

attack:
  epsilon: 0.05        # L∞ perturbation budget
  step_size: 0.005     # α — PGD step size
  num_steps: 100       # T — attack iterations
  query_budget: 5000   # hard cap on target queries per image

query_estimator:
  method: "nes"
  n_samples: 20        # q — antithetic pairs; total draws = 2q = 40
  sigma: 0.001         # finite-difference smoothing radius
  rao_blackwell: true  # variance reduction

elpd_blender:
  method: "waic"       # "waic" uses LOO cosine; "loo_psis" uses PSIS weighting
  eta_grid:
    n_points: 21       # {0.0, 0.05, ..., 1.0}
  waic:
    llik_temperature: 1.0   # T — penalty weight on std_i[score_i]
  eta_ema_alpha: 0.3         # EMA smoothing (0.3 = 30% current, 70% history)
```

---

## `src/utils/config_loader.py`

**Purpose:** Load `config.yaml` into a Python object where every value is accessible via dot notation (`cfg.attack.epsilon`) rather than dictionary syntax (`cfg["attack"]["epsilon"]`).

**Key function: `load_config(path)`**
1. Reads the YAML file with `yaml.safe_load()`
2. Recursively converts every nested dict to a `SimpleNamespace` object via `_dict_to_ns()`
3. Resolves relative paths to absolute (so `"data/"` becomes `/full/path/to/project/data/`)
4. Validates critical ranges (epsilon, step_size, n_samples) — fails loudly at load time, not mid-experiment

**Why `SimpleNamespace`?** It's a Python standard-library object that supports attribute access. `cfg.attack.epsilon` reads exactly like normal Python code. No third-party dependency.

---

## `src/attacks/elpd_blender.py`

This is the mathematical core. Three classes of things happen here.

### `_linf_normalise(g)`

```python
if g.dim() == 1:
    return g / (g.abs().max() + 1e-12)
else:
    max_vals = g.abs().amax(dim=1, keepdim=True)
    return g / (max_vals + 1e-12)
```

Projects a gradient onto the unit L∞ ball: every element is divided by the maximum absolute element. The result has `max(|g_i|) = 1`.

**Why L∞ normalisation (not L2)?** The attack uses `sign(g)` for its update step, which is equivalent to clipping to the L∞ ball. Normalising in L∞ space before ELPD computation means the cosine scores are computed in the same geometric space as the attack's actual step direction. L2 normalisation would distort the geometry.

### `_waic_elpd(g_sur, g_tgt, g_mean)` — The Main Estimator

**Step 1: LOO means (vectorised)**

```python
g_loo_mean = (q * g_mean.unsqueeze(0) - g_tgt) / (q - 1)   # (q, D)
```

For each draw index i, `g_loo_mean[i]` = mean of all draws EXCEPT i.  
Identity used: `mean_{j≠i}(g_j) = (q·ĝ - g_i) / (q-1)`.  
This computes all q LOO means simultaneously with one subtraction and one division — no Python loop needed.

**Step 2: Blended LOO predictions (broadcasted)**

```python
delta = g_sur.unsqueeze(0) - g_loo_mean           # (q, D)
mu_loo = (
    g_loo_mean.unsqueeze(1)                        # (q, 1, D)
    + self.eta_grid.view(1, -1, 1) * delta.unsqueeze(1)  # (q, n_grid, D)
)                                                  # result: (q, n_grid, D)
```

`mu_loo[i, j, :]` = the blended LOO prediction for hold-out draw i at grid point η_j.  
Shape `(q, n_grid, D)` = 20 draws × 21 η values × 150528 dimensions.  
This is the most memory-intensive operation. For ImageNet (D=150528): `20 × 21 × 150528 × 4 bytes ≈ 252 MB`. On MPS this is fine; on CPU it's slow.

**Step 3: Cosine scores**

```python
g_tgt_exp = g_tgt.unsqueeze(1)                    # (q, 1, D)
num   = (mu_loo * g_tgt_exp).sum(dim=-1)          # (q, n_grid)
denom = mu_loo.norm(dim=-1) * g_tgt_exp.norm(dim=-1)  # (q, n_grid)
cos_scores = num / denom                          # (q, n_grid)
```

For each (draw i, grid point j): cosine similarity between the blended prediction and the held-out draw.

**Step 4: ELPD score**

```python
lppd_per_eta = cos_scores.mean(dim=0)             # (n_grid,)
p_waic = cos_scores.std(dim=0, unbiased=True)     # (n_grid,)
return lppd_per_eta - T * p_waic                  # (n_grid,)
```

Mean over draws (signal) minus temperature × std over draws (inconsistency penalty).

### `_loo_psis_elpd(g_sur, g_tgt, g_mean)` — PSIS Variant

Uses the same cosine scores but weights each draw by Pareto-Smoothed Importance Sampling weights. When the IS weight distribution is well-behaved (Pareto k̂ < 0.7), this gives a lower-variance ELPD estimate than the simple mean. Falls back to simple mean when k̂ ≥ 0.7.

The Pareto shape k̂ is estimated by fitting a Generalised Pareto Distribution to the upper tail of the IS weights (Zhang & Stephens 2009 moment estimator). It's logged as a diagnostic.

### `_cosine_var_elpd(g_sur, g_tgt, g_mean)` — Low-Sample Fallback

Used when q < 4 draws (not enough for reliable variance estimation). Instead of LOO, it computes:

```
score(η) = cos(μ(η), ĝ) - λ · normalised_spread(η)
```

where `μ(η)` uses the full mean ĝ (not LOO), and `spread` is the mean squared deviation of draws from the blended mean. This is a heuristic, not a proper ELPD estimate.

### `step(g_surrogate, g_target_draws)` — Public Interface

The outer method that orchestrates everything:

1. L∞-normalise both inputs
2. Choose ELPD method based on sample count
3. Compute ELPD over the η grid
4. `argmax` to find η*
5. EMA smooth: `η_ema = 0.3·η* + 0.7·η_prev`
6. Blend in the **original (un-normalised)** gradient space — we chose η in normalised space but apply it in raw space to preserve step magnitude

Returns a `BlendResult` dataclass with: `eta_raw`, `eta_smoothed`, `g_blended`, `elpd_values`, `eta_grid`, `method_used`, `pareto_k`, `diagnostics`.

---

## `src/attacks/query_estimator.py`

### `nes_gradient(query_fn, x, sigma, n_samples, ...)`

```python
u = torch.randn(n_samples, D, device=device)
u = u / (u.norm(dim=1, keepdim=True) + 1e-12)    # unit L2 directions

x_pos = (x_flat + sigma * u).view(n_samples, C, H, W).clamp(0, 1)
x_neg = (x_flat - sigma * u).view(n_samples, C, H, W).clamp(0, 1)

x_batch = torch.cat([x_pos, x_neg], dim=0)        # 2q images in one batch
f_batch = query_fn(x_batch)                        # 2q loss values, 2q queries
```

The entire batch of 2q perturbed images is sent to the target in one call. This is critical for efficiency — a single batched forward pass is much faster than 2q individual calls.

```python
if rao_blackwell:
    baseline = f_batch.mean()    # optimal constant baseline
    f_pos = f_pos - baseline
    f_neg = f_neg - baseline

draws_pos = delta_f * u / (2.0 * sigma)           # per-draw gradient contributions
all_draws = torch.cat([draws_pos, -draws_pos], 0) # (2q, D) antithetic
g_hat = all_draws.mean(dim=0)                     # (D,) final estimate
```

**Why divide by 2σ?** The finite-difference gradient approximation is:
`∂f/∂x ≈ [f(x+σu) - f(x-σu)] / (2σ)` (central difference). Each `u_k` direction gives one estimate. Multiplying by `u_k` projects it back to the full D-dimensional gradient.

### `QueryEstimator.estimate(x)` — MPS Fix

```python
if device.type == 'mps':
    self._gen = None    # MPS doesn't support seeded generators
else:
    self._gen = torch.Generator(device=device)
    self._gen.manual_seed(self._seed)
```

Apple's Metal Performance Shaders (MPS) backend doesn't support PyTorch's `Generator` object for seeding random operations. We use `None` (relying on global seed) for MPS. This is set lazily on first call so it correctly detects the device.

---

## `src/models/model_loader.py`

### `TargetModel`

Wraps a torchvision pretrained model as a black box:

```python
def get_query_fn(self):
    def query_fn(x_batch):
        B = x_batch.shape[0]
        if self._count + B > self.budget:
            raise QueryBudgetExceeded(...)
        self._count += B
        with torch.no_grad():
            x_norm = (x_batch - self._mean) / self._std
            logits = self._model(x_norm)
            losses = F.cross_entropy(logits, labels, reduction="none")
        return losses   # (B,) loss values
    return query_fn
```

Key design choices:
- Returns a **closure** (function) rather than exposing the model directly — this is the correct interface for `QueryEstimator` and prevents accidental gradient computation through the target
- `torch.no_grad()` is enforced — the target never computes gradients
- Query budget is checked before each batch — raises `QueryBudgetExceeded` to cleanly exit the attack loop

### `SurrogateModel`

```python
def gradient(self, x, label):
    x_in = x.unsqueeze(0).requires_grad_(True)
    x_norm = (x_in - self._mean) / self._std
    logits = self._model(x_norm)
    loss = F.cross_entropy(logits, torch.tensor([label]))
    loss.backward()
    return x_in.grad.detach().view(-1)   # (D,) flat
```

Computes `∂L/∂x` through the surrogate. Note:
- `requires_grad_(True)` on the input — we want gradient w.r.t. x, not model weights
- Model weights have `requires_grad=False` (frozen at `__init__`) — this prevents PyTorch from building a computation graph for them, saving memory
- `.view(-1)` flattens C×H×W → D for the blender

---

## `src/attacks/elpd_attack.py`

The main per-image attack loop. Orchestrates: surrogate → query estimator → blender → momentum → PGD step → early stop.

```python
# Step 5: PGD step
x_flat = x.view(-1) + acfg.step_size * m.sign()
x_flat = x_orig.view(-1) + (x_flat - x_orig.view(-1)).clamp(-acfg.epsilon, acfg.epsilon)
x_flat = x_flat.clamp(acfg.clip_min, acfg.clip_max)
```

Two projections:
1. **L∞ ball projection:** `clamp(-ε, ε)` around x_orig ensures the perturbation never exceeds the budget
2. **Valid pixel range:** `clamp(0, 1)` ensures pixel values stay in the valid image range

Both are needed — they're not equivalent. A perturbation within ε of x_orig can still push pixels below 0 or above 1 near the image boundaries.

---

## `src/attacks/baselines.py`

### MI-FGSM
Standard Momentum Iterative FGSM using only surrogate gradients. Uses 0 target queries. This is the "pure transfer" baseline.

### DI-FGSM
Same as MI-FGSM but applies `_diverse_input()` before computing the surrogate gradient: randomly resizes the image and pads it back to the original size. This input diversity discourages the adversarial example from overfitting to the surrogate's specific architecture quirks.

```python
target_size = random.randint(270, 330)   # random resize for 224×224 images
x_resized = F.interpolate(x.unsqueeze(0), size=target_size, ...)
x_padded  = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))
```

### Square Attack
Score-based attack — no gradient at all. At each step, proposes a random square patch of ±ε and accepts it if it increases the loss (greedy hill-climbing in pixel space). The square size shrinks linearly over iterations. This is the strongest pure-score baseline because it makes no linearity assumptions.

---

## `src/utils/logger.py`

Dual W&B + CSV logger. On every `log_step()` call:
1. Appends to an in-memory buffer
2. Every `flush_every_n_steps` steps, writes the buffer to a CSV file
3. If W&B is enabled and available, also calls `wandb.log()` in real time

The CSV is always written as a fallback — so even if W&B fails (e.g. no internet on AutoDL), you lose nothing.

The η-per-step logging is the most important diagnostic: it lets you plot how the blending weight adapts over the course of the attack on each image.
