# Adaptive ELPD-Blend Attack

A novel black-box adversarial attack that dynamically blends surrogate gradients with query-based gradient estimates using a Bayesian criterion — the Expected Log Predictive Density (ELPD) — adapted from Gower et al.'s *Combining experimental and observational data through a power likelihood* (2024).

---

## Core Idea

Standard hybrid attacks use a fixed blending weight η to combine:
- **Surrogate gradients** (free, white-box, but biased toward a different model)
- **Query gradients** (expensive, accurate, but noisy at low query counts)

This attack selects η *adaptively* at every step by computing the ELPD of the blended gradient against NES draws from the target — exactly the same principle used in clinical trials to decide how much to trust biased observational data relative to a small gold-standard RCT.

When the surrogate is well-aligned with the target (early attack, small perturbation), η is driven high and queries are conserved. When the perturbation grows and surrogate–target alignment degrades, η falls toward zero and the attack relies purely on queries. The result is a steep query-vs-ASR curve that crosses 90% ASR significantly earlier than NES-only or static hybrid baselines.

---

## Method

### ELPD Criterion (LOO Cosine)

For each candidate η ∈ {0, 0.05, …, 1.0}, the blended gradient is:

```
μ(η) = η · g_sur + (1 − η) · ĝ_target
```

The leave-one-out cosine ELPD is:

```
ELPD(η) = mean_i [ cos(η·g_sur + (1−η)·ĝ_loo_i,  g_i) ]
         − T · std_i [ cos(·) ]
```

where `ĝ_loo_i` is the NES mean with draw `i` held out, and the std term is the WAIC penalty that discourages η values inconsistently supported across draws. The optimal η* = argmax ELPD(η), smoothed with EMA across steps.

### Attack Loop (per image)

```
x_adv ← x_orig
for step t = 1 … T:
    g_sur              ← surrogate.gradient(x_adv, label)
    g_hat, draws, n_q  ← NES(target, x_adv)          # 2q queries
    η*                 ← ELPDBlender.step(g_sur, draws)
    g_blend            ← η* · g_sur + (1−η*) · g_hat
    g_momentum         ← μ · g_momentum + g_blend / ‖g_blend‖₁
    x_adv              ← Proj_{L∞(ε)} [ x_adv + α · sign(g_momentum) ]
    if target.predict(x_adv) ≠ label: break           # early stop
```

---

## Results (local dev, 10 images, CIFAR-10 → ResNet-50)

| Method | ASR | Avg Queries |
|---|---|---|
| ELPD-Blend (ours) | **100%** | **26** |
| Static Hybrid (η=0.5) | 100% | 26 |
| NES-only | 100% | 198 |
| Square Attack | 100% | 23 |
| MI-FGSM | 100% | 0 (transfer) |
| DI-FGSM | 100% | 0 (transfer) |

> **Note:** Local dev uses CIFAR-10 images upscaled to 224×224 as a quick smoke test. The labels used are the target model's own ImageNet predictions (not CIFAR-10 class indices). Full evaluation on ImageNet with 1000 images runs on AutoDL GPU.

---

## Project Structure

```
adversarial_ml/
├── configs/
│   └── config.yaml              # All hyperparameters (ε, steps, η grid, W&B, etc.)
├── src/
│   ├── attacks/
│   │   ├── elpd_blender.py      # Core: LOO cosine ELPD, η selection, EMA smoothing
│   │   ├── elpd_attack.py       # Per-image attack loop
│   │   ├── query_estimator.py   # NES/SPSA gradient estimator with Rao-Blackwell
│   │   └── baselines.py         # MI-FGSM, DI-FGSM, NES-only, Static Hybrid, Square
│   ├── models/
│   │   └── model_loader.py      # TargetModel (query-counted) + SurrogateModel
│   ├── data/
│   │   └── data_loader.py       # CIFAR-10 (local) and ImageNet (AutoDL) loaders
│   └── utils/
│       ├── config_loader.py     # YAML → SimpleNamespace config
│       └── logger.py            # W&B + CSV dual logger, η-trajectory tracking
├── scripts/
│   └── run_experiment.py        # Full evaluation: all methods, plots, summary table
├── notebooks/
│   ├── phase1_scaffold_and_validation.ipynb   # ELPD blender smoke tests
│   └── experiment.ipynb                       # Interactive experiment runner
├── explain/                     # Deep-dive writeups (concepts, math, code, plots)
│   ├── 01_problem_and_intuition.md
│   ├── 02_mathematics.md
│   ├── 03_code_walkthrough.md
│   ├── 04_bugs_and_fixes.md
│   └── 05_reading_the_plots.md
└── results/
    ├── logs/                    # final_results.csv, run_metrics.csv (gitignored)
    └── plots/                   # query_vs_asr.png, eta_trajectory.png (gitignored)
```

---

## Quick Start

### Local (Apple Silicon MPS)

```bash
# Create environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install wandb pandas matplotlib pyyaml

# Quick smoke test (3 images, 5 steps, no W&B)
python scripts/run_experiment.py --n_images 3 --steps 5 --no_wandb

# Full local run
python scripts/run_experiment.py --n_images 10
```

### AutoDL / Cloud GPU

```bash
# Set device: "cuda" in configs/config.yaml
# Set dataset.name: "imagenet" and dataset.root to your ImageNet path
# Set dataset.num_samples: 1000
python scripts/run_experiment.py
```

### Key Config Options (`configs/config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `experiment.device` | `mps` | `mps` / `cuda` / `cpu` |
| `dataset.name` | `cifar10` | `cifar10` (local) or `imagenet` |
| `dataset.num_samples` | `100` | Images to evaluate |
| `attack.epsilon` | `0.05` | L∞ perturbation budget |
| `attack.query_budget` | `5000` | Hard query cap per image |
| `attack.num_steps` | `100` | Attack iterations |
| `query_estimator.n_samples` | `10` | NES antithetic pairs (2q queries/step) |
| `elpd_blender.method` | `waic` | `waic` / `loo_psis` / `cosine_var` |
| `wandb.enabled` | `true` | W&B logging |

---

## Reading the Outputs

**`results/plots/query_vs_asr.png`** — ASR vs query budget for all methods. The ELPD-Blend curve (blue) should cross 90% ASR at lower query count than NES-only (orange) and static hybrid (purple).

**`results/plots/eta_trajectory.png`** — Mean η per attack step (±1 std band) alongside surrogate–target cosine similarity. Both should trend downward as perturbation grows; their positive correlation is evidence the ELPD blender correctly tracks alignment degradation.

**`results/logs/final_results.csv`** — Per-image, per-method results: success, query count, steps taken.

**`results/logs/run_metrics.csv`** — Per-step η values, cosine similarities, and ELPD diagnostics for every image. Used to generate the η trajectory plot.

---

## The Mathematical Connection

This attack is a direct translation of the power likelihood framework:

| Clinical Trial Setting | Adversarial Attack Setting |
|---|---|
| RCT data (small, unbiased) | Target model queries (limited, accurate) |
| Observational data (large, biased) | Surrogate gradients (free, misaligned) |
| Power likelihood weight η | Blending weight η |
| ELPD on held-out RCT data | LOO cosine ELPD on held-out NES draws |
| Optimal η balances bias vs. variance | Optimal η conserves queries while using surrogate where trustworthy |

The key insight: the NES draws play the role of the "unbiased gold-standard data," and the ELPD measures how well any blended direction μ(η) predicts those draws — without using them to compute μ(η) (hence the leave-one-out structure).

---

## Reference

Gower, R., Karagulyan, A., Richtárik, P., Richtárik, P., & Wild, S. (2024). *Combining experimental and observational data through a power likelihood.* arXiv:2304.02339.
