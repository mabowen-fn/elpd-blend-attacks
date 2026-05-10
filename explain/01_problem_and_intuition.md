# 01 — Problem & Intuition

## The Big Idea in One Sentence

We adapted a statistical technique from clinical trial design — where doctors must combine a small perfectly-controlled study with a large messy real-world dataset — to make black-box adversarial attacks dramatically more query-efficient.

---

## Background: What Is a Black-Box Adversarial Attack?

Modern neural networks (image classifiers, speech models, etc.) can be fooled by adding a tiny, humanly-invisible perturbation to an input. A photo of a panda, with a barely perceptible noise pattern added, is classified as a gibbon with 99% confidence.

**The threat model we care about:**
- We are attacking a model we cannot inspect (a black box) — like a commercial API.
- We can only *query* it: send an image, get back a prediction (and optionally a loss value).
- Queries are expensive: each one costs money or risks detection.
- We want to fool the model with as few queries as possible.

### Two Existing Strategies (and Why They Both Fall Short)

**Strategy 1 — Pure Transfer Attack (MI-FGSM, DI-FGSM)**
- Train a *surrogate* model you control (e.g. VGG-16).
- Compute gradients on the surrogate — free and unlimited.
- Hope the adversarial example *transfers* to the black-box target.
- **Problem:** Different architectures have different loss landscapes. The surrogate gradient often points in the wrong direction for the target. Transfer rate is 30–60% for cross-architecture attacks.

**Strategy 2 — Pure Query Attack (NES, Square Attack)**
- Estimate the target's gradient directly via finite differences.
- Perturb the input by a tiny amount, measure loss change, infer gradient direction.
- **Problem:** Each gradient estimate costs 2q queries (where q ≈ 20–50). A full attack needs 100–500 steps → 4,000–50,000 queries. Very slow and expensive.

### The Hybrid Idea (and Why Static Hybrids Fail)

The obvious fix: blend surrogate gradients with query-estimated gradients:

```
g_attack = η · g_surrogate  +  (1 - η) · g_target_estimate
```

If `η = 1`: pure transfer. If `η = 0`: pure query. Any `η ∈ (0, 1)`: a blend.

**The problem:** what should η be? Papers that try this use a fixed η (e.g. 0.5) for the entire attack. But the surrogate's usefulness *changes dynamically*:
- **Early in the attack** (near the clean image): surrogate and target gradients are often well-aligned. High η is correct.
- **Late in the attack** (near the decision boundary): the perturbation has pushed the image into a region where the surrogate's loss landscape diverges from the target's. Low η is correct.

A static η is a compromise that's wrong for most of the attack.

---

## The Medical Analogy — Where the Math Comes From

The paper *"Combining Experimental and Observational Data through a Power Likelihood"* (Gower et al.) solves an almost identical problem in clinical trials:

| Clinical Trial | Our Attack |
|---|---|
| Small Randomised Controlled Trial (RCT) | Small set of target gradient estimates (expensive but unbiased) |
| Large Real-World Data (observational) | Unlimited surrogate gradients (free but biased) |
| "How much should we trust the biased large dataset?" | "How much should we trust the biased surrogate?" |
| Power η ∈ [0,1] on the observational likelihood | Blend weight η ∈ [0,1] on the surrogate gradient |
| ELPD (Expected Log Predictive Density) to select η | ELPD-analogue (LOO cosine score) to select η |

The key insight from the paper: instead of choosing η by intuition or cross-validation over a fixed dataset, they select η by **maximising the Expected Log Predictive Density (ELPD)** — a Bayesian model-selection criterion that measures how well the combined model predicts held-out data.

We translate this: at each attack step, the q NES query estimates are our "held-out unbiased data". We find the η that makes the blended gradient best predict those held-out estimates.

---

## What Makes This Novel

1. **Dynamic η per step.** No prior hybrid attack adapts η based on measured surrogate alignment. We compute it fresh at every step from actual gradient data.

2. **Bayesian model selection applied to adversarial attacks.** The ELPD criterion has a 40-year history in statistics. This is (to our knowledge) the first application to adaptive gradient blending in adversarial ML.

3. **The η trajectory is a research artifact.** By logging η at every step, we can *prove* that the surrogate is more useful near the clean image and less useful near the decision boundary — a finding that validates the entire hybrid approach theoretically.

4. **Self-calibrating with zero hyperparameter tuning.** The LOO cosine ELPD score requires no σ, no learning rate, no temperature schedule. It reads alignment directly from the query draws.

---

## Expected Results

The key figure we are producing: **Query Budget vs Attack Success Rate (ASR)**.

We expect the ELPD-Blend curve to reach 90% ASR using 2–4× fewer queries than NES-only, and to achieve 15–25 percentage points higher ASR than pure transfer at any fixed query budget.
