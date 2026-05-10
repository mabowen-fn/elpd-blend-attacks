"""
baselines.py — Baseline attack implementations for fair comparison.

All baselines expose the same interface:
    run_<name>(x_orig, true_label, ..., cfg) → AttackResult

Baselines:
  1. MI-FGSM   — pure transfer, momentum iterative FGSM on surrogate
  2. DI-FGSM   — pure transfer with input diversity (random resize+pad)
  3. NES-only  — pure query attack, no surrogate
  4. Static-Hybrid — fixed η blend (η from config, no ELPD adaptation)
  5. Square    — score-based black-box (no gradient at all)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor

from .elpd_attack    import AttackResult
from .elpd_blender   import ELPDBlender
from .query_estimator import QueryEstimator

logger = logging.getLogger(__name__)


# ── MI-FGSM (pure transfer) ───────────────────────────────────────────────────

def run_mifgsm(
    x_orig:     Tensor,
    true_label: int,
    surrogate,
    target,
    cfg:        SimpleNamespace,
) -> AttackResult:
    """Momentum Iterative FGSM using surrogate gradients only (0 target queries)."""
    acfg = cfg.attack
    mu   = cfg.momentum.mu

    x = x_orig.clone()
    m = torch.zeros_like(x.view(-1))

    eta_history:   list[float] = []
    query_history: list[int]   = []

    for step in range(acfg.num_steps):
        g = surrogate.gradient(x, true_label)
        g_norm = g / (g.abs().sum() + 1e-12)
        m = mu * m + g_norm

        x_flat = x.view(-1) + acfg.step_size * m.sign()
        x_flat = x_orig.view(-1) + (x_flat - x_orig.view(-1)).clamp(-acfg.epsilon, acfg.epsilon)
        x_flat = x_flat.clamp(acfg.clip_min, acfg.clip_max)
        x      = x_flat.view_as(x_orig)

        eta_history.append(1.0)      # pure transfer → η always 1
        query_history.append(0)

        if acfg.early_stop and target.predict(x) != true_label:
            return AttackResult(True, x, 0, step + 1, eta_history, query_history)

    success = target.predict(x) != true_label
    return AttackResult(success, x, 0, acfg.num_steps, eta_history, query_history)


# ── DI-FGSM (pure transfer with input diversity) ──────────────────────────────

def _diverse_input(x: Tensor, p: float = 0.5, resize_range: tuple = (270, 330)) -> Tensor:
    """
    Random resize-and-pad transform (Xie et al. 2019).
    Applied with probability p; identity otherwise.
    """
    if random.random() > p:
        return x
    C, H, W     = x.shape
    target_size = random.randint(*resize_range)
    x_resized   = F.interpolate(
        x.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
    ).squeeze(0)
    pad_h = H - target_size
    pad_w = W - target_size
    pad_top  = random.randint(0, max(pad_h, 0))
    pad_left = random.randint(0, max(pad_w, 0))
    x_padded = F.pad(
        x_resized,
        (pad_left, max(pad_w, 0) - pad_left, pad_top, max(pad_h, 0) - pad_top),
    )
    return x_padded[:, :H, :W]


def run_difgsm(
    x_orig:     Tensor,
    true_label: int,
    surrogate,
    target,
    cfg:        SimpleNamespace,
    diversity_prob: float = 0.5,
) -> AttackResult:
    """DI-FGSM: MI-FGSM with input diversity on the surrogate gradient."""
    acfg = cfg.attack
    mu   = cfg.momentum.mu

    x = x_orig.clone()
    m = torch.zeros_like(x.view(-1))
    eta_history:   list[float] = []
    query_history: list[int]   = []

    for step in range(acfg.num_steps):
        x_di = _diverse_input(x, p=diversity_prob)
        g    = surrogate.gradient(x_di, true_label)
        g_norm = g / (g.abs().sum() + 1e-12)
        m = mu * m + g_norm

        x_flat = x.view(-1) + acfg.step_size * m.sign()
        x_flat = x_orig.view(-1) + (x_flat - x_orig.view(-1)).clamp(-acfg.epsilon, acfg.epsilon)
        x_flat = x_flat.clamp(acfg.clip_min, acfg.clip_max)
        x      = x_flat.view_as(x_orig)

        eta_history.append(1.0)
        query_history.append(0)

        if acfg.early_stop and target.predict(x) != true_label:
            return AttackResult(True, x, 0, step + 1, eta_history, query_history)

    success = target.predict(x) != true_label
    return AttackResult(success, x, 0, acfg.num_steps, eta_history, query_history)


# ── NES-only (pure query) ─────────────────────────────────────────────────────

def run_nes_only(
    x_orig:     Tensor,
    true_label: int,
    target,
    estimator:  QueryEstimator,
    cfg:        SimpleNamespace,
) -> AttackResult:
    """Pure NES attack — no surrogate, η=0 always."""
    acfg = cfg.attack
    mu   = cfg.momentum.mu

    estimator.reset()
    x = x_orig.clone()
    m = torch.zeros_like(x.view(-1))
    eta_history:   list[float] = []
    query_history: list[int]   = []

    for step in range(acfg.num_steps):
        try:
            g_hat, _, n_q = estimator.estimate(x)
        except Exception as e:
            if "budget" in str(e).lower():
                break
            raise

        g_norm = g_hat / (g_hat.abs().sum() + 1e-12)
        m = mu * m + g_norm

        x_flat = x.view(-1) + acfg.step_size * m.sign()
        x_flat = x_orig.view(-1) + (x_flat - x_orig.view(-1)).clamp(-acfg.epsilon, acfg.epsilon)
        x_flat = x_flat.clamp(acfg.clip_min, acfg.clip_max)
        x      = x_flat.view_as(x_orig)

        q_cumul = estimator.total_queries
        eta_history.append(0.0)
        query_history.append(q_cumul)

        if acfg.early_stop and target.predict(x) != true_label:
            return AttackResult(True, x, q_cumul, step + 1, eta_history, query_history)

    q_cumul = estimator.total_queries
    success = target.predict(x) != true_label
    return AttackResult(success, x, q_cumul, acfg.num_steps, eta_history, query_history)


# ── Static Hybrid (fixed η) ───────────────────────────────────────────────────

def run_static_hybrid(
    x_orig:     Tensor,
    true_label: int,
    surrogate,
    target,
    estimator:  QueryEstimator,
    cfg:        SimpleNamespace,
) -> AttackResult:
    """Static η hybrid: blend surrogate and NES gradients at a fixed ratio."""
    acfg    = cfg.attack
    mu      = cfg.momentum.mu
    eta_fix = cfg.baselines.run_static_hybrid.eta_static

    estimator.reset()
    x = x_orig.clone()
    m = torch.zeros_like(x.view(-1))
    eta_history:   list[float] = []
    query_history: list[int]   = []

    for step in range(acfg.num_steps):
        g_sur = surrogate.gradient(x, true_label)

        try:
            g_hat, _, n_q = estimator.estimate(x)
        except Exception as e:
            if "budget" in str(e).lower():
                break
            raise

        g_blend = eta_fix * g_sur + (1.0 - eta_fix) * g_hat
        g_norm  = g_blend / (g_blend.abs().sum() + 1e-12)
        m = mu * m + g_norm

        x_flat = x.view(-1) + acfg.step_size * m.sign()
        x_flat = x_orig.view(-1) + (x_flat - x_orig.view(-1)).clamp(-acfg.epsilon, acfg.epsilon)
        x_flat = x_flat.clamp(acfg.clip_min, acfg.clip_max)
        x      = x_flat.view_as(x_orig)

        q_cumul = estimator.total_queries
        eta_history.append(eta_fix)
        query_history.append(q_cumul)

        if acfg.early_stop and target.predict(x) != true_label:
            return AttackResult(True, x, q_cumul, step + 1, eta_history, query_history)

    q_cumul = estimator.total_queries
    success = target.predict(x) != true_label
    return AttackResult(success, x, q_cumul, acfg.num_steps, eta_history, query_history)


# ── Square Attack (score-based, no gradient) ──────────────────────────────────

def run_square_attack(
    x_orig:     Tensor,
    true_label: int,
    target,
    cfg:        SimpleNamespace,
    p_init:     float = 0.05,    # initial square side as fraction of image size
) -> AttackResult:
    """
    Square Attack (Andriushchenko et al. 2020) — score-based, no gradient.

    At each step, randomly proposes a square patch of ±ε and accepts it if
    the loss increases.  The square side shrinks on a log schedule.

    This is the strongest pure-score baseline because it has no gradient
    assumptions and works well against defences that obfuscate gradients.
    """
    acfg     = cfg.attack
    device   = x_orig.device
    C, H, W  = x_orig.shape
    epsilon  = acfg.epsilon

    x        = x_orig.clone()
    # Initialise with a random ±ε vertical stripe (as in the original paper)
    delta    = torch.zeros_like(x)
    for c in range(C):
        delta[c, :, 0] = random.choice([-epsilon, epsilon])
    x = (x_orig + delta).clamp(acfg.clip_min, acfg.clip_max)

    query_fn   = target.get_query_fn()
    best_loss  = query_fn(x.unsqueeze(0))[0].item()
    q_used     = 1
    eta_history:   list[float] = []
    query_history: list[int]   = []

    n_iters = acfg.query_budget
    for i in range(1, n_iters):
        # Adaptive square size: shrink as we progress
        frac   = i / n_iters
        p      = p_init * (1.0 - frac) + 0.01 * frac  # linear schedule
        s      = max(int(round(p * min(H, W))), 1)
        r      = random.randint(0, H - s)
        c_col  = random.randint(0, W - s)
        sign   = random.choice([-1.0, 1.0])

        x_new  = x.clone()
        for ch in range(C):
            x_new[ch, r:r+s, c_col:c_col+s] = (
                x_orig[ch, r:r+s, c_col:c_col+s] + sign * epsilon
            ).clamp(acfg.clip_min, acfg.clip_max)
        # Ensure global L∞ constraint
        x_new = (x_orig + (x_new - x_orig).clamp(-epsilon, epsilon)).clamp(
            acfg.clip_min, acfg.clip_max
        )

        try:
            loss_new = query_fn(x_new.unsqueeze(0))[0].item()
            q_used  += 1
        except Exception as e:
            if "budget" in str(e).lower():
                break
            raise

        if loss_new > best_loss:
            x         = x_new
            best_loss = loss_new

        eta_history.append(0.0)
        query_history.append(q_used)

        if acfg.early_stop and target.predict(x) != true_label:
            return AttackResult(True, x, q_used, i + 1, eta_history, query_history)

    success = target.predict(x) != true_label
    return AttackResult(success, x, q_used, n_iters, eta_history, query_history)
