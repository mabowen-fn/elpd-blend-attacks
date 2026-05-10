"""
elpd_attack.py — Per-image ELPD-Blend attack loop.

Algorithm (one image):
    1. Compute surrogate gradient g_sur via backprop (free)
    2. Estimate target gradient ĝ_target via NES + collect draws (q queries)
    3. Feed to ELPDBlender → get η* and g_blended
    4. Accumulate g_blended into momentum buffer m
    5. PGD step: x ← clip(x + α · sign(m), x_orig ± ε)
    6. Early-stop if target prediction flips
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor

from .elpd_blender    import ELPDBlender, BlendResult
from .query_estimator import QueryEstimator

logger = logging.getLogger(__name__)


class QueryBudgetExceeded(Exception):
    pass


@dataclass
class AttackResult:
    success:       bool
    adv_x:         Tensor     # (C, H, W) final adversarial example
    queries_used:  int
    steps_taken:   int
    eta_history:   list[float]   # η_smoothed per step
    query_history: list[int]     # cumulative queries at each step


def elpd_blend_attack(
    x_orig:      Tensor,          # (C, H, W) clean image, [0,1]
    true_label:  int,
    surrogate,                    # SurrogateModel instance
    target,                       # TargetModel instance
    blender:     ELPDBlender,
    estimator:   QueryEstimator,
    cfg:         SimpleNamespace,  # full experiment config
    run_logger=None,              # RunLogger | None
) -> AttackResult:
    """
    Run the ELPD-Blend attack on a single image.

    Args:
        x_orig:      Clean input image (C, H, W) in [0, 1].
        true_label:  Ground-truth class index.
        surrogate:   White-box SurrogateModel.
        target:      Black-box TargetModel (query counter managed externally).
        blender:     ELPDBlender instance (reset before calling).
        estimator:   QueryEstimator bound to target.query_fn.
        cfg:         Full config namespace.
        run_logger:  Optional RunLogger for W&B / CSV logging.

    Returns:
        AttackResult with success flag, adversarial example, and diagnostics.
    """
    device = x_orig.device
    acfg   = cfg.attack
    mcfg   = cfg.momentum

    blender.reset()
    estimator.reset()

    x     = x_orig.clone()
    m     = torch.zeros_like(x.view(-1))   # momentum buffer, flat (D,)
    mu    = mcfg.mu if mcfg.enabled else 0.0

    eta_history:   list[float] = []
    query_history: list[int]   = []
    asr_running    = 0.0       # updated externally; placeholder here

    for step in range(acfg.num_steps):
        # ── 1. Surrogate gradient (free) ──────────────────────────────────
        g_sur = surrogate.gradient(x, true_label)             # (D,)

        # ── 2. Target gradient estimate (costs queries) ────────────────────
        try:
            _, draws, n_q = estimator.estimate(x.view(acfg.step_size and x.shape[0],
                                                       *x.shape) if False else x)
        except Exception as e:
            if "budget" in str(e).lower():
                logger.info("Query budget exhausted at step %d.", step)
                break
            raise

        # ── 3. ELPD blend ─────────────────────────────────────────────────
        blend: BlendResult = blender.step(g_sur, draws)
        g_blend = blend.g_blended                             # (D,)

        # ── 4. Momentum (MI-FGSM style) ───────────────────────────────────
        # Normalise by ℓ1 norm before accumulating — standard MI-FGSM trick
        g_norm = g_blend / (g_blend.abs().sum() + 1e-12)
        m      = mu * m + g_norm

        # ── 5. PGD step ───────────────────────────────────────────────────
        x_flat  = x.view(-1)
        x_flat  = x_flat + acfg.step_size * m.sign()
        # Project onto ε-ball around x_orig
        x_flat  = x_orig.view(-1) + (x_flat - x_orig.view(-1)).clamp(
            -acfg.epsilon, acfg.epsilon
        )
        # Clip to valid pixel range
        x_flat  = x_flat.clamp(acfg.clip_min, acfg.clip_max)
        x       = x_flat.view_as(x_orig)

        q_cumul = estimator.total_queries
        eta_history.append(blend.eta_smoothed)
        query_history.append(q_cumul)

        if run_logger:
            run_logger.log_step(
                step,
                eta_raw=blend.eta_raw,
                eta_smoothed=blend.eta_smoothed,
                queries_this_step=n_q,
                queries_cumulative=q_cumul,
                asr_running=asr_running,
                diagnostics=blend.diagnostics,
            )

        # ── 6. Early stop ─────────────────────────────────────────────────
        if acfg.early_stop:
            pred = target.predict(x)
            if pred != true_label:
                logger.debug("Attack succeeded at step %d (queries=%d).", step, q_cumul)
                return AttackResult(
                    success=True, adv_x=x,
                    queries_used=q_cumul, steps_taken=step + 1,
                    eta_history=eta_history, query_history=query_history,
                )

    # Final prediction check after all steps
    pred    = target.predict(x)
    success = (pred != true_label)
    return AttackResult(
        success=success, adv_x=x,
        queries_used=estimator.total_queries, steps_taken=acfg.num_steps,
        eta_history=eta_history, query_history=query_history,
    )
