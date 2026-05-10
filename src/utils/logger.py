"""
logger.py — Dual W&B + CSV logger with η-trajectory tracking.

Structured around a single RunLogger object that the attack loop calls at each
step.  W&B is used when available and enabled; CSV is always written as a
fallback so results survive W&B outages on AutoDL.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class RunLogger:
    """
    One instance per experiment run.  Thread-safe for single-process use.

    Usage:
        rlog = RunLogger(cfg)
        rlog.start_image(img_idx, true_label)
        for step in range(T):
            rlog.log_step(step, eta=..., queries=..., asr=..., ...)
        rlog.end_image(success=True, total_queries=...)
        rlog.finish()
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        self.cfg          = cfg
        self._wandb_run   = None
        self._csv_path    = Path(cfg.csv_logging.path)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file    = None
        self._csv_writer  = None
        self._step_buffer: list[dict] = []
        self._flush_every = cfg.csv_logging.flush_every_n_steps
        self._img_idx     = 0
        self._img_label   = 0
        self._run_start   = time.time()

        self._init_wandb()
        self._init_csv()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        if not (self.cfg.wandb.enabled and _WANDB_AVAILABLE):
            if self.cfg.wandb.enabled:
                logger.warning("wandb not installed — logging to CSV only.")
            return
        self._wandb_run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity or None,
            name=self.cfg.experiment.name,
            config=_ns_to_dict(self.cfg),
            tags=self.cfg.wandb.tags,
            reinit=True,
        )
        logger.info("W&B run initialised: %s", self._wandb_run.url)

    def _init_csv(self) -> None:
        if not self.cfg.csv_logging.enabled:
            return
        write_header = not self._csv_path.exists()
        self._csv_file   = open(self._csv_path, "a", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "img_idx", "label", "step",
                "eta_raw", "eta_smoothed",
                "queries_this_step", "queries_cumulative",
                "asr_running", "cosine_sim_sur_tgt", "target_grad_var",
                "elpd_at_best", "elpd_at_0", "elpd_at_1",
                "method_used",
            ],
            extrasaction="ignore",
        )
        if write_header:
            self._csv_writer.writeheader()

    # ── per-image lifecycle ───────────────────────────────────────────────────

    def start_image(self, img_idx: int, label: int) -> None:
        self._img_idx   = img_idx
        self._img_label = label
        self._step_buffer.clear()

    def log_step(
        self,
        step: int,
        *,
        eta_raw: float,
        eta_smoothed: float,
        queries_this_step: int,
        queries_cumulative: int,
        asr_running: float,
        diagnostics: dict,
    ) -> None:
        row = {
            "img_idx":            self._img_idx,
            "label":              self._img_label,
            "step":               step,
            "eta_raw":            eta_raw,
            "eta_smoothed":       eta_smoothed,
            "queries_this_step":  queries_this_step,
            "queries_cumulative": queries_cumulative,
            "asr_running":        asr_running,
            **diagnostics,
        }
        self._step_buffer.append(row)

        if self._wandb_run and self.cfg.wandb.log_eta_every_step:
            wandb.log({
                "step":               step,
                "eta_raw":            eta_raw,
                "eta_smoothed":       eta_smoothed,
                "queries_cumulative": queries_cumulative,
                "asr_running":        asr_running,
                **{k: v for k, v in diagnostics.items()
                   if isinstance(v, (int, float))},
            })

        if len(self._step_buffer) >= self._flush_every:
            self._flush_csv()

    def end_image(self, success: bool, total_queries: int) -> None:
        self._flush_csv()
        if self._wandb_run:
            wandb.log({
                "image/success":       int(success),
                "image/total_queries": total_queries,
                "image/img_idx":       self._img_idx,
            })

    def log_summary(self, summary: dict) -> None:
        """Log final per-method ASR summary."""
        logger.info("Summary: %s", summary)
        if self._wandb_run:
            wandb.log({"summary/" + k: v for k, v in summary.items()})

    def finish(self) -> None:
        self._flush_csv()
        if self._csv_file:
            self._csv_file.close()
        if self._wandb_run:
            self._wandb_run.finish()

    # ── internal ──────────────────────────────────────────────────────────────

    def _flush_csv(self) -> None:
        if not (self._csv_writer and self._step_buffer):
            return
        for row in self._step_buffer:
            self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._step_buffer.clear()


# ── helper ────────────────────────────────────────────────────────────────────

def _ns_to_dict(ns) -> dict:
    from types import SimpleNamespace
    if isinstance(ns, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(ns).items()}
    return ns
