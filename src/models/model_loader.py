"""
model_loader.py — Target (black-box) and Surrogate (white-box) model wrappers.

AutoDL note: pretrained weights are downloaded from torchvision on first run
and cached at ~/.cache/torch/hub/checkpoints/.  On subsequent runs (or after
manual upload) they load instantly from cache.

TargetModel wraps a torchvision model and:
  - counts every forward pass as a query
  - raises QueryBudgetExceeded when the budget is exhausted
  - exposes query_fn(x_batch) → loss tensor for QueryEstimator

SurrogateModel wraps a torchvision model and:
  - exposes gradient(x, y) → (D,) flat gradient tensor
  - the gradient flows through the surrogate (white-box), not the target
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch import Tensor

logger = logging.getLogger(__name__)


class QueryBudgetExceeded(Exception):
    pass


# ── Model factory ─────────────────────────────────────────────────────────────

def _load_torchvision_model(arch: str, pretrained: bool) -> nn.Module:
    """Load a torchvision model by architecture name string."""
    if not hasattr(tvm, arch):
        raise ValueError(f"torchvision has no model named '{arch}'")
    weights_enum = None
    if pretrained:
        # torchvision ≥ 0.13 uses weights= API; fall back to pretrained= for older
        weights_attr = arch.upper().replace("_", "") + "_Weights"
        # Try the new API first
        try:
            weights_cls = getattr(tvm, arch.replace("_bn", "").upper() + "_Weights", None)
            # Use DEFAULT weights if the class exists
            if weights_cls is not None:
                model = getattr(tvm, arch)(weights=weights_cls.DEFAULT)
            else:
                # Fallback: let torchvision pick
                model = getattr(tvm, arch)(pretrained=True)
        except TypeError:
            model = getattr(tvm, arch)(pretrained=True)
    else:
        model = getattr(tvm, arch)()
    return model


# ── Target model (black box) ──────────────────────────────────────────────────

class TargetModel:
    """
    Black-box target model.  Provides only a query interface — no gradients.

    The caller uses query_fn (returned by .get_query_fn()) as the TargetQueryFn
    passed to QueryEstimator.  Every image in a batch counts as one query.
    """

    def __init__(
        self,
        model_cfg: SimpleNamespace,
        dataset_cfg: SimpleNamespace,
        query_budget: int,
        device: torch.device,
        label: int | None = None,
    ) -> None:
        self.budget    = query_budget
        self._count    = 0
        self.device    = device
        self.label     = label      # true class label for loss computation

        model = _load_torchvision_model(model_cfg.arch, model_cfg.pretrained)
        model.eval().to(device)
        # Freeze all params — we never backprop through the target
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

        # Normalisation stats from dataset config
        mean = torch.tensor(dataset_cfg.mean, device=device).view(1, 3, 1, 1)
        std  = torch.tensor(dataset_cfg.std,  device=device).view(1, 3, 1, 1)
        self.register_norm(mean, std)

    def register_norm(self, mean: Tensor, std: Tensor) -> None:
        self._mean = mean
        self._std  = std

    def set_label(self, label: int) -> None:
        self.label = label

    def reset_count(self) -> None:
        self._count = 0

    @property
    def queries_used(self) -> int:
        return self._count

    def get_query_fn(self) -> Callable[[Tensor], Tensor]:
        """
        Returns a closure that:
          1. normalises the input batch
          2. runs a forward pass through the target
          3. returns per-image cross-entropy loss (no grad)
          4. increments and checks the query counter
        """
        def query_fn(x_batch: Tensor) -> Tensor:
            B = x_batch.shape[0]
            if self._count + B > self.budget:
                raise QueryBudgetExceeded(
                    f"Query budget {self.budget} exceeded "
                    f"(used {self._count}, requested {B})"
                )
            self._count += B

            with torch.no_grad():
                x_norm  = (x_batch - self._mean) / self._std
                logits  = self._model(x_norm)                    # (B, n_classes)
                labels  = torch.full(
                    (B,), self.label, dtype=torch.long, device=x_batch.device
                )
                # Untargeted: maximise cross-entropy → loss is already the right
                # direction for gradient ascent on the perturbation.
                losses = F.cross_entropy(logits, labels, reduction="none")  # (B,)
            return losses

        return query_fn

    def predict(self, x: Tensor) -> int:
        """Single-image prediction — counts as one query."""
        if self._count + 1 > self.budget:
            raise QueryBudgetExceeded(
                f"Query budget {self.budget} exceeded at predict() call"
            )
        self._count += 1
        with torch.no_grad():
            x_norm = (x.unsqueeze(0) - self._mean) / self._std
            logits = self._model(x_norm)
        return int(logits.argmax(dim=1).item())


# ── Surrogate model (white box) ───────────────────────────────────────────────

class SurrogateModel:
    """
    White-box surrogate model.  Gradients are free and unlimited.

    .gradient(x, label) computes ∂L/∂x through the surrogate and returns
    a flat (D,) tensor where D = C·H·W.
    """

    def __init__(
        self,
        model_cfg:  SimpleNamespace,
        dataset_cfg: SimpleNamespace,
        device: torch.device,
    ) -> None:
        self.device = device

        model = _load_torchvision_model(model_cfg.arch, model_cfg.pretrained)
        model.eval().to(device)
        # Surrogate params are frozen — we only need gradient w.r.t. input x
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

        mean = torch.tensor(dataset_cfg.mean, device=device).view(1, 3, 1, 1)
        std  = torch.tensor(dataset_cfg.std,  device=device).view(1, 3, 1, 1)
        self._mean = mean
        self._std  = std

    def gradient(self, x: Tensor, label: int) -> Tensor:
        """
        Compute ∂CrossEntropy/∂x through the surrogate.

        Args:
            x:     (C, H, W) adversarial example, values in [0,1], no batch dim
            label: true class index

        Returns:
            g: (D,) flat gradient tensor (D = C*H*W)
        """
        x_in = x.unsqueeze(0).requires_grad_(True)          # (1, C, H, W)
        x_norm = (x_in - self._mean) / self._std
        logits = self._model(x_norm)                         # (1, n_classes)
        lbl    = torch.tensor([label], device=self.device)
        loss   = F.cross_entropy(logits, lbl)
        loss.backward()
        g = x_in.grad.detach().view(-1)                      # (D,)
        return g
