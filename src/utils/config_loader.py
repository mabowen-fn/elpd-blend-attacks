"""
config_loader.py — loads config.yaml and exposes a typed, dot-accessible config object.

Design choices:
  - Uses PyYAML + a thin SimpleNamespace wrapper so every sub-key is reachable
    via cfg.elpd_blender.waic.llik_temperature rather than cfg["elpd_blender"]["waic"][...].
  - Performs basic validation at load-time (types, value ranges) so invalid configs
    surface immediately, not mid-experiment.
  - Resolves paths relative to the project root (the directory containing this file's
    parent), making the config portable across local machines and cloud pods.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── internal helpers ──────────────────────────────────────────────────────────

def _dict_to_ns(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to a SimpleNamespace tree."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_ns(value))
        else:
            setattr(ns, key, value)
    return ns


def _ns_to_dict(ns: SimpleNamespace) -> dict:
    """Inverse of _dict_to_ns — useful for serialising cfg back to YAML."""
    out: dict[str, Any] = {}
    for key, value in vars(ns).items():
        out[key] = _ns_to_dict(value) if isinstance(value, SimpleNamespace) else value
    return out


def _resolve_paths(cfg: SimpleNamespace, project_root: Path) -> None:
    """
    Rewrite relative paths inside cfg to absolute paths anchored at project_root.
    Only touches keys whose names end in '_dir', '_path', or 'root'.
    """
    path_suffixes = ("_dir", "_path", "root")
    for key, value in vars(cfg).items():
        if isinstance(value, SimpleNamespace):
            _resolve_paths(value, project_root)
        elif isinstance(value, str) and any(key.endswith(s) for s in path_suffixes):
            if value and not os.path.isabs(value):
                setattr(cfg, key, str(project_root / value))


def _validate(cfg: SimpleNamespace) -> None:
    """Lightweight validation of critical hyper-parameter ranges."""
    a = cfg.attack
    assert 0 < a.epsilon <= 1.0,      f"epsilon must be in (0, 1], got {a.epsilon}"
    assert 0 < a.step_size <= a.epsilon, \
        f"step_size ({a.step_size}) should be ≤ epsilon ({a.epsilon})"
    assert a.num_steps > 0,           f"num_steps must be > 0"
    assert a.query_budget > 0,        f"query_budget must be > 0"

    b = cfg.elpd_blender
    assert b.method in ("waic", "loo_psis", "cosine_var"), \
        f"Unknown elpd_blender.method: {b.method}"
    assert 0 < b.eta_grid.n_points <= 1001, \
        f"eta_grid.n_points should be in [2, 1001], got {b.eta_grid.n_points}"
    assert 0.0 <= b.eta_min < b.eta_max <= 1.0, \
        f"eta_min / eta_max out of range"
    assert 0.0 < b.eta_ema_alpha <= 1.0, \
        f"eta_ema_alpha must be in (0, 1], got {b.eta_ema_alpha}"

    q = cfg.query_estimator
    assert q.method in ("nes", "spsa", "fd_central"), \
        f"Unknown query_estimator.method: {q.method}"
    assert q.n_samples >= 2,          f"n_samples must be ≥ 2 (antithetic pairs)"
    assert q.sigma > 0,               f"sigma must be > 0"

    logger.debug("Config validation passed.")


# ── public API ────────────────────────────────────────────────────────────────

def load_config(
    config_path: str | Path | None = None,
    resolve_paths: bool = True,
) -> SimpleNamespace:
    """
    Load config.yaml and return a dot-accessible SimpleNamespace.

    Args:
        config_path:    Path to the YAML file.  If None, defaults to
                        <project_root>/configs/config.yaml.
        resolve_paths:  If True, relative filesystem paths are resolved to
                        absolute paths anchored at the project root.

    Returns:
        cfg — a SimpleNamespace tree mirroring the YAML hierarchy.
    """
    # Determine project root as two levels above this file (src/utils/ → project/)
    this_file   = Path(__file__).resolve()
    project_root = this_file.parents[2]   # adversarial_ml/

    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        raw: dict = yaml.safe_load(fh)

    cfg = _dict_to_ns(raw)

    if resolve_paths:
        _resolve_paths(cfg, project_root)

    _validate(cfg)

    logger.info("Loaded config from %s", config_path)
    return cfg


def dump_config(cfg: SimpleNamespace, output_path: str | Path) -> None:
    """Serialise a cfg object back to YAML — useful for run-reproducibility snapshots."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.safe_dump(_ns_to_dict(cfg), fh, default_flow_style=False, sort_keys=False)
    logger.info("Config snapshot saved to %s", output_path)
