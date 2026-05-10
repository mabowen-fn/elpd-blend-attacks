"""
run_experiment.py — Full evaluation script. Runs all attacks on the configured
dataset and writes results to results/logs/. Also produces Query-vs-ASR and
η trajectory plots.

Usage:
    .venv/bin/python scripts/run_experiment.py
    .venv/bin/python scripts/run_experiment.py --n_images 20   # quick test
    .venv/bin/python scripts/run_experiment.py --steps 20      # fewer steps

The script runs ELPD-Blend + all baselines on each image and logs everything
to results/logs/final_results.csv and results/logs/run_metrics.csv.
"""

import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless — no display needed
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('run_experiment')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config, dump_config
from src.utils.logger        import RunLogger
from src.models.model_loader import TargetModel, SurrogateModel
from src.data.data_loader    import get_dataloader
from src.attacks.elpd_blender    import ELPDBlender
from src.attacks.query_estimator import QueryEstimator
from src.attacks.elpd_attack     import elpd_blend_attack
from src.attacks.baselines       import (
    run_mifgsm, run_difgsm, run_nes_only,
    run_static_hybrid, run_square_attack,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',   default='configs/config.yaml')
    p.add_argument('--n_images', type=int, default=None, help='Override num_samples')
    p.add_argument('--steps',    type=int, default=None, help='Override num_steps')
    p.add_argument('--no_wandb', action='store_true',    help='Disable W&B')
    return p.parse_args()


def build_asr_curve(df: pd.DataFrame, query_checkpoints: list[int]) -> dict[str, list[float]]:
    """For each method, compute ASR at each query checkpoint."""
    curves = {}
    for method in df['method'].unique():
        dm = df[df['method'] == method]
        n_images = len(dm['img_idx'].unique())
        asr_vals = []
        for budget in query_checkpoints:
            n_success = len(dm[dm['success'] & (dm['queries'] <= budget)]['img_idx'].unique())
            asr_vals.append(n_success / max(n_images, 1))
        curves[method] = asr_vals
    return curves


def plot_asr_curves(curves: dict, checkpoints: list, cfg, output_path: str):
    styles = {
        'elpd_blend':   ('steelblue',  '-',  2.5),
        'mifgsm':       ('green',      '--', 1.5),
        'difgsm':       ('limegreen',  '--', 1.5),
        'nes_only':     ('orange',     '-.',  1.5),
        'static_hybrid':('purple',     ':',  1.5),
        'square':       ('tomato',     '-.',  1.5),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, asr_vals in curves.items():
        colour, ls, lw = styles.get(method, ('gray', '-', 1))
        ax.plot(checkpoints, asr_vals, color=colour, linestyle=ls, linewidth=lw,
                marker='o', markersize=4, label=method)
    ax.axhline(0.9, color='black', linestyle=':', linewidth=1, alpha=0.5, label='90% target')
    ax.set_xlabel('Query Budget')
    ax.set_ylabel('Attack Success Rate')
    ax.set_title(
        f'Query Efficiency: ELPD-Blend vs Baselines\n'
        f'Target={cfg.models.target.arch}  Surrogate={cfg.models.surrogate.arch}'
        f'  ε={cfg.attack.epsilon}  dataset={cfg.dataset.name}'
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info('ASR curve saved → %s', output_path)


def plot_eta_trajectory(step_csv: str, output_path: str):
    if not os.path.exists(step_csv):
        logger.warning('Step CSV not found, skipping η trajectory plot.')
        return
    df = pd.read_csv(step_csv)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    gb = df.groupby('step')['eta_smoothed'].agg(['mean', 'std'])
    ax.plot(gb.index, gb['mean'], color='steelblue', linewidth=2)
    ax.fill_between(gb.index,
                    (gb['mean'] - gb['std']).clip(0, 1),
                    (gb['mean'] + gb['std']).clip(0, 1),
                    alpha=0.25, color='steelblue')
    ax.set_xlabel('Attack Step')
    ax.set_ylabel('η (smoothed)')
    ax.set_title('η Trajectory (mean ± std across images)')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if 'cosine_sim_sur_tgt' in df.columns:
        gc = df.groupby('step')['cosine_sim_sur_tgt'].agg(['mean', 'std'])
        ax.plot(gc.index, gc['mean'], color='tomato', linewidth=2)
        ax.fill_between(gc.index,
                        gc['mean'] - gc['std'],
                        gc['mean'] + gc['std'],
                        alpha=0.25, color='tomato')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Attack Step')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Surrogate–Target Alignment over Steps')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info('η trajectory saved → %s', output_path)


def main():
    args  = parse_args()
    cfg   = load_config(args.config)

    if args.n_images:
        cfg.dataset.num_samples = args.n_images
    if args.steps:
        cfg.attack.num_steps = args.steps
    if args.no_wandb:
        cfg.wandb.enabled = False

    DEVICE = torch.device(
        cfg.experiment.device
        if (cfg.experiment.device == 'cpu' or
            (cfg.experiment.device == 'mps' and torch.backends.mps.is_available()) or
            (cfg.experiment.device == 'cuda' and torch.cuda.is_available()))
        else 'cpu'
    )
    logger.info('Device: %s', DEVICE)
    torch.manual_seed(cfg.experiment.seed)

    os.makedirs('results/logs',  exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    dump_config(cfg, f'results/logs/config_snapshot_{cfg.experiment.name}.yaml')

    # ── Load models ───────────────────────────────────────────────────────
    logger.info('Loading surrogate: %s', cfg.models.surrogate.arch)
    surrogate = SurrogateModel(cfg.models.surrogate, cfg.dataset, device=DEVICE)

    logger.info('Loading target: %s', cfg.models.target.arch)
    _target_shell = TargetModel(
        cfg.models.target, cfg.dataset,
        query_budget=cfg.attack.query_budget,
        device=DEVICE, label=0,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    loader = get_dataloader(cfg)

    run_logger = RunLogger(cfg)
    records:  list[dict] = []
    n_eval    = 0

    # One shared target instance for label discovery (no budget consumed)
    _label_probe = TargetModel(
        cfg.models.target, cfg.dataset,
        query_budget=10**9,
        device=DEVICE, label=0,
    )

    for img_idx, (x_batch, y_batch) in enumerate(loader):
        x_orig = x_batch[0].to(DEVICE)

        # Use the target model's own prediction as the true label.
        # CIFAR-10 labels (0-9) are incompatible with ImageNet-pretrained models
        # (0-999), so we always attack the model's current prediction instead.
        label = _label_probe.predict(x_orig)

        n_eval += 1
        run_logger.start_image(img_idx, label)
        logger.info('[%04d/%04d] label=%d', img_idx, cfg.dataset.num_samples, label)

        def _make_est(seed_offset=0):
            t = TargetModel(
                cfg.models.target, cfg.dataset,
                query_budget=cfg.attack.query_budget,
                device=DEVICE, label=label,
            )
            return t, QueryEstimator(
                cfg.query_estimator,
                query_fn=t.get_query_fn(),
                seed=cfg.experiment.seed + img_idx + seed_offset,
            )

        # ── ELPD-Blend ────────────────────────────────────────────────────
        t_elpd, est_elpd = _make_est(0)
        res_elpd = elpd_blend_attack(
            x_orig, label, surrogate, t_elpd,
            ELPDBlender(cfg.elpd_blender, sigma=cfg.query_estimator.sigma),
            est_elpd, cfg, run_logger=run_logger,
        )
        records.append({'img_idx': img_idx, 'label': label, 'method': 'elpd_blend',
                        'success': res_elpd.success, 'queries': res_elpd.queries_used,
                        'steps': res_elpd.steps_taken})
        run_logger.end_image(res_elpd.success, res_elpd.queries_used)

        # ── MI-FGSM ───────────────────────────────────────────────────────
        if cfg.baselines.run_mifgsm:
            t_mi, _ = _make_est(1000)
            res_mi  = run_mifgsm(x_orig, label, surrogate, t_mi, cfg)
            records.append({'img_idx': img_idx, 'label': label, 'method': 'mifgsm',
                            'success': res_mi.success, 'queries': 0, 'steps': res_mi.steps_taken})

        # ── DI-FGSM ───────────────────────────────────────────────────────
        if cfg.baselines.run_difgsm:
            t_di, _ = _make_est(2000)
            res_di  = run_difgsm(x_orig, label, surrogate, t_di, cfg)
            records.append({'img_idx': img_idx, 'label': label, 'method': 'difgsm',
                            'success': res_di.success, 'queries': 0, 'steps': res_di.steps_taken})

        # ── NES-only ──────────────────────────────────────────────────────
        if cfg.baselines.run_nes:
            t_nes, est_nes = _make_est(3000)
            res_nes = run_nes_only(x_orig, label, t_nes, est_nes, cfg)
            records.append({'img_idx': img_idx, 'label': label, 'method': 'nes_only',
                            'success': res_nes.success, 'queries': res_nes.queries_used,
                            'steps': res_nes.steps_taken})

        # ── Static Hybrid ─────────────────────────────────────────────────
        if cfg.baselines.run_static_hybrid.enabled:
            t_sh, est_sh = _make_est(4000)
            res_sh = run_static_hybrid(x_orig, label, surrogate, t_sh, est_sh, cfg)
            records.append({'img_idx': img_idx, 'label': label, 'method': 'static_hybrid',
                            'success': res_sh.success, 'queries': res_sh.queries_used,
                            'steps': res_sh.steps_taken})

        # ── Square Attack ─────────────────────────────────────────────────
        if cfg.baselines.run_square:
            t_sq, _ = _make_est(5000)
            res_sq  = run_square_attack(x_orig, label, t_sq, cfg)
            records.append({'img_idx': img_idx, 'label': label, 'method': 'square',
                            'success': res_sq.success, 'queries': res_sq.queries_used,
                            'steps': res_sq.steps_taken})

        parts = [f"ELPD({'✓' if res_elpd.success else '✗'},{res_elpd.queries_used}q)"]
        if cfg.baselines.run_mifgsm:
            parts.append(f"MI({'✓' if res_mi.success else '✗'})")
        if cfg.baselines.run_nes:
            parts.append(f"NES({'✓' if res_nes.success else '✗'},{res_nes.queries_used}q)")
        logger.info('  %s', '  '.join(parts))

    if not records:
        run_logger.finish()
        logger.error('No images evaluated. Check dataset path and config.')
        return

    # ── Results table ─────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv('results/logs/final_results.csv', index=False)

    summary = df.groupby('method').agg(
        ASR=('success', 'mean'),
        avg_queries=('queries', 'mean'),
        n_images=('img_idx', 'nunique'),
    ).round(4)
    summary['ASR_pct'] = (summary['ASR'] * 100).round(1)
    print('\n' + '='*60)
    print(summary[['ASR_pct', 'avg_queries', 'n_images']].sort_values('ASR_pct', ascending=False).to_string())
    print('='*60)

    run_logger.log_summary(summary['ASR_pct'].to_dict())

    # ── Plots ──────────────────────────────────────────────────────────────
    checkpoints = [40, 100, 200, 400, 800, 1000, 2000, cfg.attack.query_budget]
    checkpoints = sorted(set(c for c in checkpoints if c <= cfg.attack.query_budget))
    curves = build_asr_curve(df, checkpoints)
    plot_asr_curves(curves, checkpoints, cfg, 'results/plots/query_vs_asr.png')
    plot_eta_trajectory('results/logs/run_metrics.csv', 'results/plots/eta_trajectory.png')

    run_logger.finish()
    logger.info('Done. Results in results/logs/ and results/plots/')


if __name__ == '__main__':
    main()
