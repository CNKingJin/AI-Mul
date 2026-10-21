"""Regenerate figures and overview exclusively from saved CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import config

METHOD_ORDER = ['zero', 'mean', 'knn', 'ridge', 'mlp', 'diffusion']
COLORS = dict(zip(METHOD_ORDER, plt.get_cmap('tab10').colors))
METRICS = ['ACC', 'NMI', 'ARI']


def _read(path):
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _methods(df):
    return [m for m in METHOD_ORDER if m in set(df.method)]


def _plot_exp2(df, path):
    if df is None or not {'rate', 'method'} <= set(df.columns):
        return
    methods = _methods(df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            continue
        for method in methods:
            sub = df[df.method == method].sort_values('rate')
            ax.plot(sub.rate, sub[metric], marker='o', ms=3, label=method,
                    color=COLORS.get(method))
        ax.set(xlabel='Fraction of samples missing one modality',
               ylabel=metric, title=metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_exp3(df, path):
    if df is None or not {'pattern', 'method'} <= set(df.columns):
        return
    methods = _methods(df)
    patterns = [p for p in ('image', 'text', 'random') if p in set(df.pattern)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index='pattern', columns='method',
                               values=metric, aggfunc='mean').reindex(patterns)
        pivot = pivot[[m for m in methods if m in pivot.columns]]
        pivot.plot.bar(ax=ax, color=[COLORS.get(m) for m in pivot.columns], rot=0)
        ax.set(xlabel='Missing pattern', ylabel=metric, title=metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_exp4(df, path):
    if df is None or 'setting' not in df.columns:
        return
    df = df.sort_values('setting')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    positions = np.arange(len(df))
    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            continue
        ax.bar(positions, df[metric].to_numpy(), color=COLORS['diffusion'])
        ax.set_xticks(positions, df.setting, rotation=25, ha='right', fontsize=7)
        ax.set(ylabel=metric, title=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_exp6(df, path):
    if df is None or 'steps' not in df.columns:
        return
    df = df.sort_values('steps')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            continue
        ax.plot(df.steps, df[metric], marker='o', color=COLORS['diffusion'])
        ax.set(xlabel='DDIM steps', ylabel=metric, title=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_exp7(df, path):
    if df is None or 'method' not in df.columns:
        return
    methods = _methods(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, direction, cols in [
            (axes[0], 'image to text', ('R1_i2t', 'R5_i2t')),
            (axes[1], 'text to image', ('R1_t2i', 'R5_t2i'))]:
        width = .8 / max(len(cols), 1)
        positions = np.arange(len(methods))
        for j, column in enumerate(cols):
            if column not in df.columns:
                continue
            values = [df[df.method == m][column].mean() if (df.method == m).any() else np.nan
                      for m in methods]
            ax.bar(positions + j * width, values, width, label=column)
        ax.set_xticks(positions + width * (len(cols) - 1) / 2, methods,
                      rotation=25, ha='right', fontsize=7)
        ax.set(ylabel='Recall', title=direction, ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_exp5(df, path):
    if df is None or not {'panel', 'label', 'x', 'y'} <= set(df.columns):
        return
    panels = list(dict.fromkeys(df.panel))
    labels = sorted(df.label.astype(str).unique())
    cmap = plt.get_cmap('tab20', len(labels))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, panel in zip(axes, panels):
        subset = df[df.panel == panel]
        for j, label in enumerate(labels):
            part = subset[subset.label.astype(str) == label]
            ax.scatter(part.x, part.y, s=12, color=cmap(j), alpha=.8, label=label)
        ax.set_title(panel)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), fontsize=6)
    fig.suptitle('Joint t-SNE: qualitative view only')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot(c):
    root = Path(c['output_dir'])
    metrics, figures = root / 'metrics', root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    overview = []
    for filename, plotter in [
            ('exp2_missing_rate', _plot_exp2),
            ('exp3_missing_pattern', _plot_exp3),
            ('exp4_ablation', _plot_exp4),
            ('exp6_ddim_steps', _plot_exp6),
            ('exp7_retrieval', _plot_exp7)]:
        df = _read(metrics / f'{filename}.csv')
        if df is not None:
            plotter(df, figures / f'{filename}.png')
            overview.append(df.assign(experiment=filename))
    if overview:
        pd.concat(overview, ignore_index=True).to_csv(metrics / 'overview.csv', index=False)
    _plot_exp5(_read(metrics / 'exp5_tsne_coordinates.csv'), figures / 'exp5_tsne.png')
    for path in sorted((root / 'logs').glob('*_loss.csv')):
        df = pd.read_csv(path)
        fig, ax = plt.subplots()
        ax.plot(df.epoch, df.train_loss, label='train')
        ax.plot(df.epoch, df.val_loss, label='validation (fixed noise)')
        ax.legend()
        ax.set(xlabel='Epoch', ylabel='Noise MSE')
        fig.tight_layout()
        fig.savefig(figures / f'{path.stem}.png', dpi=150)
        plt.close(fig)
    print(f'Figures saved to {figures}')


def _errorbar_by_method(ax, df, metric, x_col):
    methods = _methods(df)
    for method in methods:
        sub = df[(df.method == method) & (df.metric == metric)].sort_values(x_col)
        if sub.empty:
            continue
        ax.errorbar(sub[x_col], sub['mean'], yerr=sub['std'].fillna(0),
                    marker='o', ms=3, capsize=2, label=method,
                    color=COLORS.get(method))
    ax.set(ylabel=metric, title=metric)


def _grouped_bars(ax, df, metric, x_col, order):
    methods = _methods(df)
    width = .8 / max(len(methods), 1)
    positions = np.arange(len(order))
    for j, method in enumerate(methods):
        sub = df[(df.method == method) & (df.metric == metric)]
        lookup = sub.set_index(x_col)['mean'] if not sub.empty else pd.Series(dtype=float)
        std = sub.set_index(x_col)['std'] if not sub.empty else pd.Series(dtype=float)
        values = [lookup.get(x, np.nan) for x in order]
        errors = [0 if not np.isfinite(std.get(x, np.nan)) else std.get(x) for x in order]
        ax.bar(positions + j * width, values, width, yerr=errors, capsize=2,
               label=method, color=COLORS.get(method))
    ax.set_xticks(positions + width * (len(methods) - 1) / 2, order,
                  rotation=20, ha='right', fontsize=7)


def plot_summary(summary, figures):
    figures = Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    if summary is None or summary.empty:
        return
    summary = summary.copy()
    for column in ('mean', 'std', 'ci_low', 'ci_high'):
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors='coerce')

    sub = summary[summary.experiment == 'exp2_missing_rate']
    if not sub.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric in zip(axes, METRICS):
            _errorbar_by_method(ax, sub, metric, 'rate')
            ax.set_xlabel('Fraction of samples missing one modality')
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / 'exp2_missing_rate.png', dpi=180)
        plt.close(fig)

    sub = summary[summary.experiment == 'exp3_missing_pattern']
    if not sub.empty:
        patterns = [p for p in ('image', 'text', 'random')
                    if p in set(sub.pattern.dropna())]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric in zip(axes, METRICS):
            _grouped_bars(ax, sub, metric, 'pattern', patterns)
            ax.set_ylabel(metric)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / 'exp3_missing_pattern.png', dpi=180)
        plt.close(fig)

    sub = summary[summary.experiment == 'exp4_ablation']
    if not sub.empty:
        settings = list(dict.fromkeys(sub.setting.dropna()))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric in zip(axes, METRICS):
            _grouped_bars(ax, sub, metric, 'setting', settings)
            ax.set_ylabel(metric)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / 'exp4_ablation.png', dpi=180)
        plt.close(fig)

    sub = summary[summary.experiment == 'exp6_ddim_steps']
    if not sub.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric in zip(axes, METRICS):
            _errorbar_by_method(ax, sub, metric, 'steps')
            ax.set_xlabel('DDIM steps')
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / 'exp6_ddim_steps.png', dpi=180)
        plt.close(fig)

    sub = summary[summary.experiment == 'exp7_retrieval']
    if not sub.empty:
        methods = _methods(sub)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, metrics, title in [(axes[0], ('R1_i2t', 'R5_i2t'), 'image to text'),
                                   (axes[1], ('R1_t2i', 'R5_t2i'), 'text to image')]:
            width = .8 / max(len(metrics), 1)
            positions = np.arange(len(methods))
            for j, metric in enumerate(metrics):
                lookup = sub[sub.metric == metric].set_index('method')['mean']
                std = sub[sub.metric == metric].set_index('method')['std']
                values = [lookup.get(m, np.nan) for m in methods]
                errors = [0 if not np.isfinite(std.get(m, np.nan)) else std.get(m)
                          for m in methods]
                ax.bar(positions + j * width, values, width, yerr=errors,
                       capsize=2, label=metric)
            ax.set_xticks(positions + width * (len(metrics) - 1) / 2, methods,
                          rotation=25, ha='right', fontsize=7)
            ax.set_ylabel('Recall')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / 'exp7_retrieval.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    plot(config(p.parse_args().config))
