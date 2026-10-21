"""Regenerate figures and overview exclusively from saved CSVs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from common import config


def plot(c):
    root = Path(c['output_dir'])
    metrics, figures = root / 'metrics', root / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    overview = []
    for filename in ['exp2_missing_rate', 'exp3_missing_pattern', 'exp4_ablation']:
        path = metrics / f'{filename}.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        overview.append(df.assign(experiment=filename))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if filename == 'exp2_missing_rate':
            for col in ('ACC', 'NMI', 'ARI'):
                ax.plot(df.rate, df[col], 'o-', label=col)
            ax.set_xlabel('Fraction of samples missing one modality')
        else:
            df.set_index('pattern' if 'pattern' in df else 'setting')[['ACC', 'NMI', 'ARI']].plot.bar(ax=ax, rot=12)
        ax.set_ylabel('Score (ARI may be negative)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f'{filename}.png', dpi=180)
        plt.close(fig)
    if overview:
        pd.concat(overview, ignore_index=True).to_csv(metrics / 'overview.csv', index=False)
    path = metrics / 'exp5_tsne_coordinates.csv'
    if path.exists():
        df = pd.read_csv(path, dtype={'label': str})
        labels = sorted(df.label.unique())
        cmap = plt.get_cmap('tab20', len(labels))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
        for ax, panel in zip(axes, ('observed', 'completed', 'complete_reference')):
            subset = df[df.panel == panel]
            for j, label in enumerate(labels):
                part = subset[subset.label == label]
                ax.scatter(part.x, part.y, s=12, color=cmap(j), label=label, alpha=0.8)
            ax.set_title(panel)
        axes[-1].legend(bbox_to_anchor=(1.02, 1), fontsize=6)
        fig.suptitle('Joint t-SNE: qualitative view only')
        fig.tight_layout()
        fig.savefig(figures / 'exp5_tsne.png', dpi=180)
        plt.close(fig)
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    plot(config(p.parse_args().config))
