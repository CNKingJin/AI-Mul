"""Isolated CPU end-to-end check with synthetic paired features. NOT scientific results."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import config, dump, norm
from train_diffusion import train
from run_experiments import Experiments
from plot_results import plot


def run(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    c = yaml.safe_load(Path('configs/default.yaml').read_text())
    c.update(device='cpu', splits_dir=str(root / 'splits'), features_dir=str(root / 'features'),
             output_dir=str(root / 'outputs'))
    c['model'].update(dim=16, hidden=32, time_dim=16, timesteps=20)
    c['train'].update(epochs=3, batch_size=32, feature_scale=4.)
    c['sample'].update(steps=5)
    path = root / 'smoke.yaml'
    path.write_text(yaml.safe_dump(c))
    c = config(path)
    rng = np.random.default_rng(100)
    centers = rng.normal(size=(3, 16))
    for split, n in [('train', 90), ('val', 30), ('test', 30)]:
        y = np.arange(n) % 3
        latent = centers[y] + rng.normal(0, .1, (n, 16))
        i, t = norm(latent + rng.normal(0, .05, (n, 16))), norm(latent + rng.normal(0, .05, (n, 16)))
        ids = pd.DataFrame({'sample_id': [f'{split}_{j}' for j in range(n)]})
        feat = root / 'features' / split
        feat.mkdir(parents=True, exist_ok=True)
        np.save(feat / 'z_img.npy', i)
        np.save(feat / 'z_txt.npy', t)
        ids.to_csv(feat / 'ids.csv', index=False)
        ids.assign(label=y).to_csv(root / 'splits' / f'{split}.csv', index=False)
    dump(root / 'splits' / 'metadata.json', {'n_clusters': 3, 'synthetic': True})
    for unconditional in (False, True):
        for direction in ('img2txt', 'txt2img'):
            train(c, direction, unconditional)
    Experiments(c).run([1, 2, 3, 4, 5])
    plot(c)
    required = ['exp1_reconstruction', 'exp2_missing_rate', 'exp3_missing_pattern', 'exp4_ablation', 'exp5_tsne_coordinates', 'overview']
    for name in required:
        assert (root / 'outputs' / 'metrics' / f'{name}.csv').stat().st_size > 0
    assert (root / 'outputs' / 'figures' / 'exp5_tsne.png').stat().st_size > 0
    print('PASS: synthetic pipeline only; do not use these values as real experiment results.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='outputs/smoke')
    run(Path(p.parse_args().output))
