"""Isolated CPU end-to-end check with synthetic paired features. NOT scientific results."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import dump, norm
from train_diffusion import train
from run_experiments import Experiments
from plot_results import plot
from aggregate import aggregate


def _config(root):
    c = yaml.safe_load(Path('configs/default.yaml').read_text())
    c.update(device='cpu', splits_dir=str(root / 'splits'),
             features_dir=str(root / 'features'), output_dir=str(root / 'seeds'),
             seeds=[42, 43])
    c['model'].update(dim=16, hidden=32, time_dim=16, timesteps=20)
    c['train'].update(epochs=3, batch_size=32, feature_scale=4.)
    c['sample'].update(steps=5, batch_size=64)
    c['experiments'].update(rates=[0.1, 0.3, 0.5, 0.7], rate=0.5,
                            reconstruction_rate=0.3,
                            generated_weight=0.5,
                            generated_weights=[0.0, 0.5, 1.0],
                            ddim_steps=[3, 5], tsne_perplexity=10)
    c.setdefault('baselines', {})
    c['baselines'].update(methods=['zero', 'mean', 'knn', 'ridge', 'mlp'],
                          knn_k=[1, 3, 5], ridge_alpha=[0.1, 1.0, 10.0],
                          val_rate=0.3, val_seed_offset=777)
    c['baselines']['mlp'] = {'hidden': 32, 'epochs': 20, 'patience': 5,
                             'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 32}
    return c


def _synthetic(root, dim):
    rng = np.random.default_rng(100)
    centers = rng.normal(size=(3, dim))
    for split, n in [('train', 90), ('val', 30), ('test', 30)]:
        y = np.arange(n) % 3
        latent = centers[y] + rng.normal(0, .1, (n, dim))
        img = norm(latent + rng.normal(0, .05, (n, dim)))
        txt = norm(latent + rng.normal(0, .05, (n, dim)))
        ids = pd.DataFrame({'sample_id': [f'{split}_{j}' for j in range(n)]})
        feat = root / 'features' / split
        feat.mkdir(parents=True, exist_ok=True)
        np.save(feat / 'z_img.npy', img)
        np.save(feat / 'z_txt.npy', txt)
        ids.to_csv(feat / 'ids.csv', index=False)
        ids.assign(label=y).to_csv(root / 'splits' / f'{split}.csv', index=False)
    dump(root / 'splits' / 'metadata.json', {'n_clusters': 3, 'synthetic': True})


def run(root):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'splits').mkdir(parents=True, exist_ok=True)
    c = _config(root)
    _synthetic(root, c['model']['dim'])
    for seed in c['seeds']:
        sc = copy.deepcopy(c)
        sc['seed'] = seed
        sc['output_dir'] = str(root / 'seeds' / f'seed_{seed}')
        Path(sc['output_dir']).mkdir(parents=True, exist_ok=True)
        (Path(sc['output_dir']) / 'config.yaml').write_text(
            yaml.safe_dump(sc, sort_keys=False), encoding='utf-8')
        print(f'\n=== smoke seed {seed} ===', flush=True)
        for unconditional in (False, True):
            for direction in ('img2txt', 'txt2img'):
                train(sc, direction, unconditional)
        Experiments(sc).run(list(range(1, 8)))
        plot(sc)
    aggregate(c, seed_root=root / 'seeds', out_root=root / 'aggregated')
    required = ['exp1_reconstruction', 'exp2_missing_rate', 'exp3_missing_pattern',
                'exp4_ablation', 'exp5_tsne_coordinates', 'exp6_ddim_steps',
                'exp7_retrieval', 'metrics_long']
    for seed in c['seeds']:
        base = root / 'seeds' / f'seed_{seed}'
        for name in required:
            assert (base / 'metrics' / f'{name}.csv').stat().st_size > 0, name
        assert (base / 'figures' / 'exp5_tsne.png').stat().st_size > 0
        assert (base / 'figures' / 'exp7_retrieval.png').stat().st_size > 0
    for name in ('summary', 'paired_delta'):
        assert (root / 'aggregated' / f'{name}.csv').stat().st_size > 0, name
    assert (root / 'aggregated' / 'figures' / 'exp2_missing_rate.png').stat().st_size > 0
    assert (root / 'aggregated' / 'figures' / 'exp7_retrieval.png').stat().st_size > 0
    print('PASS: synthetic pipeline only; do not use these values as real experiment results.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='outputs/smoke')
    run(Path(p.parse_args().output))
