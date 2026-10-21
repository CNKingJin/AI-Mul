"""Run the full pipeline for multiple seeds and aggregate the results."""
import argparse
import copy
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import config
from train_diffusion import train
from run_experiments import Experiments
from plot_results import plot
from aggregate import aggregate


def run(base, seeds, experiments, seed_root=None):
    base_output = Path(base['output_dir'])
    seed_root = Path(seed_root) if seed_root else base_output / 'seeds'
    for seed in seeds:
        sc = copy.deepcopy(base)
        sc['seed'] = int(seed)
        sc['output_dir'] = str(seed_root / f'seed_{seed}')
        output = Path(sc['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'config.yaml').write_text(
            yaml.safe_dump(sc, sort_keys=False), encoding='utf-8')
        print(f'\n=== seed {seed} -> {output} ===', flush=True)
        for unconditional in (False, True):
            for direction in ('img2txt', 'txt2img'):
                train(sc, direction, unconditional)
        Experiments(sc).run(experiments)
        plot(sc)
    aggregate(base, seed_root=seed_root, out_root=base_output / 'aggregated')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--seed-root', default=None)
    p.add_argument('--exp', choices=[str(i) for i in range(1, 8)] + ['all'],
                   default='all')
    a = p.parse_args()
    c = config(a.config)
    chosen = a.seeds if a.seeds else c.get('seeds', [c['seed']])
    selected = list(range(1, 8)) if a.exp == 'all' else [int(a.exp)]
    run(c, chosen, selected, a.seed_root)
