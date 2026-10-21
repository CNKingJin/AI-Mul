"""Five experiments. Test target features only score completion or form full-input reference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from common import (config, device, dump, load_features, checkpoint_path,
                    feature_signature, fingerprint, runtime, seed_all)
from diffusion import Denoiser, scheduler, generate
from evaluation import make_mask, fuse, cluster_metrics


class Experiments:
    def __init__(self, c):
        self.c = c
        self.dev = device(c)
        seed_all(c['seed'])
        self.img, self.txt, self.ids = load_features(c, 'test')
        df = pd.read_csv(Path(c['splits_dir']) / 'test.csv', dtype=str)
        # Feature IDs are validated by load_features; labels are read only in evaluation.
        self.labels = df.label.to_numpy()
        self.meta = json.loads((Path(c['splits_dir']) / 'metadata.json').read_text())
        self.k = c.get('n_clusters') or self.meta['n_clusters']
        self.signature = feature_signature(c)
        self.root = Path(c['output_dir'])
        for d in ('metrics', 'masks', 'predictions', 'figures'):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.e = c['experiments']
        self.artifacts = []

    def predict(self, direction, condition, unconditional):
        key = (direction, unconditional)
        if key not in self.models:
            path = checkpoint_path(self.c, direction, unconditional)
            ck = torch.load(path, map_location='cpu', weights_only=True)
            if ck['feature_signature'] != self.signature:
                raise ValueError(f'Stale checkpoint {path}: feature cache changed; retrain')
            if ck['direction'] != direction or ck['unconditional'] != unconditional:
                raise ValueError('Checkpoint direction/mode mismatch')
            model = Denoiser(**ck['model_config']).to(self.dev)
            model.load_state_dict(ck['state_dict'])
            self.models[key] = (model, scheduler(ck['model_config']), ck['scale'])
            self.artifacts.append({'path': str(path), 'sha256': fingerprint(path)})
        model, schedule, scale = self.models[key]
        return generate(model, schedule, torch.from_numpy(condition).to(self.dev), scale,
                        self.c['sample']['steps'], self.c['sample']['seed'] + (0 if direction == 'img2txt' else 1),
                        unconditional, self.c['sample']['batch_size']).cpu().numpy()

    def complete(self, mask, unconditional=False):
        # Do not pass hidden targets into model; initialize unknown slots with zero.
        img = np.where(mask[:, :1], self.img, 0.).astype(np.float32)
        txt = np.where(mask[:, 1:], self.txt, 0.).astype(np.float32)
        for column, direction, source, target in [(0, 'txt2img', txt, img), (1, 'img2txt', img, txt)]:
            missing = mask[:, column] == 0
            if missing.any():
                target[missing] = self.predict(direction, source[missing], unconditional)
        assert np.array_equal(img[mask[:, 0] == 1], self.img[mask[:, 0] == 1])
        assert np.array_equal(txt[mask[:, 1] == 1], self.txt[mask[:, 1] == 1])
        return img, txt

    def mask(self, rate, pattern, tag):
        m = make_mask(len(self.img), rate, pattern, self.e['mask_seed'])
        pd.DataFrame({'sample_id': self.ids.sample_id, 'm_img': m[:, 0], 'm_txt': m[:, 1]}).to_csv(
            self.root / 'masks' / f'{tag}.csv', index=False)
        return m

    def score(self, x, mask, tag, **extra):
        metrics, predicted = cluster_metrics(x, self.labels, self.k, self.e['kmeans_seed'], self.e['n_init'])
        np.savez_compressed(self.root / 'predictions' / f'{tag}.npz', features=x,
                            clusters=predicted, sample_ids=self.ids.sample_id.to_numpy(dtype=str))
        return {**extra, **metrics, 'n': len(x), 'K': self.k,
                'missing_image': int((mask[:, 0] == 0).sum()),
                'missing_text': int((mask[:, 1] == 0).sum()),
                'actual_rate': float((mask.min(axis=1) == 0).mean())}

    def table(self, name, rows):
        path = self.root / 'metrics' / f'{name}.csv'
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f'\n{path}\n{df.head(12).to_string(index=False)}', flush=True)

    def exp1(self):
        rows, details = [], []
        for pattern, direction, col in [('text', 'img2txt', 1), ('image', 'txt2img', 0)]:
            m = self.mask(self.e['reconstruction_rate'], pattern, f'exp1_{direction}')
            i, t = self.complete(m)
            missing = m[:, col] == 0
            if not missing.any():
                raise ValueError('No missing samples for exp1; increase reconstruction_rate')
            real, pred = (self.img, i) if col == 0 else (self.txt, t)
            mse = ((real[missing] - pred[missing]) ** 2).mean(axis=1)
            cosine = (real[missing] * pred[missing]).sum(axis=1)
            rows.append({'direction': direction, 'n_missing': len(mse), 'MSE': mse.mean(), 'cosine': cosine.mean()})
            for sid, a, b in zip(self.ids.sample_id[missing], mse, cosine):
                details.append({'sample_id': sid, 'direction': direction, 'MSE': a, 'cosine': b})
            np.savez_compressed(self.root / 'predictions' / f'exp1_{direction}.npz',
                                real=real[missing], generated=pred[missing],
                                sample_ids=self.ids.sample_id[missing].to_numpy(dtype=str))
        self.table('exp1_reconstruction', rows)
        self.table('exp1_per_sample', details)

    def exp2(self):
        rows = []
        for rate in [0.] + list(self.e['rates']):
            tag = f'exp2_r{rate:g}'
            m = self.mask(rate, 'random', tag)
            i, t = self.complete(m)
            rows.append(self.score(fuse(i, t), m, tag, rate=rate))
        self.table('exp2_missing_rate', rows)

    def exp3(self):
        rows = []
        for pattern in ('image', 'text', 'random'):
            tag = f'exp3_{pattern}'
            m = self.mask(self.e['rate'], pattern, tag)
            i, t = self.complete(m)
            rows.append(self.score(fuse(i, t), m, tag, pattern=pattern, rate=self.e['rate']))
        self.table('exp3_missing_pattern', rows)

    def exp4(self):
        m = self.mask(self.e['rate'], 'random', 'exp4_shared')
        i, t = self.complete(m)
        ui, ut = self.complete(m, unconditional=True)
        rows = []
        for name, x in [('conditional_concat', fuse(i, t)), ('unconditional_concat', fuse(ui, ut)),
                        ('conditional_downweight', fuse(i, t, m, self.e['generated_weight']))]:
            rows.append(self.score(x, m, 'exp4_' + name, setting=name))
        self.table('exp4_ablation', rows)

    def exp5(self):
        m = self.mask(self.e['rate'], 'random', 'exp5_shared')
        i, t = self.complete(m)
        matrices = [fuse(self.img * m[:, :1], self.txt * m[:, 1:]), fuse(i, t), fuse(self.img, self.txt)]
        joined = np.concatenate(matrices)
        pca = PCA(n_components=min(50, joined.shape[1], len(joined) - 1), svd_solver='full')
        reduced = pca.fit_transform(joined)
        coordinates = TSNE(n_components=2, perplexity=min(self.e['tsne_perplexity'], len(joined) - 1),
                           init='pca', learning_rate='auto', random_state=self.c['seed']).fit_transform(reduced)
        rows = []
        for j, name in enumerate(('observed', 'completed', 'complete_reference')):
            for idx, (x, y) in enumerate(coordinates[j * len(i):(j + 1) * len(i)]):
                rows.append({'panel': name, 'sample_id': self.ids.sample_id.iloc[idx],
                             'label': self.labels[idx], 'x': x, 'y': y})
        self.table('exp5_tsne_coordinates', rows)
        # Save PCA parameters and source matrices to make provenance explicit.
        np.savez_compressed(self.root / 'predictions' / 'exp5_inputs.npz', joined=joined,
                            pca_components=pca.components_, pca_mean=pca.mean_)

    def run(self, selected):
        for number in selected:
            getattr(self, f'exp{number}')()
        dump(self.root / 'logs' / 'evaluation_run.json', {**runtime(self.c),
             'experiments': selected, 'feature_signature': self.signature, 'checkpoints': self.artifacts,
             'synthetic': self.meta.get('synthetic', False),
             'note': 'Single seed; not a statistical significance study.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--exp', choices=['1', '2', '3', '4', '5', 'all'], default='all')
    a = p.parse_args()
    Experiments(config(a.config)).run(list(range(1, 6)) if a.exp == 'all' else [int(a.exp)])
