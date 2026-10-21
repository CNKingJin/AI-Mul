"""Seven experiments. Test targets only score completion or form references."""
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
from evaluation import (make_mask, fuse, cluster_metrics, retrieval_metrics,
                        METRIC_COLUMNS)
from baselines import CompletionBaselines, METHODS


class Experiments:
    def __init__(self, c):
        self.c = c
        self.seed = int(c['seed'])
        self.dev = device(c)
        seed_all(self.seed)
        self.img, self.txt, self.ids = load_features(c, 'test')
        train_img, train_txt, _ = load_features(c, 'train')
        val_img, val_txt, _ = load_features(c, 'val')
        df = pd.read_csv(Path(c['splits_dir']) / 'test.csv', dtype=str)
        # Feature IDs are validated by load_features; labels are read only in evaluation.
        self.labels = df.label.to_numpy()
        self.meta = json.loads((Path(c['splits_dir']) / 'metadata.json').read_text())
        self.k = c.get('n_clusters') or self.meta['n_clusters']
        self.signature = feature_signature(c)
        self.root = Path(c['output_dir'])
        for d in ('metrics', 'masks', 'predictions', 'figures', 'logs'):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.e = c['experiments']
        self.artifacts = []
        self.long_rows = []
        baseline_config = c.get('baselines', {})
        mlp = baseline_config.get('mlp', {})
        self.baselines = CompletionBaselines(
            train_img, train_txt, val_img, val_txt, seed=self.seed, device=self.dev,
            val_rate=baseline_config.get('val_rate', .3),
            val_seed=self.seed + baseline_config.get('val_seed_offset', 777),
            knn_k=baseline_config.get('knn_k', (1, 3, 5, 10, 20)),
            ridge_alpha=baseline_config.get('ridge_alpha',
                                            (0.001, .01, .1, 1., 10., 100., 1000.)),
            mlp_hidden=mlp.get('hidden', 512),
            mlp_epochs=mlp.get('epochs', 100),
            mlp_patience=mlp.get('patience', 15),
            mlp_lr=mlp.get('lr', .001),
            mlp_weight_decay=mlp.get('weight_decay', .0001),
            mlp_batch_size=mlp.get('batch_size', 128))
        self.best_baseline, self.baseline_scores = self.baselines.best_method()
        configured = baseline_config.get('methods', list(METHODS))
        self.methods = [m for m in METHODS if m in configured] + ['diffusion']
        self.generated_weights = self.e.get(
            'generated_weights', [self.e.get('generated_weight', .5)])
        self.ddim_steps = self.e.get('ddim_steps', [c['sample']['steps']])

    def predict(self, direction, condition, unconditional, steps=None):
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
                        self.c['sample']['steps'] if steps is None else steps,
                        self.c['sample']['seed'] + (0 if direction == 'img2txt' else 1),
                        unconditional, self.c['sample']['batch_size']).cpu().numpy()

    def complete(self, mask, unconditional=False, steps=None):
        # Do not pass hidden targets into model; initialize unknown slots with zero.
        img = np.where(mask[:, :1], self.img, 0.).astype(np.float32)
        txt = np.where(mask[:, 1:], self.txt, 0.).astype(np.float32)
        for column, direction, source, target in [(0, 'txt2img', txt, img), (1, 'img2txt', img, txt)]:
            missing = mask[:, column] == 0
            if missing.any():
                if steps is None:
                    target[missing] = self.predict(direction, source[missing], unconditional)
                else:
                    target[missing] = self.predict(direction, source[missing], unconditional, steps)
        assert np.array_equal(img[mask[:, 0] == 1], self.img[mask[:, 0] == 1])
        assert np.array_equal(txt[mask[:, 1] == 1], self.txt[mask[:, 1] == 1])
        return img, txt

    def complete_with(self, method, mask, unconditional=False, steps=None):
        if method == 'diffusion':
            return self.complete(mask, unconditional, steps)
        if unconditional:
            raise ValueError('Unconditional mode is only defined for diffusion')
        img, txt = self.baselines.complete(method, self.img, self.txt, mask)
        assert np.array_equal(img[mask[:, 0] == 1], self.img[mask[:, 0] == 1])
        assert np.array_equal(txt[mask[:, 1] == 1], self.txt[mask[:, 1] == 1])
        return img, txt

    def mask(self, rate, pattern, tag):
        m = make_mask(len(self.img), rate, pattern, self.e['mask_seed'])
        pd.DataFrame({'sample_id': self.ids.sample_id, 'm_img': m[:, 0], 'm_txt': m[:, 1]}).to_csv(
            self.root / 'masks' / f'{tag}.csv', index=False)
        return m

    def score(self, img, txt, mask, tag, method, fusion=None, **extra):
        fused = fuse(img, txt) if fusion is None else fusion
        metrics, predicted = cluster_metrics(fused, self.labels, self.k,
                                             self.e['kmeans_seed'], self.e['n_init'])
        retrieval = retrieval_metrics(img, txt)
        reconstruction = {}
        for name, column, real, pred in [('i2t', 1, self.txt, txt), ('t2i', 0, self.img, img)]:
            missing = mask[:, column] == 0
            if missing.any():
                diff = real[missing] - pred[missing]
                reconstruction[f'MSE_{name}'] = float(np.mean(diff ** 2))
                reconstruction[f'cos_{name}'] = float(np.mean(
                    np.sum(real[missing] * pred[missing], axis=1)))
        np.savez_compressed(self.root / 'predictions' / f'{tag}.npz', features=fused,
                            img=img, txt=txt, clusters=predicted,
                            sample_ids=self.ids.sample_id.to_numpy(dtype=str))
        return {**extra, 'method': method, **metrics, **retrieval, **reconstruction,
                'n': len(fused), 'K': self.k,
                'missing_image': int((mask[:, 0] == 0).sum()),
                'missing_text': int((mask[:, 1] == 0).sum()),
                'actual_rate': float((mask.min(axis=1) == 0).mean())}

    def table(self, name, rows, long=True):
        df = pd.DataFrame(rows)
        if 'seed' not in df.columns:
            df['seed'] = self.seed
        path = self.root / 'metrics' / f'{name}.csv'
        df.to_csv(path, index=False)
        if long and not df.empty:
            keys = [k for k in ('seed', 'method', 'direction', 'pattern', 'rate',
                                'setting', 'steps') if k in df.columns]
            metrics = [m for m in METRIC_COLUMNS if m in df.columns]
            for _, row in df.iterrows():
                for metric in metrics:
                    value = row[metric]
                    if pd.notna(value) and np.isfinite(value):
                        self.long_rows.append({**{k: row[k] for k in keys},
                                               'experiment': name, 'metric': metric,
                                               'value': float(value)})
        print(f'\n{path}\n{df.head(12).to_string(index=False)}', flush=True)

    def exp1(self):
        rows, details = [], []
        for pattern, direction, column in [('text', 'img2txt', 1), ('image', 'txt2img', 0)]:
            m = self.mask(self.e['reconstruction_rate'], pattern, f'exp1_{direction}')
            missing = m[:, column] == 0
            if not missing.any():
                raise ValueError('No missing samples for exp1; increase reconstruction_rate')
            real = self.txt if column == 1 else self.img
            for method in self.methods:
                img, txt = self.complete_with(method, m)
                pred = txt if column == 1 else img
                mse = ((real[missing] - pred[missing]) ** 2).mean(axis=1)
                cosine = (real[missing] * pred[missing]).sum(axis=1)
                rows.append({'method': method, 'direction': direction,
                             'n_missing': len(mse), 'MSE': mse.mean(),
                             'cosine': cosine.mean()})
                for sid, a, b in zip(self.ids.sample_id[missing], mse, cosine):
                    details.append({'method': method, 'sample_id': sid,
                                    'direction': direction, 'MSE': a, 'cosine': b})
                np.savez_compressed(self.root / 'predictions' /
                                    f'exp1_{direction}_{method}.npz',
                                    real=real[missing], generated=pred[missing],
                                    sample_ids=self.ids.sample_id[missing].to_numpy(dtype=str))
        self.table('exp1_reconstruction', rows)
        self.table('exp1_per_sample', details, long=False)

    def exp2(self):
        rows = []
        for rate in [0.] + list(self.e['rates']):
            tag = f'exp2_r{rate:g}'
            m = self.mask(rate, 'random', tag)
            for method in self.methods:
                img, txt = self.complete_with(method, m)
                rows.append(self.score(img, txt, m, f'{tag}_{method}', method, rate=rate))
        self.table('exp2_missing_rate', rows)

    def exp3(self):
        rows = []
        for pattern in ('image', 'text', 'random'):
            tag = f'exp3_{pattern}'
            m = self.mask(self.e['rate'], pattern, tag)
            for method in self.methods:
                img, txt = self.complete_with(method, m)
                rows.append(self.score(img, txt, m, f'{tag}_{method}', method,
                                       pattern=pattern, rate=self.e['rate']))
        self.table('exp3_missing_pattern', rows)

    def exp4(self):
        m = self.mask(self.e['rate'], 'random', 'exp4_shared')
        img, txt = self.complete(m)
        uncond_img, uncond_txt = self.complete(m, unconditional=True)
        rows = [
            self.score(img, txt, m, 'exp4_conditional_concat', 'diffusion',
                       setting='conditional_concat', rate=self.e['rate']),
            self.score(uncond_img, uncond_txt, m, 'exp4_unconditional_concat', 'diffusion',
                       setting='unconditional_concat', rate=self.e['rate']),
        ]
        for weight in self.generated_weights:
            rows.append(self.score(img, txt, m, f'exp4_weight_{weight:g}', 'diffusion',
                                   fusion=fuse(img, txt, m, weight),
                                   setting=f'conditional_downweight_{weight:g}',
                                   rate=self.e['rate']))
        self.table('exp4_ablation', rows)

    def exp5(self):
        m = self.mask(self.e['rate'], 'random', 'exp5_shared')
        img, txt = self.complete(m)
        baseline_img, baseline_txt = self.complete_with(self.best_baseline, m)
        panels = [
            ('observed', fuse(self.img * m[:, :1], self.txt * m[:, 1:])),
            ('completed', fuse(img, txt)),
            (f'best_baseline_{self.best_baseline}', fuse(baseline_img, baseline_txt)),
            ('complete_reference', fuse(self.img, self.txt)),
        ]
        joined = np.concatenate([matrix for _, matrix in panels])
        pca = PCA(n_components=min(50, joined.shape[1], len(joined) - 1), svd_solver='full')
        reduced = pca.fit_transform(joined)
        coordinates = TSNE(n_components=2,
                           perplexity=max(1, min(self.e['tsne_perplexity'], len(joined) - 1)),
                           init='pca', learning_rate='auto',
                           random_state=self.seed).fit_transform(reduced)
        rows = []
        for j, (name, _) in enumerate(panels):
            for idx, (x, y) in enumerate(coordinates[j * len(img):(j + 1) * len(img)]):
                rows.append({'panel': name, 'sample_id': self.ids.sample_id.iloc[idx],
                             'label': self.labels[idx], 'x': x, 'y': y})
        self.table('exp5_tsne_coordinates', rows, long=False)
        np.savez_compressed(self.root / 'predictions' / 'exp5_inputs.npz', joined=joined,
                            pca_components=pca.components_, pca_mean=pca.mean_,
                            panels=np.array([name for name, _ in panels]))

    def exp6(self):
        rows = []
        m = self.mask(self.e['rate'], 'random', 'exp6_shared')
        for steps in self.ddim_steps:
            img, txt = self.complete(m, steps=steps)
            rows.append(self.score(img, txt, m, f'exp6_steps_{steps}', 'diffusion',
                                   steps=steps, rate=self.e['rate']))
        self.table('exp6_ddim_steps', rows)

    def exp7(self):
        rows = []
        m = self.mask(self.e['rate'], 'random', 'exp7_shared')
        for method in self.methods:
            img, txt = self.complete_with(method, m)
            rows.append(self.score(img, txt, m, f'exp7_{method}', method,
                                   rate=self.e['rate']))
        self.table('exp7_retrieval', rows)

    def run(self, selected):
        for number in selected:
            getattr(self, f'exp{number}')()
        long_path = self.root / 'metrics' / 'metrics_long.csv'
        columns = ['seed', 'experiment', 'method', 'direction', 'pattern', 'rate',
                   'setting', 'steps', 'metric', 'value']
        long_df = pd.DataFrame(self.long_rows, columns=columns)
        long_df.to_csv(long_path, index=False)
        dump(self.root / 'logs' / 'evaluation_run.json',
             {**runtime(self.c), 'seed': self.seed, 'experiments': selected,
              'feature_signature': self.signature, 'checkpoints': self.artifacts,
              'methods': self.methods, 'best_baseline': self.best_baseline,
              'baseline_scores': self.baseline_scores,
              'baseline_choices': self.baselines.choices,
              'synthetic': self.meta.get('synthetic', False),
              'note': 'Single seed per directory; aggregate seeds before reporting.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--exp', choices=[str(i) for i in range(1, 8)] + ['all'], default='all')
    a = p.parse_args()
    Experiments(config(a.config)).run(
        list(range(1, 8)) if a.exp == 'all' else [int(a.exp)])
