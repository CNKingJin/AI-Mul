"""Shared I/O; commands run from repository root. No labels in model inputs."""
import hashlib
import json
import os
from pathlib import Path
import random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
import yaml


def config(path):
    with open(path, encoding='utf-8') as f:
        c = yaml.safe_load(f)
    for k in ('splits_dir', 'features_dir', 'output_dir'):
        Path(c[k]).mkdir(parents=True, exist_ok=True)
    return c


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def device(c):
    requested = c['device']
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check GPU PyTorch/driver, or set device: cpu for self-test.')
    return torch.device(requested)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Features must be a finite 2D matrix')
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError('Zero feature vector')
    return x / lengths


def fingerprint(*paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()


def feature_signature(c):
    root = Path(c['features_dir'])
    return fingerprint(*[root / s / name for s in ('train', 'val')
                         for name in ('z_img.npy', 'z_txt.npy', 'ids.csv')])


def load_features(c, split):
    root = Path(c['features_dir']) / split
    img, txt = [norm(np.load(root / n, allow_pickle=False)) for n in ('z_img.npy', 'z_txt.npy')]
    ids = pd.read_csv(root / 'ids.csv', dtype=str)
    if img.shape != txt.shape or img.shape != (len(ids), c['model']['dim']):
        raise ValueError(f'{split}: feature dimension/order length mismatch')
    if ids.sample_id.duplicated().any():
        raise ValueError('Duplicate feature IDs')
    split_path = Path(c['splits_dir']) / f'{split}.csv'
    provenance = root / 'provenance.json'
    if provenance.exists():
        import json
        recorded = json.loads(provenance.read_text())
        if recorded['split_sha256'] != fingerprint(split_path):
            raise ValueError('Stale features: split content changed; rerun extraction')
    split_ids = pd.read_csv(split_path, dtype=str)
    if ids.sample_id.tolist() != split_ids.sample_id.tolist():
        raise ValueError('Stale features: split IDs/order changed; rerun extraction')
    return img, txt, ids


def checkpoint_path(c, direction, unconditional=False):
    return Path(c['output_dir']) / 'checkpoints' / f'{direction}_{"uncond" if unconditional else "cond"}.pt'


def runtime(c):
    import importlib.metadata
    import platform
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'packages': {n: importlib.metadata.version(n) for n in
                         ('numpy', 'scikit-learn', 'diffusers', 'transformers')}, 'config': c}
