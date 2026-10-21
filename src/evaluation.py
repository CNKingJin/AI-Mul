"""Exact missing-sample semantics; label-free completion and fusion."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def make_mask(n, rate, pattern, seed):
    if not 0 <= rate <= 1:
        raise ValueError('rate must be in [0,1]')
    if pattern not in ('random', 'image', 'text'):
        raise ValueError('invalid mask pattern')
    rng = np.random.default_rng(seed)
    mask = np.ones((n, 2), dtype=np.int64)
    chosen = rng.permutation(n)[:round(n * rate)]
    if pattern == 'random':
        # Fixed interleaved direction assignments make rates nested under the same seed.
        mask[chosen[::2], 0] = 0
        mask[chosen[1::2], 1] = 0
    else:
        mask[chosen, 0 if pattern == 'image' else 1] = 0
    return mask


def fuse(img, txt, mask=None, generated_weight=1.):
    weights = np.ones((len(img), 2)) if mask is None else np.where(mask == 1, 1., generated_weight)
    fused = np.concatenate([img * weights[:, :1], txt * weights[:, 1:]], axis=1)
    return (fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def cluster_metrics(features, labels, k, seed=42, n_init=20):
    y = np.unique(labels, return_inverse=True)[1]
    if len(features) < k or k < 2:
        raise ValueError('K must be >=2 and <= number of test samples')
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(features)
    contingency = np.zeros((k, len(np.unique(y))), dtype=np.int64)
    np.add.at(contingency, (km.labels_, y), 1)
    r, col = linear_sum_assignment(-contingency)
    return {'ACC': float(contingency[r, col].sum() / len(y)),
            'NMI': float(normalized_mutual_info_score(y, km.labels_)),
            'ARI': float(adjusted_rand_score(y, km.labels_)), 'inertia': float(km.inertia_)}, km.labels_
