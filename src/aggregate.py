"""Aggregate per-seed metrics with t-based intervals and paired deltas."""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from common import config, dump

GROUP_COLUMNS = ['experiment', 'method', 'direction', 'pattern', 'rate',
                 'setting', 'steps', 'metric']
BASE_COLUMNS = [k for k in GROUP_COLUMNS if k != 'method']


def load_long(seed_root):
    files = sorted(Path(seed_root).glob('seed_*/metrics/metrics_long.csv'))
    if not files:
        raise FileNotFoundError(f'No metrics_long.csv under {seed_root}')
    frames = [pd.read_csv(path) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError('All metrics_long.csv files are empty')
    df = pd.concat(frames, ignore_index=True)
    if 'seed' not in df.columns:
        raise ValueError('metrics_long.csv must contain a seed column')
    return df


def summarize(long_df):
    keys = [k for k in GROUP_COLUMNS if k in long_df.columns
            and long_df[k].notna().any()]
    if not keys:
        raise ValueError('No grouping columns available')
    summary = long_df.groupby(keys, dropna=False)['value'].agg(
        ['mean', 'std', 'count']).reset_index()
    bounds = summary.apply(_interval, axis=1, result_type='expand')
    summary['ci_low'] = bounds[0]
    summary['ci_high'] = bounds[1]
    return summary.sort_values(keys).reset_index(drop=True)


def _interval(row):
    if row['count'] < 2 or not np.isfinite(row['std']):
        return row['mean'], row['mean']
    half = stats.t.ppf(.975, row['count'] - 1) * row['std'] / np.sqrt(row['count'])
    return row['mean'] - half, row['mean'] + half


def _pvalues(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return float('nan'), float('nan')
    t_p = float(stats.ttest_1samp(values, 0.).pvalue)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            w_p = float(stats.wilcoxon(values).pvalue)
        except ValueError:
            w_p = float('nan')
    return t_p, w_p


def paired_deltas(long_df):
    empty = pd.DataFrame(columns=BASE_COLUMNS + ['baseline', 'mean_delta',
                                                 'std_delta', 'count', 'ci_low',
                                                 'ci_high', 't_p', 'wilcoxon_p'])
    if 'diffusion' not in set(long_df.method):
        return empty
    records = []
    for experiment, group in long_df.groupby('experiment'):
        keys = [k for k in ('direction', 'pattern', 'rate', 'setting', 'steps', 'metric')
                if group[k].notna().any()]
        pivot = group.pivot_table(index=['seed'] + keys, columns='method',
                                  values='value', aggfunc='mean')
        if 'diffusion' not in pivot.columns:
            continue
        for method in pivot.columns:
            if method == 'diffusion':
                continue
            sub = pivot[['diffusion', method]].dropna()
            for index, row in sub.iterrows():
                records.append({'experiment': experiment, 'seed': index[0],
                                **dict(zip(keys, index[1:])), 'baseline': method,
                                'delta': float(row['diffusion'] - row[method])})
    if not records:
        return empty
    delta_df = pd.DataFrame(records)
    keys = [k for k in ('experiment', 'direction', 'pattern', 'rate', 'setting',
                        'steps', 'metric')
            if k in delta_df.columns and delta_df[k].notna().any()]
    rows = []
    for group_key, group in delta_df.groupby(keys + ['baseline'], dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group['delta'].to_numpy(dtype=float)
        count = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else float('nan')
        half = (stats.t.ppf(.975, count - 1) * std / np.sqrt(count)
                if count > 1 and np.isfinite(std) else 0.)
        t_p, w_p = _pvalues(values)
        rows.append({**dict(zip(keys + ['baseline'], group_key)),
                     'mean_delta': mean, 'std_delta': std, 'count': count,
                     'ci_low': mean - half, 'ci_high': mean + half,
                     't_p': t_p, 'wilcoxon_p': w_p})
    paired = pd.DataFrame(rows)
    return paired.sort_values(keys + ['baseline']).reset_index(drop=True)


def aggregate(c, seed_root=None, out_root=None):
    seed_root = Path(seed_root) if seed_root else Path(c['output_dir']) / 'seeds'
    out_root = Path(out_root) if out_root else Path(c['output_dir']) / 'aggregated'
    out_root.mkdir(parents=True, exist_ok=True)
    long_df = load_long(seed_root)
    summary = summarize(long_df)
    paired = paired_deltas(long_df)
    summary.to_csv(out_root / 'summary.csv', index=False)
    paired.to_csv(out_root / 'paired_delta.csv', index=False)
    dump(out_root / 'aggregate.json', {
        'seed_root': str(seed_root), 'seeds': sorted(long_df.seed.unique().tolist()),
        'n_seeds': int(long_df.seed.nunique()), 'rows': int(len(long_df)),
        'note': 'Intervals are t-based across seeds; exploratory with few seeds.'})
    from plot_results import plot_summary
    plot_summary(summary, out_root / 'figures')
    print(f'Aggregated {long_df.seed.nunique()} seeds into {out_root}', flush=True)
    return summary, paired


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--seed-root', default=None)
    p.add_argument('--out-root', default=None)
    a = p.parse_args()
    aggregate(config(a.config), a.seed_root, a.out_root)
