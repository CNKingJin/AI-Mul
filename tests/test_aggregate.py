import numpy as np
import pandas as pd

from aggregate import aggregate, paired_deltas, summarize


def _write_seed(root, seed, diffusion, zero):
    path = root / f'seed_{seed}' / 'metrics' / 'metrics_long.csv'
    path.parent.mkdir(parents=True)
    rows = []
    for method, value in [('diffusion', diffusion), ('zero', zero)]:
        for rate in (0.1, 0.3):
            rows.append({'seed': seed, 'experiment': 'exp2_missing_rate',
                         'method': method, 'direction': np.nan, 'pattern': np.nan,
                         'rate': rate, 'setting': np.nan, 'steps': np.nan,
                         'metric': 'ACC', 'value': value + rate / 10})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_summary_and_paired_delta(tmp_path):
    _write_seed(tmp_path / 'seeds', 42, .8, .6)
    _write_seed(tmp_path / 'seeds', 43, .9, .7)
    df = pd.concat([pd.read_csv(tmp_path / 'seeds' / f'seed_{seed}' /
                                'metrics' / 'metrics_long.csv')
                    for seed in (42, 43)], ignore_index=True)
    summary = summarize(df)
    row = summary[(summary.method == 'diffusion') & (summary.rate == 0.1)].iloc[0]
    assert np.isclose(row['mean'], .86)
    assert row['count'] == 2
    assert row['ci_low'] < row['mean'] < row['ci_high']
    paired = paired_deltas(df)
    delta = paired[(paired.baseline == 'zero') & (paired.rate == 0.1)].iloc[0]
    assert np.isclose(delta['mean_delta'], .2)
    assert delta['count'] == 2


def test_aggregate_writes_artifacts(tmp_path):
    _write_seed(tmp_path / 'seeds', 42, .8, .6)
    _write_seed(tmp_path / 'seeds', 43, .9, .7)
    summary, paired = aggregate({}, seed_root=tmp_path / 'seeds',
                                out_root=tmp_path / 'aggregated')
    assert not summary.empty
    assert not paired.empty
    assert (tmp_path / 'aggregated' / 'summary.csv').stat().st_size > 0
    assert (tmp_path / 'aggregated' / 'paired_delta.csv').stat().st_size > 0
    assert (tmp_path / 'aggregated' / 'aggregate.json').stat().st_size > 0
    assert (tmp_path / 'aggregated' / 'figures' / 'exp2_missing_rate.png').stat().st_size > 0
