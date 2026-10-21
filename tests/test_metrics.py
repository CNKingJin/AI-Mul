import numpy as np

from evaluation import reconstruction_metrics, retrieval_metrics


def test_retrieval_identity_is_perfect():
    x = np.eye(5, dtype=np.float32)
    metrics = retrieval_metrics(x, x)
    assert metrics['R1_i2t'] == 1.
    assert metrics['R5_i2t'] == 1.
    assert metrics['R1_t2i'] == 1.
    assert metrics['median_rank_i2t'] == 1.


def test_retrieval_zero_rows_count_as_failures():
    x = np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.float32)
    metrics = retrieval_metrics(x, x)
    assert np.isclose(metrics['R1_i2t'], 2. / 3.)
    assert np.isclose(metrics['R1_t2i'], 2. / 3.)


def test_reconstruction_metrics_only_score_missing():
    real = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    pred = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
    mask = np.array([True, False])
    metrics = reconstruction_metrics(real, pred, mask)
    assert metrics['n_missing'] == 1
    assert np.isclose(metrics['MSE'], 1.)
    assert np.isclose(metrics['cosine'], 0.)
