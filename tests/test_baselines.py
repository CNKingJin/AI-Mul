import numpy as np
import pytest

from baselines import CompletionBaselines, METHODS
from common import norm


def _data(seed=0, n_train=48, n_val=18, dim=8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim))
    out = {}
    for name, n in [('train', n_train), ('val', n_val)]:
        latent = centers[np.arange(n) % 3] + rng.normal(0, .1, (n, dim))
        out[name] = (norm(latent + rng.normal(0, .05, (n, dim))),
                     norm(latent + rng.normal(0, .05, (n, dim))))
    return out


@pytest.fixture(scope='module')
def fitted():
    data = _data()
    model = CompletionBaselines(
        data['train'][0], data['train'][1], data['val'][0], data['val'][1],
        seed=42, device='cpu', val_rate=.3, val_seed=777,
        knn_k=(1, 3, 5), ridge_alpha=(.1, 1., 10.),
        mlp_hidden=16, mlp_epochs=5, mlp_patience=2, mlp_batch_size=8)
    return model, data


def test_choices_and_scores(fitted):
    model, _ = fitted
    for direction in ('img2txt', 'txt2img'):
        assert set(model.choices[direction]) == {'knn', 'ridge', 'mlp'}
        assert 1 <= model.choices[direction]['knn'] <= 48
        assert model.choices[direction]['ridge'] in (.1, 1., 10.)
        assert np.isfinite(model.val_mse[direction]['zero'])
        assert set(model.val_mse[direction]) == set(METHODS)
    best, scores = model.best_method()
    assert best in METHODS
    assert set(scores) == set(METHODS)


def test_completion_preserves_observations_and_fills_missing(fitted):
    model, data = fitted
    img, txt = data['val']
    mask = np.array([[0, 1], [1, 0], [0, 1]])
    for method in METHODS:
        out_img, out_txt = model.complete(method, img[:3], txt[:3], mask)
        np.testing.assert_array_equal(out_img[mask[:, 0] == 1], img[:3][mask[:, 0] == 1])
        np.testing.assert_array_equal(out_txt[mask[:, 1] == 1], txt[:3][mask[:, 1] == 1])
        if method == 'zero':
            np.testing.assert_array_equal(out_img[0], np.zeros(8))
        else:
            np.testing.assert_allclose(np.linalg.norm(out_img[0]), 1., atol=1e-5)


def test_completion_does_not_read_hidden_targets(fitted):
    model, data = fitted
    img, txt = data['val']
    mask = np.array([[0, 1], [1, 0], [0, 1]])
    for method in METHODS:
        a_img, a_txt = model.complete(method, img[:3].copy(), txt[:3].copy(), mask)
        changed_img = img[:3].copy()
        changed_txt = txt[:3].copy()
        changed_img[0] = 100.
        changed_txt[1] = -100.
        b_img, b_txt = model.complete(method, changed_img, changed_txt, mask)
        np.testing.assert_array_equal(a_img[0], b_img[0])
        np.testing.assert_array_equal(a_txt[1], b_txt[1])


def test_unknown_baseline_rejected(fitted):
    model, data = fitted
    with pytest.raises(ValueError):
        model.complete('nearest', data['val'][0][:2], data['val'][1][:2],
                       np.ones((2, 2), dtype=int))
