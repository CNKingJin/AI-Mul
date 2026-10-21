import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluation import make_mask, fuse, cluster_metrics
from diffusion import Denoiser, scheduler, generate
from common import norm, config, load_features
from prepare_data import prepare
from download_pascal import parse_index


@pytest.mark.parametrize('pattern', ['random', 'image', 'text'])
def test_masks(pattern):
    for r in (0, .1, .3, .5, .7, 1):
        m = make_mask(101, r, pattern, 42)
        assert (m.sum(axis=1) >= 1).all()
        assert (m.sum(axis=1) == 1).sum() == round(101 * r)
        np.testing.assert_array_equal(m, make_mask(101, r, pattern, 42))
        if pattern == 'image':
            assert m[:, 1].all()
        if pattern == 'text':
            assert m[:, 0].all()


def test_acc_hungarian_and_scores():
    x = np.array([[1, 0], [1, .01], [-1, 0], [-1, .01]], dtype=float)
    metrics, _ = cluster_metrics(x, ['z', 'z', 'a', 'a'], 2)
    assert all(metrics[k] == 1 for k in ['ACC', 'NMI', 'ARI'])


def test_fusion_downweights_only_generated_side():
    img, txt = np.eye(2), np.eye(2)
    mask = np.array([[0, 1], [1, 1]])
    z = fuse(img, txt, mask, .5)
    assert np.isclose(np.linalg.norm(z[0, :2]) / np.linalg.norm(z[0, 2:]), .5)
    np.testing.assert_allclose(z[1], fuse(img, txt)[1])


def test_ddim_reproducible_and_terminal_noise():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = dict(dim=8, hidden=16, time_dim=8, timesteps=100, beta_schedule='squaredcos_cap_v2')
    net, sch = Denoiser(**cfg), scheduler(cfg)
    assert sch.alphas_cumprod[-1] < 1e-3
    condition = torch.nn.functional.normalize(torch.randn(7, 8), dim=1)
    a = generate(net, sch, condition, 8 ** .5, 10, 42)
    b = generate(net, sch, condition, 8 ** .5, 10, 42)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.allclose(a.norm(dim=1), torch.ones(7))
    assert not torch.allclose(a, generate(net, sch, -condition, 8 ** .5, 10, 42))
    u1 = generate(net, sch, condition, 8 ** .5, 10, 42, True)
    u2 = generate(net, sch, -condition, 8 ** .5, 10, 42, True)
    torch.testing.assert_close(u1, u2)


def test_normalization_rejects_bad_features():
    with pytest.raises(ValueError):
        norm(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        norm(np.array([[np.nan]]))


def test_data_preparation_and_duplicate_isolation(tmp_path):
    from PIL import Image
    rows = []
    for cl in range(3):
        for j in range(20):
            p = tmp_path / f'{cl}_{j}.png'
            Image.new('RGB', (8, 8), (cl * 50, j * 5, 0)).save(p)
            rows.append(dict(sample_id=f'{cl}_{j}', image_path=p.name, text='test caption', label=str(cl)))
    rows.append({**rows[0], 'sample_id': 'duplicate'})
    pd.DataFrame(rows).to_csv(tmp_path / 'manifest.csv', index=False)
    c = {'manifest': str(tmp_path / 'manifest.csv'), 'splits_dir': str(tmp_path / 'splits'), 'seed': 42}
    Path(c['splits_dir']).mkdir()
    prepare(c)
    parts = [pd.read_csv(Path(c['splits_dir']) / f'{s}.csv') for s in ['train', 'val', 'test']]
    assert sum(len(p) for p in parts) == 60
    for a in range(3):
        for b in range(a + 1, 3):
            assert not set(parts[a].image_sha256) & set(parts[b].image_sha256)


def test_tiny_clip_projection_api():
    from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel
    tc = CLIPTextConfig(vocab_size=100, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)
    vc = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, image_size=16, patch_size=8)
    model = CLIPModel(CLIPConfig.from_text_vision_configs(tc, vc, projection_dim=8)).eval()
    with torch.no_grad():
        assert model.get_image_features(pixel_values=torch.randn(2, 3, 16, 16)).shape == (2, 8)
        assert model.get_text_features(input_ids=torch.ones(2, 8, dtype=torch.long)).shape == (2, 8)


def test_pascal_parser_fails_closed():
    with pytest.raises(ValueError, match='1000'):
        parse_index('<html>unavailable</html>')


def test_completion_ignores_hidden_targets_and_preserves_observations():
    from run_experiments import Experiments
    exp = Experiments.__new__(Experiments)
    exp.img = norm(np.array([[1., 2.], [3., 1.], [2., 1.]]))
    exp.txt = norm(np.array([[2., 1.], [1., 3.], [1., 2.]]))
    exp.predict = lambda direction, condition, unconditional: condition.copy()
    mask = np.array([[0, 1], [1, 0], [1, 1]])
    a, b = exp.complete(mask)
    exp.img[0] = [100., -100.]
    exp.txt[1] = [-100., 100.]
    aa, bb = exp.complete(mask)
    np.testing.assert_array_equal(a, aa)
    np.testing.assert_array_equal(b, bb)
    np.testing.assert_array_equal(aa[mask[:, 0] == 1], exp.img[mask[:, 0] == 1])
    np.testing.assert_array_equal(bb[mask[:, 1] == 1], exp.txt[mask[:, 1] == 1])


def test_full_clip_extraction_offline_with_tiny_random_model(tmp_path):
    """Exercise actual from_pretrained, processor, both encoders and cache I/O; no pretrained claims."""
    import json
    from PIL import Image
    from transformers import (CLIPConfig, CLIPTextConfig, CLIPVisionConfig, CLIPModel,
                              CLIPTokenizer, CLIPProcessor, CLIPImageProcessor)
    from transformers.models.clip.tokenization_clip import bytes_to_unicode
    from extract_clip import extract
    vocab = list(bytes_to_unicode().values())
    vocab += [v + '</w>' for v in vocab]
    vocab += ['<|startoftext|>', '<|endoftext|>']
    model_path = tmp_path / 'model'
    model_path.mkdir()
    (model_path / 'vocab.json').write_text(json.dumps(dict(zip(vocab, range(len(vocab))))))
    (model_path / 'merges.txt').write_text('#version: 0.2\n')
    tokenizer = CLIPTokenizer(str(model_path / 'vocab.json'), str(model_path / 'merges.txt'))
    processor = CLIPProcessor(tokenizer=tokenizer, image_processor=CLIPImageProcessor(
        size={'shortest_edge': 16}, crop_size={'height': 16, 'width': 16}))
    processor.save_pretrained(model_path)
    tc = CLIPTextConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
                        bos_token_id=512, eos_token_id=513, pad_token_id=513)
    vc = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, image_size=16, patch_size=8)
    CLIPModel(CLIPConfig.from_text_vision_configs(tc, vc, projection_dim=8)).save_pretrained(model_path)
    splits = tmp_path / 'splits'
    splits.mkdir()
    for split in ('train', 'val', 'test'):
        rows = []
        for j in range(2):
            image_path = tmp_path / f'{split}_{j}.png'
            Image.new('RGB', (20, 20), (j * 80, 10, 40)).save(image_path)
            rows.append(dict(sample_id=f'{split}_{j}', image_path=str(image_path), text='test image', label='a'))
        pd.DataFrame(rows).to_csv(splits / f'{split}.csv', index=False)
    c = {'seed': 42, 'device': 'cpu', 'splits_dir': str(splits), 'features_dir': str(tmp_path / 'features'),
         'model': {'dim': 8}, 'clip': {'model': str(model_path), 'local_files_only': True, 'batch_size': 2}}
    extract(c)
    for split in ('train', 'val', 'test'):
        img, txt, ids = load_features(c, split)
        assert img.shape == txt.shape == (2, 8)
        np.testing.assert_allclose(np.linalg.norm(img, axis=1), 1., atol=1e-6)
    stale = pd.read_csv(splits / 'test.csv')
    stale.loc[0, 'text'] = 'changed caption'
    stale.to_csv(splits / 'test.csv', index=False)
    with pytest.raises(ValueError, match='Stale features'):
        load_features(c, 'test')


def test_pascal_accepts_official_caption_count_variation():
    # Regression: live official page has three 4-caption entries and one 6-caption entry.
    from download_pascal import CATEGORIES
    rows = []
    for category in sorted(CATEGORIES):
        for j in range(50):
            n = 4 if j == 0 else 6 if j == 1 else 5
            captions = ''.join(f'<tr><td>caption {i}</td></tr>' for i in range(n))
            rows.append(f'<tr><td><img src="{category}/{category}_{j}.jpg"></td><td><table>{captions}</table></td></tr>')
    parsed = parse_index('<html><body><table>' + ''.join(rows) + '</table></body></html>')
    assert len(parsed) == 1000
    assert len(parsed[0]['captions']) == 4
    assert len(parsed[1]['captions']) == 6
