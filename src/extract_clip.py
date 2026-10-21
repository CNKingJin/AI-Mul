"""Frozen CLIP projected embeddings, L2 normalized and cached with source provenance."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from common import config, device, dump, fingerprint, norm, seed_all


def extract(c):
    seed_all(c['seed'])
    dev = device(c)
    options = {'local_files_only': c['clip']['local_files_only']}
    if c['clip'].get('revision'):
        options['revision'] = c['clip']['revision']
    model = CLIPModel.from_pretrained(c['clip']['model'], **options).to(dev).eval()
    model.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(c['clip']['model'], **options)
    if model.config.projection_dim != c['model']['dim']:
        raise ValueError('CLIP projection dimension does not match config')
    for split in ('train', 'val', 'test'):
        source = Path(c['splits_dir']) / f'{split}.csv'
        df = pd.read_csv(source, dtype=str)
        images, texts = [], []
        batch = c['clip']['batch_size']
        with torch.inference_mode():
            for start in range(0, len(df), batch):
                part = df.iloc[start:start + batch]
                pil = []
                for path in part.image_path:
                    with Image.open(path) as image:
                        pil.append(image.convert('RGB'))
                ip = processor(images=pil, return_tensors='pt').to(dev)
                tp = processor(text=part.text.tolist(), truncation=True, max_length=77,
                               padding=True, return_tensors='pt').to(dev)
                images.append(model.get_image_features(**ip).float().cpu().numpy())
                texts.append(model.get_text_features(**tp).float().cpu().numpy())
        out = Path(c['features_dir']) / split
        out.mkdir(parents=True, exist_ok=True)
        for name, vectors in [('z_img', images), ('z_txt', texts)]:
            np.save(out / f'{name}.npy', norm(np.concatenate(vectors)))
        df[['sample_id']].to_csv(out / 'ids.csv', index=False)
        # Labels deliberately stay in splits/*.csv, never in the feature cache.
        dump(out / 'provenance.json', {'split_sha256': fingerprint(source),
             'clip': c['clip'], 'resolved_model_commit': getattr(model.config, '_commit_hash', None),
             'rows': len(df), 'dim': c['model']['dim']})
        print(f'{split}: cached {len(df)} aligned pairs')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    extract(config(p.parse_args().config))
