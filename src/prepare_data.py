"""Validate paired data, reject image leakage, create stratified 70/15/15 splits."""
import argparse
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from common import config, dump


def prepare(c):
    source = Path(c['manifest']).resolve()
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = ['sample_id', 'image_path', 'text', 'label']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Manifest requires {required}')
    if df.sample_id.duplicated().any():
        raise ValueError('Duplicate sample_id: choose one caption per image first')
    valid, rejected, seen = [], [], set()
    for row in df[required].to_dict('records'):
        try:
            if any(not row[k].strip() for k in required):
                raise ValueError('empty field')
            p = Path(row['image_path'])
            p = p if p.is_absolute() else source.parent / p
            p = p.resolve()
            with Image.open(p) as image:
                image.convert('RGB').load()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError('duplicate image content (one caption per image required)')
            seen.add(digest)
            row.update(image_path=str(p), image_sha256=digest)
            valid.append(row)
        except (OSError, ValueError) as e:
            rejected.append({**row, 'reason': str(e)})
    root = Path(c['splits_dir'])
    pd.DataFrame(rejected, columns=required + ['reason']).to_csv(root / 'rejected.csv', index=False)
    clean = pd.DataFrame(valid)
    if len(clean) == 0:
        raise ValueError('No valid pairs; inspect rejected.csv')
    counts = clean.label.value_counts()
    if len(counts) < 2 or counts.min() < 7:
        raise ValueError('Need >=2 classes and >=7 valid unique images per class for 70/15/15 split')
    train, rest = train_test_split(clean, test_size=0.3, stratify=clean.label, random_state=c['seed'])
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=c['seed'])
    classes = sorted(counts.index.tolist())
    for name, part in [('train', train), ('val', val), ('test', test)]:
        if set(part.label) != set(classes):
            raise ValueError(f'{name} misses a class; use more samples')
        part.to_csv(root / f'{name}.csv', index=False)
    metadata = {'n_clusters': len(classes), 'classes': classes, 'valid_pairs': len(clean),
                'rejected': len(rejected), 'counts': counts.to_dict(),
                'split_counts': {'train': len(train), 'val': len(val), 'test': len(test)},
                'source_manifest': str(source), 'synthetic': False}
    dump(root / 'metadata.json', metadata)
    print(metadata)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    prepare(config(p.parse_args().config))
