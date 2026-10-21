"""Download official UIUC Pascal Sentences; choose the first available caption.
Category comes from the original image URL directory, never from caption keywords.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from PIL import Image

URL = 'https://vision.cs.uiuc.edu/pascal-sentences/'
CATEGORIES = set('aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split())


def parse_index(html):
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for img in soup.find_all('img'):
        src = img.get('src', '')
        parts = urlparse(src).path.strip('/').split('/')
        if len(parts) < 2 or parts[-2] not in CATEGORIES:
            continue
        category, filename = parts[-2:]
        row = img.find_parent('tr')
        # Official page has a nested caption table; take leaf table cells only.
        captions = [td.get_text(' ', strip=True) for td in row.find_all('td')
                    if not td.find('table') and not td.find('img') and not td.find('a')]
        captions = [x for x in captions if x and x != '|']
        if not captions:
            raise ValueError(f'{filename}: expected nonempty captions, found {len(captions)}; source layout changed')
        rows.append({'sample_id': Path(filename).stem, 'category': category,
                     'filename': filename, 'url': urljoin(URL, src), 'captions': captions})
    if len(rows) != 1000 or len({r['sample_id'] for r in rows}) != 1000:
        raise ValueError(f'Expected 1000 unique official entries, got {len(rows)}; inspect source before proceeding')
    if {r['category'] for r in rows} != CATEGORIES:
        raise ValueError('Official category coverage changed')
    return rows


def download(out):
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    response = session.get(URL, timeout=60)
    response.raise_for_status()
    (out / 'official_index.html').write_text(response.text, encoding='utf-8')
    entries = parse_index(response.text)
    manifest = []
    for j, entry in enumerate(entries):
        relative = Path('images') / entry['category'] / entry['filename']
        image_path = out / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        valid = False
        if image_path.exists():
            try:
                with Image.open(image_path) as im:
                    im.verify()
                valid = True
            except OSError:
                pass
        if not valid:
            response = session.get(entry['url'], timeout=60)
            response.raise_for_status()
            temporary = image_path.with_suffix('.part')
            temporary.write_bytes(response.content)
            with Image.open(temporary) as im:
                im.verify()
            temporary.replace(image_path)
        manifest.append({'sample_id': entry['sample_id'], 'image_path': relative.as_posix(),
                         'text': entry['captions'][0], 'label': entry['category']})
        if (j + 1) % 50 == 0:
            print(f'{j+1}/{len(entries)} downloaded/verified', flush=True)
    (out / 'all_captions.json').write_text(json.dumps(entries, indent=2), encoding='utf-8')
    pd.DataFrame(manifest).to_csv(out / 'manifest.csv', index=False)
    print(f'Ready: {out / "manifest.csv"}. Use the original caption/image terms; see official source.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data')
    download(Path(parser.parse_args().output))
