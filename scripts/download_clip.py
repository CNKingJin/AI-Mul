"""Run on an online machine; copy the resulting directory to the offline server."""
import argparse
from huggingface_hub import snapshot_download
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', default='models/clip-vit-base-patch32')
p.add_argument('--revision', default=None)
a = p.parse_args()
path = snapshot_download('openai/clip-vit-base-patch32', local_dir=a.output, revision=a.revision,
                         allow_patterns=['*.json', 'merges.txt', 'vocab.json', 'pytorch_model.bin', '*.safetensors'])
print('Downloaded:', path)
