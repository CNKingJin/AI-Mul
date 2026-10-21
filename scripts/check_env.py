import sys
import torch
from transformers import CLIPModel, CLIPProcessor
from diffusers import DDIMScheduler
print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    a = torch.randn(32, 32, device='cuda')
    print('GPU matmul:', (a @ a.T).shape)
else:
    raise SystemExit('GPU unavailable: install CUDA PyTorch and check nvidia-smi. CPU smoke_test still works.')
