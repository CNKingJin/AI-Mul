"""Train both directions; validation noise is fixed across epochs."""
import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from common import (config, device, dump, seed_all, load_features, checkpoint_path,
                    feature_signature, runtime)
from diffusion import Denoiser, scheduler


def train(c, direction, unconditional=False):
    seed_all(c['seed'])
    dev = device(c)
    img, txt, _ = load_features(c, 'train')
    vi, vt, _ = load_features(c, 'val')
    # Training and checkpoint provenance use only train/val, never test.
    signature = feature_signature(c)
    target, cond = (txt, img) if direction == 'img2txt' else (img, txt)
    vtarget, vcond = (vt, vi) if direction == 'img2txt' else (vi, vt)
    scale = float(c['train']['feature_scale'])
    x, condition = torch.from_numpy(target) * scale, torch.from_numpy(cond) * scale
    vx = torch.from_numpy(vtarget).to(dev) * scale
    vc = torch.from_numpy(vcond).to(dev) * scale
    if unconditional:
        condition.zero_()
        vc.zero_()
    loader = DataLoader(TensorDataset(x, condition), batch_size=c['train']['batch_size'],
                        shuffle=True, generator=torch.Generator().manual_seed(c['seed']), num_workers=0)
    model = Denoiser(**c['model']).to(dev)
    schedule = scheduler(c['model'])
    if float(schedule.alphas_cumprod[-1]) > 0.01:
        raise ValueError('Terminal alpha_bar too large for Gaussian initialization; use cosine schedule')
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['train']['lr'],
                                 weight_decay=c['train']['weight_decay'])
    vg = torch.Generator(device=dev).manual_seed(c['seed'] + 1000)
    val_noise = torch.randn(vx.shape, generator=vg, device=dev)
    val_t = torch.randint(0, c['model']['timesteps'], (len(vx),), generator=vg, device=dev)
    best, history, best_state = float('inf'), [], None
    for epoch in range(1, c['train']['epochs'] + 1):
        model.train()
        total, count = 0., 0
        for target_batch, cond_batch in loader:
            target_batch, cond_batch = target_batch.to(dev), cond_batch.to(dev)
            t = torch.randint(0, c['model']['timesteps'], (len(target_batch),), device=dev)
            noise = torch.randn_like(target_batch)
            noisy = schedule.add_noise(target_batch, noise, t)
            loss = torch.nn.functional.mse_loss(model(noisy, t, cond_batch), noise)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['train']['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(target_batch)
            count += len(target_batch)
        model.eval()
        with torch.inference_mode():
            vloss = torch.nn.functional.mse_loss(model(schedule.add_noise(vx, val_noise, val_t), val_t, vc), val_noise).item()
        if not np.isfinite(vloss):
            raise ValueError('Non-finite validation loss')
        history.append({'epoch': epoch, 'train_loss': total / count, 'val_loss': vloss})
        if vloss < best:
            best, best_epoch = vloss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f'{direction} uncond={unconditional} epoch={epoch} train={total/count:.6f} val={vloss:.6f}', flush=True)
    if best_state is None:
        raise ValueError('epochs must be positive')
    path = checkpoint_path(c, direction, unconditional)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'model_config': c['model'], 'scale': scale,
                'direction': direction, 'unconditional': unconditional,
                'feature_signature': signature, 'val_loss': best, 'epoch': best_epoch}, path)
    logs = Path(c['output_dir']) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs / f'{path.stem}_loss.csv', index=False)
    dump(logs / f'{path.stem}_run.json', runtime(c))
    print(f'Saved {path}, best epoch={best_epoch}, val={best:.6f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--direction', choices=['img2txt', 'txt2img', 'both'], default='both')
    p.add_argument('--unconditional', action='store_true')
    a = p.parse_args()
    c = config(a.config)
    for d in (['img2txt', 'txt2img'] if a.direction == 'both' else [a.direction]):
        train(c, d, a.unconditional)
