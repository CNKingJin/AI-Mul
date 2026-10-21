"""Feature-space epsilon predictor and official diffusers DDIM scheduler."""
import math
import torch
from torch import nn
from diffusers import DDIMScheduler


class Denoiser(nn.Module):
    def __init__(self, dim=512, hidden=512, time_dim=128, **_):
        super().__init__()
        if time_dim % 2:
            raise ValueError('time_dim must be even')
        self.time_dim = time_dim
        self.net = nn.Sequential(nn.Linear(2 * dim + time_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x, t, condition):
        freq = torch.exp(-math.log(10000) * torch.arange(
            self.time_dim // 2, device=x.device) / max(self.time_dim // 2 - 1, 1))
        angles = t.float()[:, None] * freq[None]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        return self.net(torch.cat([x, condition, embedding], dim=1))


def scheduler(model_config):
    return DDIMScheduler(num_train_timesteps=model_config['timesteps'],
                         beta_schedule=model_config['beta_schedule'],
                         prediction_type='epsilon', clip_sample=False,
                         timestep_spacing='trailing', set_alpha_to_one=True)


@torch.inference_mode()
def generate(model, schedule, condition, scale, steps, seed, unconditional=False, batch_size=256):
    model.eval()
    dev = condition.device
    generator = torch.Generator(device=dev).manual_seed(seed)
    # Draw all noise at once so chunk size does not change the starting noise.
    noise = torch.randn(condition.shape, generator=generator, device=dev)
    schedule.set_timesteps(steps, device=dev)
    outputs = []
    for start in range(0, len(condition), batch_size):
        c = condition[start:start + batch_size] * scale
        if unconditional:
            c = torch.zeros_like(c)  # identical capacity; only condition information removed
        x = noise[start:start + batch_size].clone()
        for t in schedule.timesteps:
            ts = torch.full((len(x),), int(t), device=dev, dtype=torch.long)
            prediction = model(x, ts, c)
            x = schedule.step(prediction, t, x, eta=0.0).prev_sample
        outputs.append(torch.nn.functional.normalize(x / scale, dim=1))
    if not outputs:
        return condition.clone()
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite generated features')
    return result
