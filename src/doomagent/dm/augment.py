import torch
import torch.nn.functional as F


def _uniform(n, lo, hi, dev):
    return torch.empty(n, 1, 1, 1, device=dev).uniform_(lo, hi)


def frame_augment(obs, max_shift=6):
    # whole-frame changes only: gamma, contrast, noise, a vertical shift (no horizontal shift: it
    # would move the crosshair against the target). obs: (B, C, H, W) float in [0, 1]; one set of
    # parameters per sample, so a stack or a whole M1 window (flattened into C) changes together.
    B, dev = obs.shape[0], obs.device
    x = obs.clamp(0, 1) ** torch.exp(_uniform(B, -0.4, 0.4, dev))
    x = (x - 0.5) * _uniform(B, 0.7, 1.3, dev) + 0.5
    x = x + torch.randn_like(x) * _uniform(B, 0.0, 0.03, dev)
    if max_shift > 0:
        dy = int(torch.randint(-max_shift, max_shift + 1, ()).item())
        if dy:
            pad = abs(dy)
            p = F.pad(x, (0, 0, pad, pad), mode='replicate')
            x = p[:, :, pad - dy:pad - dy + x.shape[2]]
    return x.clamp(0, 1)
