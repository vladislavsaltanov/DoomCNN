import math

import torch
import torch.nn.functional as F


def _uniform(n, lo, hi, dev):
    return torch.empty(n, 1, 1, 1, device=dev).uniform_(lo, hi)


def random_texture(b, h, w, dev):
    # low-frequency noise mixed with oriented stripes, values in [0, 1]
    cells = int(torch.randint(3, 13, ()).item())
    noise = F.interpolate(torch.rand(b, 1, cells, cells, device=dev), size=(h, w),
                          mode='bilinear', align_corners=False)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=dev),
                            torch.linspace(0, 1, w, device=dev), indexing='ij')
    theta, freq = _uniform(b, 0, math.pi, dev), _uniform(b, 2, 20, dev)
    stripes = 0.5 + 0.5 * torch.sin(2 * math.pi * freq * (xx * torch.cos(theta)
                                                         + yy * torch.sin(theta)))
    mix = _uniform(b, 0, 1, dev)
    return mix * noise + (1 - mix) * stripes


def _vshift(t, dy):
    # shift the content down by dy rows (up if negative), replicating the edge row
    pad = abs(dy)
    p = F.pad(t, (0, 0, pad, pad), mode='replicate')
    start = pad - dy
    return p[:, :, start:start + t.shape[2]]


def augment(obs, mask, p_bg=0.8, p_enemy=0.5, global_aug=True, max_shift=6):
    # obs: (B, S, H, W) float in [0, 1]; mask: (B, S, H, W) bool from the labels buffer.
    # No horizontal shift: it would move the crosshair relative to the enemy and break the labels.
    B, S, H, W = obs.shape
    dev = obs.device
    x = obs
    if p_bg > 0:
        tex = random_texture(B, H, W, dev)  # one texture for the whole stack
        alpha = _uniform(B, 0.3, 0.8, dev)
        use = torch.rand(B, 1, 1, 1, device=dev) < p_bg
        x = torch.where(use & ~mask, alpha * x + (1 - alpha) * tex, x)
    if p_enemy > 0:
        gain, bias = _uniform(B, 0.6, 1.4, dev), _uniform(B, -0.2, 0.2, dev)
        use = torch.rand(B, 1, 1, 1, device=dev) < p_enemy
        x = torch.where(use & mask, x * gain + bias, x)
    if global_aug:
        x = x.clamp(0, 1) ** torch.exp(_uniform(B, -0.4, 0.4, dev))
        x = (x - 0.5) * _uniform(B, 0.7, 1.3, dev) + 0.5
        x = x + torch.randn_like(x) * _uniform(B, 0.0, 0.03, dev)
    if max_shift > 0:
        dy = int(torch.randint(-max_shift, max_shift + 1, ()).item())
        if dy:
            x = _vshift(x, dy)
            mask = _vshift(mask.float(), dy) > 0.5
    return x.clamp(0, 1), mask
