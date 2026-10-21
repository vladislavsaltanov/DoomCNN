import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from doomagent.config import N_ACTIONS, SHOOT, Cfg


class StudentCNN(nn.Module):
    def __init__(self, cfg: Cfg = Cfg()):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cfg.stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        with torch.no_grad():
            f = self.body(torch.zeros(1, cfg.stack, cfg.height, cfg.width))
        self.seg_hw = tuple(f.shape[2:])
        self.fc = nn.Sequential(nn.Linear(f[0].numel(), 512), nn.ReLU())
        self.pi = nn.Linear(512, N_ACTIONS)
        self.vis = nn.Linear(512, 1)
        self.off = nn.Linear(512, 1)
        self.seg = nn.Conv2d(64, 1, 1)  # enemy mask on the last feature map

    def forward(self, obs):
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs
        f = self.body(x)
        h = self.fc(f.flatten(1))
        return {'logits': self.pi(h), 'vis': self.vis(h).squeeze(1),
                'off': self.off(h).squeeze(1), 'seg': self.seg(f).squeeze(1)}

    def _batch(self, obs):
        return torch.from_numpy(np.asarray(obs)).unsqueeze(0).to(next(self.parameters()).device)

    @torch.no_grad()
    def act(self, obs):
        # the only thing an agent ever sees: a stack of pixels
        self.eval()
        return int(self(self._batch(obs))['logits'].argmax(1).item())

    @torch.no_grad()
    def explain(self, obs):
        # act() plus what the heads think, for the demo overlay
        self.eval()
        out = self(self._batch(obs))
        probs = torch.softmax(out['logits'], 1)[0].cpu().numpy()
        return dict(action=int(probs.argmax()), probs=probs,
                    vis=float(torch.sigmoid(out['vis'])[0]),
                    off=float(out['off'][0].clamp(-1, 1)),
                    seg=torch.sigmoid(out['seg'])[0].cpu().numpy())


def seg_target(mask_last, hw):
    # (B, H, W) bool enemy mask -> (B, h, w) float: a cell is positive if any pixel is
    return F.adaptive_max_pool2d(mask_last.float().unsqueeze(1), hw).squeeze(1)


def bc_loss(out, batch, shoot_weight=1.0, lam=0.5, lam_seg=0.5):
    a = batch['action']
    cw = torch.ones(N_ACTIONS, device=a.device)
    cw[SHOOT] = shoot_weight
    w = cw[a] * batch['valid'].float()
    ce = (F.cross_entropy(out['logits'], a, reduction='none') * w).sum() / w.sum().clamp_min(1e-8)
    vis = batch['vis'].float()
    bce = F.binary_cross_entropy_with_logits(out['vis'], vis)
    m = vis.bool()
    off = batch['off'].float()
    mse = ((out['off'] - off) ** 2)[m].mean() if m.any() else out['off'].sum() * 0.0
    if lam_seg > 0:
        seg = F.binary_cross_entropy_with_logits(out['seg'], batch['seg'])
    else:
        seg = out['seg'].sum() * 0.0
    loss = ce + lam * (bce + mse) + lam_seg * seg
    return loss, dict(ce=ce.item(), bce=bce.item(), mse=mse.item(), seg=seg.item())
