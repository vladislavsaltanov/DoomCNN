import torch
import torch.nn as nn
import torch.nn.functional as F

from doomagent.dm.config import DM, N_ACTIONS


class DMNet(nn.Module):
    # M0 (memory=False): conv features + [health, rockets] -> action head.
    # M1 (memory=True): the same features + vector go through a GRU of 512; the head reads its state.
    def __init__(self, cfg=DM, memory=False):
        super().__init__()
        self.memory = memory
        self.body = nn.Sequential(
            nn.Conv2d(cfg.stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        with torch.no_grad():
            flat = self.body(torch.zeros(1, cfg.stack, cfg.height, cfg.width)).numel()
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU())
        if memory:
            self.gru = nn.GRU(514, 512, batch_first=True)
            self.head = nn.Linear(512, N_ACTIONS)
        else:
            self.head = nn.Linear(514, N_ACTIONS)

    def encode(self, frames):
        x = frames.float() / 255.0 if frames.dtype == torch.uint8 else frames
        return self.fc(self.body(x).flatten(1))

    def features(self, frames, vec, h=None):
        # frames (B, T, stack, H, W), vec (B, T, 2); h is the GRU state before the window (zero if None)
        B, T = frames.shape[:2]
        feat = self.encode(frames.reshape(B * T, *frames.shape[2:])).view(B, T, -1)
        if not self.memory:
            return feat, None, None
        hid, h_out = self.gru(torch.cat([feat, vec.to(feat.dtype)], -1), h)
        return feat, hid, h_out

    def forward(self, frames, vec, h=None):
        feat, hid, h_out = self.features(frames, vec, h)
        return self.head(hid if self.memory else torch.cat([feat, vec.to(feat.dtype)], -1)), h_out


def bc_loss(logits, actions, mask):
    # cross-entropy over the steps in mask
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), actions.reshape(-1), reduction='none')
    m = mask.reshape(-1).float()
    n = m.sum().clamp_min(1.0)
    loss = (ce * m).sum() / n
    acc = ((logits.reshape(-1, logits.shape[-1]).argmax(1) == actions.reshape(-1)).float() * m).sum() / n
    return loss, dict(ce=loss.item(), acc=acc.item())


def load_conv(model, path):
    # the three conv layers of the first section's final model (.pt StudentCNN or PPO .zip)
    if str(path).endswith('.zip'):
        from stable_baselines3 import PPO
        from doomagent.model import StudentCNN
        from doomagent.ppo import policy_state
        state = policy_state(PPO.load(path, device='cpu'), StudentCNN().state_dict())
    else:
        state = torch.load(path, map_location='cpu')
    body = {k[len('body.'):]: v for k, v in state.items() if k.startswith('body.')}
    model.body.load_state_dict(body)
    return len(body)


def gru_gates(gru, x, h):
    # one GRU step from its own weights (PyTorch gate order r, z, n): r, z, n, new state
    gi = x @ gru.weight_ih_l0.T + gru.bias_ih_l0
    gh = h @ gru.weight_hh_l0.T + gru.bias_hh_l0
    i_r, i_z, i_n = gi.chunk(3, -1)
    h_r, h_z, h_n = gh.chunk(3, -1)
    r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return r, z, n, (1 - z) * n + z * h


def save_model(model, path):
    torch.save(dict(state=model.state_dict(), memory=model.memory), path)


def load_model(path, cfg=DM, map_location='cpu'):
    d = torch.load(path, map_location=map_location)
    m = DMNet(cfg, memory=d['memory'])
    m.load_state_dict(d['state'])
    return m
