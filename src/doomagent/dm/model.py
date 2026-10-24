import torch
import torch.nn as nn
import torch.nn.functional as F

from doomagent.dm.actions import HEAD_SIZES, HEADS, N_LOGITS, VEC_DIM
from doomagent.dm.auxtargets import AUX_SIZES
from doomagent.dm.config import DM
from doomagent.dm.spatial import FUT_IN, FUT_OUT, SPATIAL_SIZES


class DMNet(nn.Module):
    # M0 (memory=False): conv features + the vector (dm.actions.make_vec) -> one linear layer of N_LOGITS, read as
    # four heads (split_logits). M1 (memory=True): the same features + vector go through a GRU of 512 first.
    def __init__(self, cfg=DM, memory=False, aux=False, spatial=False):
        super().__init__()
        self.memory, self.aux_on, self.spatial_on = memory, aux, spatial
        self.body = nn.Sequential(
            nn.Conv2d(cfg.stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU())
        with torch.no_grad():
            flat = self.body(torch.zeros(1, cfg.stack, cfg.height, cfg.width)).numel()
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU())
        if memory:
            self.gru = nn.GRU(512 + VEC_DIM, 512, batch_first=True)
            self.head = nn.Linear(512, N_LOGITS)
        else:
            self.head = nn.Linear(512 + VEC_DIM, N_LOGITS)
        if aux:  # auxiliary heads (dm.auxtargets) read what the action heads read
            self.aux = nn.Linear(self.head.in_features, sum(AUX_SIZES.values()))
        if spatial:  # spatial heads (dm.spatial), and a forward model: features + the next moves -> motion, depth
            z = self.head.in_features
            self.spatial = nn.Linear(z, sum(SPATIAL_SIZES.values()))
            self.fwd = nn.Sequential(nn.Linear(z + FUT_IN, 256), nn.ReLU(), nn.Linear(256, FUT_OUT))

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

    def features_seq(self, frames, vec, first, h=None):
        # features() over a window that may cross life starts (PPO rollouts): the memory is zeroed before every
        # step with first (B, T), as DMAgent does step by step; h is the state before the window
        if not self.memory:
            return self.features(frames, vec)
        B, T = frames.shape[:2]
        feat = self.encode(frames.reshape(B * T, *frames.shape[2:])).view(B, T, -1)
        x = torch.cat([feat, vec.to(feat.dtype)], -1)
        h = x.new_zeros(1, B, self.gru.hidden_size) if h is None else h
        keep = (~first.bool()).to(x.dtype).T[:, None, :, None]  # (T, 1, B, 1)
        if not first[:, 1:].any():  # a life start only before the first step: one call
            hid, h_out = self.gru(x, h * keep[0])
            return feat, hid, h_out
        outs = []
        for t in range(T):
            o, h = self.gru(x[:, t:t + 1], h * keep[t])
            outs.append(o)
        return feat, torch.cat(outs, 1), h

    def trunk(self, frames, vec, h=None, first=None):
        # what the heads read (the GRU output, or the features and the vector) and the GRU state after the window
        feat, hid, h_out = self.features(frames, vec, h) if first is None else self.features_seq(frames, vec, first, h)
        return (hid if self.memory else torch.cat([feat, vec.to(feat.dtype)], -1)), h_out

    def outputs(self, z, aux=True, fut=None):
        # the action logits and {target: logits} of the auxiliary and spatial heads (None without them)
        out = {}
        if aux and self.aux_on:
            out.update(zip(AUX_SIZES, torch.split(self.aux(z), list(AUX_SIZES.values()), dim=-1)))
        if aux and self.spatial_on:
            out.update(zip(SPATIAL_SIZES, torch.split(self.spatial(z), list(SPATIAL_SIZES.values()), dim=-1)))
            if fut is not None:
                out['fwd'] = self.fwd(torch.cat([z, fut.to(z.dtype)], -1))
        return self.head(z), out or None

    def forward(self, frames, vec, h=None):
        logits, h_out, _ = self.forward_aux(frames, vec, h, aux=False)
        return logits, h_out

    def forward_aux(self, frames, vec, h=None, aux=True, fut=None):
        # the action logits, the GRU state and {target: logits} of the auxiliary and spatial heads (None without
        # them); fut (B, T, FUT_IN): the moves and turns played next (fut_input), for the forward model 'fwd'
        z, h_out = self.trunk(frames, vec, h)
        logits, out = self.outputs(z, aux, fut)
        return logits, h_out, out


def split_logits(logits):
    # (..., N_LOGITS) -> [move, turn, fire, weapon] logits
    return list(torch.split(logits, HEAD_SIZES, dim=-1))


def bc_loss(logits, actions, mask, weights=(1.0, 1.0, 1.0, 1.0)):
    # sum over the heads of the cross-entropy over the steps in mask; actions (..., 4)
    m = mask.reshape(-1).float()
    n = m.sum().clamp_min(1.0)
    loss, parts, all_ok = 0.0, {}, torch.ones_like(m, dtype=torch.bool)
    for k, (name, lg) in enumerate(zip(HEADS, split_logits(logits))):
        a = actions[..., k].reshape(-1)
        ce = (F.cross_entropy(lg.reshape(-1, lg.shape[-1]), a, reduction='none') * m).sum() / n
        ok = lg.reshape(-1, lg.shape[-1]).argmax(1) == a
        all_ok &= ok
        loss = loss + weights[k] * ce
        parts[f'ce_{name}'], parts[f'acc_{name}'] = ce.item(), ((ok.float() * m).sum() / n).item()
    parts['ce'], parts['acc'] = float(loss.item()), ((all_ok.float() * m).sum() / n).item()
    return loss, parts


def aux_loss(aux, b, mask):
    # mean of the auxiliary losses over the steps in mask that have the target: enemy on screen (BCE),
    # its column, the map cell, the heading (CE; -1 = none or mirrored), items in view (BCE), coarse depth (MSE)
    m = mask.reshape(-1)
    T = mask.shape[-1] if mask.dim() > 1 else 1
    per_step = lambda ok: ok.reshape(-1, 1).expand(-1, T).reshape(-1) if ok.dim() == 1 else ok.reshape(-1)
    flat = {k: v.reshape(m.shape[0], -1) for k, v in aux.items()}
    total, parts = 0.0, {}

    def ce(name, key):
        nonlocal total
        y = b[key].reshape(-1).to(m.device)
        ok = m & (y >= 0)
        if ok.any():
            lg = flat[name][ok]
            total = total + F.cross_entropy(lg, y[ok])
            parts[key + '_acc'] = (lg.argmax(1) == y[ok]).float().mean().item()

    y = b['aux_enemy'].reshape(-1).to(m.device).float()
    if m.any():
        lg = flat['enemy'][:, 0][m]
        total = total + F.binary_cross_entropy_with_logits(lg, y[m])
        parts['aux_enemy_acc'] = ((lg > 0).float() == (y[m] > 0.5).float()).float().mean().item()
    ce('enemy_x', 'aux_x')
    ce('pos', 'aux_pos')
    ce('heading', 'aux_head')
    ok = m & per_step(b['aux_items_ok'].to(m.device))
    if ok.any():
        total = total + F.binary_cross_entropy_with_logits(flat['items'][ok], b['aux_items'].reshape(m.shape[0], -1).to(m.device)[ok])
    ok = m & per_step(b['aux_depth_ok'].to(m.device))
    if ok.any():
        mse = F.mse_loss(torch.sigmoid(flat['depth'][ok]), b['aux_depth'].reshape(m.shape[0], -1).to(m.device)[ok])
        total = total + mse
        parts['aux_depth_mse'] = mse.item()
    parts['aux'] = float(total.detach()) if torch.is_tensor(total) else 0.0
    return total, parts


def fut_input(move, turn):
    # (..., FUT_K) moves and turns played next -> (..., FUT_IN) one-hot; padding (-1) gives zeros
    def one_hot(v, n):
        return (F.one_hot(v.clamp_min(0), n) * (v >= 0)[..., None]).flatten(-2)
    return torch.cat([one_hot(move, HEAD_SIZES[0]), one_hot(turn, HEAD_SIZES[1])], -1).float()


def spatial_loss(out, b, mask):
    # sum of the spatial losses (dm.spatial) over the steps in mask that have the target: walkable cells around
    # (BCE), motion over the last steps and after the next ones (smooth L1), depth after them (MSE), the
    # teacher's goal direction and distance and its mode (CE; -1 = none)
    m = mask.reshape(-1)
    N = m.shape[0]
    flat = {k: v.reshape(N, -1) for k, v in out.items()}
    total, parts = 0.0, {}

    def y(key):
        return b[key].to(m.device).reshape(N, -1)

    def add(loss, key, value):
        nonlocal total
        total = total + loss
        parts[key] = value

    ok = m & y('sp_map_ok')[:, 0]
    if ok.any():
        lg, t = flat['egomap'][ok], y('sp_map')[ok]
        add(F.binary_cross_entropy_with_logits(lg, t), 'sp_map_acc', ((lg > 0).float() == t).float().mean().item())
    ok = m[:, None] & y('sp_odo_ok').repeat_interleave(3, 1)
    if ok.any():
        p, t = flat['odo'][ok], y('sp_odo')[ok]
        add(F.smooth_l1_loss(p, t), 'sp_odo_err', (p - t).abs().mean().item())
    if 'fwd' in flat:
        ok = m & y('sp_fut_ok')[:, 0]
        if ok.any():
            p, t = flat['fwd'][ok, :3], y('sp_fut')[ok]
            add(F.smooth_l1_loss(p, t), 'sp_fut_err', (p - t).abs().mean().item())
        ok = m & y('sp_fut_depth_ok')[:, 0]
        if ok.any():
            mse = F.mse_loss(torch.sigmoid(flat['fwd'][ok, 3:]), y('sp_fut_depth')[ok])
            add(mse, 'sp_fut_depth_mse', mse.item())
    for name, key in (('goal_dir', 'sp_goal_dir'), ('goal_dist', 'sp_goal_dist'), ('mode', 'sp_mode')):
        t = y(key)[:, 0].long()
        ok = m & (t >= 0)
        if ok.any():
            lg = flat[name][ok]
            add(F.cross_entropy(lg, t[ok]), key + '_acc', (lg.argmax(1) == t[ok]).float().mean().item())
    parts['spatial'] = float(total.detach()) if torch.is_tensor(total) else 0.0
    return total, parts


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
    torch.save(dict(state=model.state_dict(), memory=model.memory, aux=model.aux_on, spatial=model.spatial_on), path)


def init_from(model, state):
    # the weights of state in model; heads model has and state lacks (auxiliary, spatial) keep their fresh init
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = list(unexpected) + [k for k in missing if not k.startswith(('aux.', 'spatial.', 'fwd.'))]
    if bad:
        raise RuntimeError(f'init_from: keys do not match: {bad[:6]}')
    return model


def load_model(path, cfg=DM, map_location='cpu'):
    d = torch.load(path, map_location=map_location)
    m = DMNet(cfg, memory=d['memory'], aux=d.get('aux', False), spatial=d.get('spatial', False))
    m.load_state_dict(d['state'])
    return m
