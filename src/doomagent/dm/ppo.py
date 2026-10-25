"""E6: recurrent PPO on DMNet with the teacher's hints (docs/E6_PPO_проект.md).

- The policy is DMNet itself: four independent heads (the action's log-probability is their sum); M1's GRU state is
  kept per env, zeroed at life starts, stored at the start of every training window (seg_len steps), and zeroed
  inside windows at life starts too (DMNet.features_seq).
- An asymmetric critic: an MLP on the policy's features without gradient into them, the vector and the privileged
  vector of dm.rlenv.critic_vec. It exists only in training.
- An anchor to the start: KL(policy || start policy) on the rollout states, weight annealed from prior to prior_end
  (fine-tuning after imitation as in AlphaStar or RLHF). Kickstarting, cross-entropy to the shadow teacher's label of
  every step (ks), is there but off: with weight 1 it broke the policy in 250k steps (val 3.57 -> 1.06, stuck 42 %);
  the teacher's labels on the student's states depend on its path, which the picture does not show.
- The auxiliary and spatial heads keep training on the rollout steps: dm.rlenv gives their targets in the
  recorder's format, so dm.data._aux, dm.spatial.spatial_targets and the losses of dm.model are reused.
- Reward: frags plus small shaping parts (dm.rlenv.PARTS) weighted by REWARD; the evaluation counts frags only.
- The first evaluation runs before any update and the critic warms up with the policy frozen, so the best
  checkpoint is never below the start (lessons of the first section's S7).

Run through scripts/dm/dm_ppo.py: env processes re-import the main module, and that one does not import torch.
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from doomagent.dm.actions import HEADS, VEC_DIM
from doomagent.dm.agent import DMChooser, pick_device
from doomagent.dm.config import DM, RL_SEEDS, VARIANTS_DIR, val_plan
from doomagent.dm.data import _aux
from doomagent.dm.evaluate import evaluate_plan, summarize
from doomagent.dm.model import DMNet, bc_loss, fut_input, load_model, save_model, split_logits
from doomagent.dm.rlenv import N_PARTS, PARTS, PRIV_DIM, Runner
from doomagent.dm.spatial import spatial_targets
from doomagent.evaluate import val_score
from doomagent.variants import BASE, load_variants

REWARD = dict(frag=1.0, death=-0.2, damage=0.005, item=0.02, weapon=0.1, stuck=-0.01, explore=0.0, linger=0.0,
              territory=0.0, blind_fire=0.0, region=0.0, region_stay=0.0)


@dataclass(frozen=True)
class PPOCfg:
    steps: int = 3_000_000       # env steps in all, the warm-up included
    n_envs: int = 8
    horizon: int = 256           # steps per env per rollout: n_envs * horizon per update
    gamma: float = 0.995         # a step is 4 tics; a frag pays off seconds after the decisions behind it
    lam: float = 0.95
    clip: float = 0.1
    epochs: int = 3
    mb_windows: int = 16         # windows of seg_len steps per minibatch
    lr: float = 2e-5
    lr_end: float = 0.3          # the policy's lr falls linearly to this share of lr by the end
    critic_lr: float = 3e-4
    ent: float = 0.001           # on the sum of the heads' entropies
    prior: float = 0.1           # weight of KL(policy || start policy) after the warm-up, falling linearly ...
    prior_end: float = 0.02      # ... to this by the end
    ks: float = 0.0              # kickstarting weight after the warm-up (0: off) ...
    ks_end: float = 0.0          # ... falling linearly to this ...
    ks_steps: int = 2_000_000    # ... over these steps
    aux_w: float = 1.0           # the BC weights of the auxiliary and spatial losses
    spatial_w: float = 1.0
    warmup: int = 50_000         # steps that train the critic only
    temp: float = 1.0            # the policy is softmax(logits / temp)
    max_grad: float = 0.5
    target_kl: float = 0.03      # an update stops its epochs past this KL
    base_share: float = 0.5      # matches on the base map; the rest on the train variants
    eval_every: int = 250_000
    ckpt_every: int = 100_000
    latest_every: int = 25_000   # runs/<out>/latest.pt this often: the dashboard's live game can play the network now
    eval_workers: int = 4
    live: bool = True            # the training games on the dashboard (runs/dm/live)
    warp_p: float = 0.0          # chance that a match or a life starts at a random walkable spot (dm.rlenv.Runner)
    demo_dirs: tuple = ()        # the teacher's recorded matches mixed into every update as BC (RLPD / DQfD style) ...
    demo: float = 0.1            # ... with this weight, falling linearly ...
    demo_end: float = 0.02       # ... to this by the end
    demo_windows: int = 8        # windows of seg_len steps per update minibatch
    seed: int = 0


class Critic(nn.Module):
    # V(s) from what the policy's heads read (detached: the value loss never moves the policy's features), the
    # agent's vector and the privileged vector
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + VEC_DIM + PRIV_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, z, vec, priv):
        return self.net(torch.cat([z.detach(), vec, priv], -1)).squeeze(-1)


def head_dists(logits, temp=1.0):
    return [torch.distributions.Categorical(logits=lg / temp) for lg in split_logits(logits)]


def log_prob_entropy(logits, actions, temp=1.0):
    # log-probability of the (..., 4) actions (sum over the heads) and the entropy of each head (..., 4)
    ds = head_dists(logits, temp)
    lp = sum(d.log_prob(actions[..., k]) for k, d in enumerate(ds))
    return lp, torch.stack([d.entropy() for d in ds], -1)


def gae(rewards, values, dones, last_value, gamma, lam):
    # (N, T) tensors; dones[:, t] = 1 when the match ended after step t (no value is carried across);
    # last_value (N,) is the value of the observation after the last step. Advantages and returns.
    adv = torch.zeros_like(rewards)
    run = torch.zeros_like(last_value)
    T = rewards.shape[1]
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[:, t + 1]
        go = 1.0 - dones[:, t]
        run = rewards[:, t] + gamma * nxt * go - values[:, t] + gamma * lam * go * run
        adv[:, t] = run
    return adv, adv + values


def stack_index(first, stack):
    # (T, stack) positions of every step's frame stack, oldest..newest, in a rollout's frame buffer: the buffer holds
    # the stack - 1 frames before the rollout, then frame t at stack - 1 + t; a life start pads the stack with its
    # first frame (the DMObs rule)
    T = len(first)
    idx, start = np.empty((T, stack), np.int64), None
    for t in range(T):
        if first[t]:
            start = t
        for j in range(stack):
            u = t - (stack - 1 - j)
            idx[t, j] = stack - 1 + (max(u, start) if start is not None else u)
    return idx


class _Z(dict):
    # a dict that answers .files like a loaded npz, for dm.data._aux and dm.spatial.spatial_targets
    @property
    def files(self):
        return list(self)


def step_targets(rec):
    # per-env (T, ...) recorded arrays of a rollout -> the auxiliary and spatial targets in the training batch format
    z = _Z(pose=rec['pose'], first=rec['first'], actions=rec['label'], executed=rec['action'], modes=rec['mode'],
           bot_visible=rec['bot_visible'], enemy_x=rec['enemy_x'], items=rec['items'])
    if rec['depth_ok'].all():
        z['depth'] = rec['depth']
    if rec['map_ok'].all():
        z['egomap'], z['goal'] = rec['egomap'], rec['goal']
    a, sp = _aux(z), spatial_targets(z)
    out = {k: a[k] for k in ('aux_enemy', 'aux_x', 'aux_pos', 'aux_head', 'aux_items')}
    out['aux_depth'] = a['aux_depth'].astype(np.float32) / 255.0
    out.update({k: sp[k] for k in ('sp_map_ok', 'sp_odo', 'sp_odo_ok', 'sp_fut', 'sp_fut_ok', 'sp_fut_move',
                                   'sp_fut_turn', 'sp_goal_dir', 'sp_goal_dist', 'sp_mode')})
    out['sp_map'] = sp['sp_map'].astype(np.float32)
    out['sp_fut_depth'] = out['aux_depth'][sp['sp_fut_idx']]
    out['sp_fut_depth_ok'] = sp['sp_fut_ok'] & a['depth_ok']
    return out, bool(a['items_ok']), bool(a['depth_ok'])


class ProcEnvs:
    # n Runner processes (dm.rlenv.worker); start() and step(actions) give one record per env. A process that
    # fails is started again with a new match (its record says done, the step's reward parts are lost).
    def __init__(self, n, cfg, variants, seeds, seed, base_share, live=None, warp_p=0.0):
        import multiprocessing as mp
        self.ctx, self.args = mp.get_context('spawn'), (cfg, list(variants), list(seeds), base_share)
        self.live, self.warp_p = live, warp_p
        self.seed, self.restarts = seed, 0
        self.procs, self.conns = [None] * n, [None] * n
        for i in range(n):
            self._spawn(i)

    def _spawn(self, i):
        from doomagent.dm.rlenv import worker
        cfg, variants, seeds, base_share = self.args
        a, b = self.ctx.Pipe()
        p = self.ctx.Process(target=worker, args=(b, cfg, variants, seeds, 1000 * self.seed + i + 97 * self.restarts,
                                                  base_share, self.live, f'ppo{i + 1}', self.warp_p), daemon=True)
        p.start()
        self.procs[i], self.conns[i] = p, a

    def _ask(self, msgs):
        for c, m in zip(self.conns, msgs):
            c.send(m)
        out = []
        for i, c in enumerate(self.conns):
            try:
                out.append(c.recv())
            except (EOFError, ConnectionError, OSError):
                print(f'env {i} failed, restarting it', flush=True)
                self.procs[i].kill()
                self.restarts += 1
                self._spawn(i)
                self.conns[i].send(('start', None))
                rec = self.conns[i].recv()
                rec['done'] = True
                out.append(rec)
        return out

    def start(self):
        return self._ask([('start', None)] * len(self.conns))

    def step(self, actions):
        return self._ask([('step', tuple(int(x) for x in a)) for a in actions])

    def close(self):
        for c, p in zip(self.conns, self.procs):
            try:
                c.send(('close', None))
            except (OSError, BrokenPipeError):
                pass
            p.join(timeout=10)
            if p.is_alive():
                p.kill()


class LocalEnvs:
    # the same interface with Runners in this process (tests, debugging)
    def __init__(self, runners):
        self.runners = runners

    def start(self):
        return [r.start() for r in self.runners]

    def step(self, actions):
        return [r.step(a) for r, a in zip(self.runners, actions)]

    def close(self):
        for r in self.runners:
            r.close()


REC_KEYS = ('vec', 'first', 'label', 'mode', 'pose', 'bot_visible', 'enemy_x', 'items', 'depth', 'depth_ok', 'egomap',
            'map_ok', 'goal', 'critic')


class Collector:
    # plays rollouts: the policy acts on every env at once on the device; keeps every env's frame stack and GRU
    # state across rollouts
    def __init__(self, envs, n, cfg, dev, memory, temp=1.0):
        self.envs, self.n, self.cfg, self.dev, self.memory, self.temp = envs, n, cfg, dev, memory, temp
        self.recs = envs.start()
        f = self._frames(self.recs)
        self.stack = f[:, None].repeat(1, cfg.stack, 1, 1)
        self.h = None
        self.matches = []

    def _frames(self, recs):
        return torch.from_numpy(np.stack([r['frame'] for r in recs])).to(self.dev)

    def _push(self, recs):
        f = self._frames(recs)
        first = torch.tensor([r['first'] for r in recs], device=self.dev)
        rolled = torch.cat([self.stack[:, 1:], f[:, None]], 1)
        self.stack = torch.where(first[:, None, None, None], f[:, None].expand_as(self.stack), rolled)

    def _inputs(self, recs):
        t = lambda k, dt: torch.from_numpy(np.stack([r[k] for r in recs]).astype(dt)).to(self.dev)  # noqa: E731
        return t('vec', np.float32), t('first', bool), t('critic', np.float32)

    @torch.no_grad()
    def _policy(self, model, critic, recs, h):
        vec, first, priv = self._inputs(recs)
        z, h_out = model.trunk(self.stack[:, None], vec[:, None], h, first[:, None])
        logits = model.head(z[:, 0])
        return logits, critic(z[:, 0], vec, priv), h_out

    @torch.no_grad()
    def collect(self, model, critic, T, weights):
        N, S, dev = self.n, self.cfg.stack, self.dev
        model.eval()
        fbuf = torch.empty((N, S - 1 + T) + tuple(self.stack.shape[-2:]), dtype=torch.uint8, device=dev)
        fbuf[:, :S] = self.stack
        rec = {k: [] for k in REC_KEYS + ('action', 'logp', 'value', 'reward', 'done', 'parts')}
        h0 = []
        w = np.array([weights[k] for k in PARTS], np.float32)
        for t in range(T):
            if t:
                fbuf[:, S - 1 + t] = self.stack[:, -1]
            if self.memory and t % self.cfg.seg_len == 0:  # the state before the window, zeroed at a life start
                h0.append(self.h[0].clone() if self.h is not None else torch.zeros(N, model.gru.hidden_size, device=dev))
            for k in REC_KEYS:
                rec[k].append(np.stack([r[k] for r in self.recs]))
            logits, value, self.h = self._policy(model, critic, self.recs, self.h)
            action = torch.stack([d.sample() for d in head_dists(logits, self.temp)], -1)
            lp, _ = log_prob_entropy(logits, action, self.temp)
            a = action.cpu().numpy()
            nxt = self.envs.step(a)
            parts = np.stack([r['parts'] for r in nxt])
            rec['action'].append(a)
            rec['logp'].append(lp.cpu().numpy())
            rec['value'].append(value.cpu().numpy())
            rec['parts'].append(parts)
            rec['reward'].append(parts @ w)
            rec['done'].append(np.array([r['done'] for r in nxt], np.float32))
            self.matches += [r['match'] for r in nxt if r['match'] is not None]
            self._push(nxt)
            self.recs = nxt
        _, last_value, _ = self._policy(model, critic, self.recs, self.h)
        out = {k: np.stack(v, 1) for k, v in rec.items()}  # (N, T, ...)
        out['fbuf'], out['last_value'] = fbuf, last_value
        out['h0'] = torch.stack(h0, 1) if self.memory else None  # (N, K, 512)
        out['fidx'] = np.stack([stack_index(out['first'][e], S) for e in range(N)])
        model.train()
        return out


def batch_tensors(roll, dev, L):
    # the rollout's steps as windows of L steps: {key: (N * T / L, L, ...)} tensors on the device, the targets of
    # the auxiliary and spatial heads included; per-window 'env' and 'aux_*_ok' flags
    N, T = roll['first'].shape
    K = T // L
    per_env = [step_targets({k: roll[k][e] for k in roll if isinstance(roll[k], np.ndarray)}) for e in range(N)]
    tg = {k: np.stack([p[0][k] for p in per_env]) for k in per_env[0][0]}
    win = lambda x: x.reshape(N * K, L, *x.shape[2:])  # noqa: E731
    out = {k: win(torch.as_tensor(v).to(dev)) for k, v in tg.items()}
    for k, dt in (('vec', torch.float32), ('first', torch.bool), ('critic', torch.float32), ('label', torch.long),
                  ('action', torch.long), ('logp', torch.float32), ('fidx', torch.long)):
        out[k] = win(torch.as_tensor(roll[k]).to(dev, dt))
    out['env'] = torch.arange(N, device=dev).repeat_interleave(K)
    out['aux_items_ok'] = torch.tensor([p[1] for p in per_env], device=dev).repeat_interleave(K)
    out['aux_depth_ok'] = torch.tensor([p[2] for p in per_env], device=dev).repeat_interleave(K)
    if roll['h0'] is not None:
        out['h0'] = roll['h0'].reshape(N * K, -1)
    return out


def minibatch(b, idx, fbuf):
    mb = {k: v[idx] for k, v in b.items()}
    mb['frames'] = fbuf[mb['env'][:, None, None], mb['fidx']]  # (B, L, stack, H, W)
    return mb


def prior_kl(logits, prior_logits, temp=1.0):
    # KL(policy || start policy) per step, summed over the heads
    kl = 0.0
    for lg, lp in zip(split_logits(logits / temp), split_logits(prior_logits / temp)):
        a, b = torch.log_softmax(lg, -1), torch.log_softmax(lp, -1)
        kl = kl + (a.exp() * (a - b)).sum(-1)
    return kl


def demo_batches(dirs, cfg, n, dev):
    # an endless stream of the teacher's M1 windows (the BC batch format of dm.train): the PPO update's demonstrations
    from torch.utils.data import DataLoader
    from doomagent.dm.data import MatchSegments, match_files
    ds = MatchSegments(match_files(dirs), cfg.seg_len, cfg.burn_in, cfg.stack, compact=True)
    dl = DataLoader(ds, batch_size=n, shuffle=True, num_workers=2, pin_memory=dev == 'cuda', persistent_workers=True,
                    drop_last=True)
    while True:
        yield from dl


def demo_loss(model, demo, temp, dev):
    # BC on a batch of the teacher's own states, mirrored half of the time like in dm.train (cross-entropy of the
    # policy softmax(logits / temp) to the teacher's labels, burn-in steps masked)
    from doomagent.dm.data import mirror_batch
    from doomagent.dm.train import _device, _to_model
    b = _device(next(demo), dev)
    if torch.rand(()) < 0.5:
        b = mirror_batch(b)
    f, v, a, m = _to_model(b, model.memory, dev)
    logits, _ = model(f, v)
    return bc_loss(logits / temp, a, m)


def update(model, critic, opt_a, opt_c, roll, pc, ks_w, frozen, cfg=DM, extra_losses=None, prior=None, prior_w=0.0,
           demo=None, demo_w=0.0):
    # one PPO update on a rollout; frozen: the critic only (warm-up); prior: the frozen start model, pulled towards
    # with weight prior_w. Returns mean numbers for the log.
    from doomagent.dm.train import _extra_losses
    extra_losses = extra_losses or _extra_losses
    dev = roll['fbuf'].device
    L = cfg.seg_len
    rew, val = torch.as_tensor(roll['reward'], device=dev), torch.as_tensor(roll['value'], device=dev)
    done = torch.as_tensor(roll['done'], device=dev)
    adv, ret = gae(rew, val, done, roll['last_value'], pc.gamma, pc.lam)
    b = batch_tensors(roll, dev, L)
    win = lambda x: x.reshape(-1, L)  # noqa: E731
    b['adv'], b['ret'] = win(adv), win(ret)
    n = b['adv'].shape[0]
    sums, count, stop = {}, 0, False
    for _ in range(pc.epochs):
        kls = []
        for idx in torch.randperm(n, device=dev).split(pc.mb_windows):
            mb = minibatch(b, idx, roll['fbuf'])
            h0 = mb['h0'][None].contiguous() if 'h0' in mb else None
            mask = torch.ones_like(mb['first'])
            with torch.set_grad_enabled(not frozen):
                z, _ = model.trunk(mb['frames'], mb['vec'], h0, mb['first'])
            v = critic(z, mb['vec'], mb['critic'])
            v_loss = 0.5 * ((v - mb['ret']) ** 2).mean()
            opt_c.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), pc.max_grad)
            opt_c.step()
            parts = dict(v_loss=v_loss.item())
            if not frozen:
                fut = fut_input(mb['sp_fut_move'], mb['sp_fut_turn']) if model.spatial_on else None
                logits, out = model.outputs(z, aux=True, fut=fut)
                lp, ent = log_prob_entropy(logits, mb['action'], pc.temp)
                ratio = torch.exp(lp - mb['logp'])
                a = mb['adv']
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - pc.clip, 1 + pc.clip) * a).mean()
                ks, kp = bc_loss(logits / pc.temp, mb['label'], mask)
                ex, ep = extra_losses(model, out, mb, mask, dev, pc.aux_w, pc.spatial_w)
                loss = pg - pc.ent * ent.sum(-1).mean() + ks_w * ks + ex
                if prior is not None:
                    with torch.no_grad():
                        zp, _ = prior.trunk(mb['frames'], mb['vec'], h0, mb['first'])
                        pl = prior.head(zp)
                    pk = prior_kl(logits, pl, pc.temp).mean()
                    loss = loss + prior_w * pk
                    parts['prior_kl'] = pk.item()
                if demo is not None and demo_w > 0:
                    dl, dp = demo_loss(model, demo, pc.temp, dev)
                    loss = loss + demo_w * dl
                    parts.update(demo_ce=dp['ce'], demo_acc=dp['acc'])
                opt_a.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), pc.max_grad)
                opt_a.step()
                with torch.no_grad():
                    lr_ = lp - mb['logp']
                    kl = ((torch.exp(lr_) - 1) - lr_).mean().item()
                kls.append(kl)
                parts.update(pg=pg.item(), kl=kl, clipfrac=((ratio - 1).abs() > pc.clip).float().mean().item(),
                             ks=kp['ce'], agree=kp['acc'], aux=ep.get('aux', 0.0), spatial=ep.get('spatial', 0.0),
                             **{f'agree_{h}': kp[f'acc_{h}'] for h in HEADS},
                             **{f'ent_{h}': ent[..., i].mean().item() for i, h in enumerate(HEADS)})
            for k, x in parts.items():
                sums[k] = sums.get(k, 0.0) + x
            count += 1
        if kls and np.mean(kls) > pc.target_kl:
            stop = True
            break
    out = {k: x / max(count, 1) for k, x in sums.items()}
    with torch.no_grad():
        out['explained_var'] = float(1 - torch.var(ret - val) / (torch.var(ret) + 1e-8))
    out['early_stop'] = stop
    return out


def schedule(pc, step):
    # (policy lr, kickstarting weight, prior weight) at this env step; all start after the warm-up
    p = min(max(step - pc.warmup, 0) / max(pc.steps - pc.warmup, 1), 1.0)
    k = min(max(step - pc.warmup, 0) / max(pc.ks_steps, 1), 1.0)
    return (pc.lr * (1 - (1 - pc.lr_end) * p), pc.ks + (pc.ks_end - pc.ks) * k, pc.prior + (pc.prior_end - pc.prior) * p,
            pc.demo + (pc.demo_end - pc.demo) * p)


def evaluate(model, plan, cfg, workers):
    # the policy by argmax on the plan, through the same evaluate_plan as E4-E5
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    res = evaluate_plan(plan, DMChooser(state, model.memory, cfg), cfg, workers=workers)
    score, per = val_score(res)
    s = summarize(res)
    return dict(val_score=score, per_variant=per, frags=s['frags'], kd_total=s['kd_total'],
                stuck_share=s['stuck_share']['mean'], first_share=s['first_share'])


def match_numbers(matches):
    # means over the training matches finished since the last row
    if not matches:
        return {}
    out = dict(matches=len(matches))
    for k in ('frags', 'deaths', 'best_bot', 'stuck_share'):
        out['m_' + k] = float(np.mean([m[k] for m in matches]))
    for k in PARTS:
        out['m_part_' + k] = float(np.mean([m['parts'][k] for m in matches]))
    return out


def _atomic_save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(out, init, pc=PPOCfg(), reward=None, cfg=DM, plan=None, envs=None, variants=None):
    # PPO from init (a save_model .pt; None: a fresh M1 with every head, PPO from scratch) into folder out;
    # restarts from out/state.pt if it is there. plan: the evaluation plan (val_plan; [] for none)
    reward = dict(REWARD, **(reward or {}))
    cfg = replace(cfg, depth=True)  # the coarse depth target
    os.makedirs(out, exist_ok=True)
    dev = pick_device()
    torch.manual_seed(pc.seed)
    torch.backends.cudnn.benchmark = True
    model = (load_model(init, cfg) if init else DMNet(cfg, memory=True, aux=True, spatial=True)).to(dev)
    critic = Critic(model.head.in_features).to(dev)
    prior = None
    if init and pc.prior > 0:
        prior = load_model(init, cfg).to(dev).eval()
        prior.requires_grad_(False)
    opt_a = torch.optim.Adam(model.parameters(), lr=pc.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=pc.critic_lr)
    state_path = os.path.join(out, 'state.pt')
    step, best, last_eval, last_ckpt = 0, -math.inf, None, 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev, weights_only=True)
        model.load_state_dict(st['model'])
        critic.load_state_dict(st['critic'])
        opt_a.load_state_dict(st['opt_a'])
        opt_c.load_state_dict(st['opt_c'])
        step, best, last_eval = st['step'], st['best'], st['last_eval']
        last_ckpt = step
        print(f'resumed at step {step}', flush=True)
    with open(os.path.join(out, 'config.json'), 'w') as fh:
        json.dump(dict(init=init, ppo=asdict(pc), reward=reward, parts=PARTS, memory=model.memory,
                       aux=model.aux_on, spatial=model.spatial_on), fh, indent=1)
    plan = val_plan() if plan is None else plan
    log_path = os.path.join(out, 'log.jsonl')

    def log(row):
        print(json.dumps(row), flush=True)
        with open(log_path, 'a') as fh:
            fh.write(json.dumps(row) + '\n')

    def save_state():
        _atomic_save(dict(model=model.state_dict(), critic=critic.state_dict(), opt_a=opt_a.state_dict(),
                          opt_c=opt_c.state_dict(), step=step, best=best, last_eval=last_eval), state_path)

    def eval_now():
        nonlocal best, last_eval
        last_eval = step
        if not plan:
            return
        torch.cuda.empty_cache()
        t0 = time.time()
        ecfg = replace(cfg, depth=False, cheats=False)  # evaluation as always: no depth buffer, no teleports
        row = dict(kind='eval', step=step, **evaluate(model, plan, ecfg, pc.eval_workers),
                   minutes=(time.time() - t0) / 60)
        save_model(model, os.path.join(out, f'step_{step // 1000:05d}k.pt'))
        if row['val_score'] > best:
            best = row['val_score']
            save_model(model, os.path.join(out, 'best.pt'))
        row['best'] = best
        log(row)

    if last_eval is None:  # before any update: the start's level, best.pt can never be below it
        eval_now()
        save_state()
    variants = [BASE] + load_variants('train', VARIANTS_DIR) if variants is None else variants
    live = None
    if pc.live and envs is None:
        from doomagent.dm.livetap import LIVE_DIR, write_plan
        from doomagent.dm.rlenv import match_steps
        live = LIVE_DIR
        write_plan(f'{os.path.basename(os.path.normpath(out))} · обучение PPO', out,
                   (pc.steps - step) // match_steps(cfg), kind='ppo')
    envs = envs or ProcEnvs(pc.n_envs, replace(cfg, cheats=pc.warp_p > 0), variants, RL_SEEDS, pc.seed + step,
                            pc.base_share, live, pc.warp_p)
    col = Collector(envs, pc.n_envs, cfg, dev, model.memory, pc.temp)
    demo = demo_batches(list(pc.demo_dirs), cfg, pc.demo_windows, dev) if pc.demo_dirs else None
    t_start, s_start = time.time(), step
    try:
        while step < pc.steps:
            lr, ks_w, prior_w, demo_w = schedule(pc, step)
            for g in opt_a.param_groups:
                g['lr'] = lr
            frozen = step < pc.warmup
            t0 = time.time()
            roll = col.collect(model, critic, pc.horizon, reward)
            t1 = time.time()
            u = update(model, critic, opt_a, opt_c, roll, pc, ks_w, frozen, cfg, prior=prior, prior_w=prior_w, demo=demo,
                       demo_w=demo_w)
            step += pc.n_envs * pc.horizon
            parts = roll['parts'].reshape(-1, N_PARTS).mean(0)
            row = dict(kind='update', step=step, phase='warmup' if frozen else 'ppo', lr=lr, ks_w=ks_w, prior_w=prior_w,
                       demo_w=demo_w if demo is not None else 0.0,
                       collect_s=t1 - t0, update_s=time.time() - t1,
                       fps=(step - s_start) / (time.time() - t_start), reward=float(roll['reward'].mean()),
                       **{'step_' + k: float(x) for k, x in zip(PARTS, parts)}, **u, **match_numbers(col.matches))
            col.matches = []
            log(row)
            if step - last_eval >= pc.eval_every or step >= pc.steps:
                eval_now()
            if step // pc.latest_every != (step - pc.n_envs * pc.horizon) // pc.latest_every:
                save_model(model, os.path.join(out, 'latest.pt'))
            if step - last_ckpt >= pc.ckpt_every or step >= pc.steps:
                save_state()
                last_ckpt = step
    finally:
        envs.close()
    return os.path.join(out, 'best.pt')


def local_envs(n, cfg, variants, seeds, env_factory=None, teacher_factory=None, base_share=0.5):
    # Runners in this process (tests): env_factory(cfg, variant), teacher_factory() for each
    return LocalEnvs([Runner(cfg, variants, seeds, i, base_share, env_factory,
                             teacher_factory() if teacher_factory else None) for i in range(n)])
