import math
from collections import deque
from dataclasses import replace

import numpy as np
import pytest
import torch

from dmfiles import FakeMatchEnv, StubTeacher, row  # tests/dm is on sys.path
from doomagent.dm.actions import FORWARD, VEC_DIM
from doomagent.dm.agent import DMAgent
from doomagent.dm.config import DM
from doomagent.dm.env import DMPrivileged
from doomagent.dm.evaluate import stuck_steps
from doomagent.dm.model import DMNet, split_logits
from doomagent.dm.ppo import (REWARD, Collector, Critic, PPOCfg, gae, local_envs, prior_kl, schedule,
                              stack_index, update)
from doomagent.dm.rlenv import (CELL, LINGER_STEPS, N_PARTS, PARTS, PRIV_DIM, REVISIT_STEPS, Runner, critic_vec,
                                explore_step, step_parts, stuck_now)
from doomagent.variants import BASE


def test_features_seq_zeroes_the_memory_at_life_starts_like_step_by_step():
    torch.manual_seed(0)
    m = DMNet(DM, memory=True).eval()
    B, T = 2, 5
    frames = torch.randint(0, 255, (B, T, DM.stack, DM.height, DM.width), dtype=torch.uint8)
    vec, h0 = torch.rand(B, T, VEC_DIM), torch.randn(1, B, 512)
    first = torch.zeros(B, T, dtype=torch.bool)
    first[0, 3] = first[1, 0] = True
    with torch.no_grad():
        z, h = m.trunk(frames, vec, h0, first)
        hh, outs = h0.clone(), []
        for t in range(T):
            zt, hh = m.trunk(frames[:, t:t + 1], vec[:, t:t + 1], hh * (~first[:, t]).float()[None, :, None])
            outs.append(zt)
    assert torch.allclose(z, torch.cat(outs, 1), atol=1e-5) and torch.allclose(h, hh, atol=1e-5)


def test_policy_logits_equal_the_agent_that_plays_evaluations():
    torch.manual_seed(1)
    m = DMNet(DM, memory=True).eval()
    T = 4
    frames = torch.randint(0, 255, (1, T, DM.stack, DM.height, DM.width), dtype=torch.uint8)
    vec = torch.rand(1, T, VEC_DIM)
    first = torch.tensor([[True, False, True, False]])
    agent = DMAgent(m)
    with torch.no_grad():
        logits = m.head(m.trunk(frames, vec, None, first)[0])[0]
        for t in range(T):
            want = tuple(int(lg.argmax()) for lg in split_logits(logits[t]))
            assert agent.act(frames[0, t].numpy(), vec[0, t].numpy(), bool(first[0, t])) == want


def test_gae_does_not_carry_value_across_a_match_end():
    adv, ret = gae(torch.tensor([[1.0, 0.0, 2.0]]), torch.tensor([[0.5, 0.4, 0.3]]), torch.tensor([[0.0, 1.0, 0.0]]),
                   torch.tensor([1.0]), 0.9, 0.8)
    assert torch.allclose(adv, torch.tensor([[0.572, -0.4, 2.6]])) and torch.allclose(ret, adv + torch.tensor([[0.5, 0.4, 0.3]]))


def test_stack_index_follows_the_dmobs_rule_across_the_rollout_start():
    S, T = DM.stack, 9
    first = np.zeros(T, bool)
    first[[2, 3]] = True
    stack = deque([-3, -2, -1, 0], maxlen=S)  # frame ids; the stack of step 0 came from the rollout before
    buf, want = list(stack), []
    for t in range(T):
        if t:
            if first[t]:
                stack.extend([t] * S)
            else:
                stack.append(t)
            buf.append(t)
        want.append(list(stack))
    got = np.array(buf)[stack_index(first, S)]
    assert (got == np.array(want)).all()


def priv(**kw):
    base = dict(frags=0.0, deaths=0.0, damage=0.0, health=100.0, armor=0.0, ammo=20.0, weapon=2,
                owned=(0, 1, 1, 0, 0, 0, 0, 0), player_frags=(0.0,) * 8)
    return DMPrivileged(**dict(base, **kw))


def test_step_parts():
    p = priv()
    assert step_parts(p, priv(frags=1.0, damage=40.0), False, 1.0).tolist() == [1, 0, 40, 0, 0, 0, 0, 0, 0]
    assert step_parts(p, priv(health=125.0), False, 0.0)[PARTS.index('item')] == 1
    assert step_parts(p, priv(health=125.0, deaths=1.0), True, 0.0)[[1, 3]].tolist() == [1, 0]  # a respawn: no pickup
    got = step_parts(p, priv(weapon=4, owned=(0, 1, 1, 0, 1, 0, 0, 0), ammo=40.0), False, 0.0)
    assert got[PARTS.index('weapon')] == 1 and got[PARTS.index('item')] == 0  # a switch is not an ammo pickup
    assert step_parts(p, priv(frags=-1.0), False, -1.0)[0] == -1


def test_stuck_now_counts_the_steps_the_evaluator_counts():
    rng = np.random.default_rng(0)
    T = 400
    pos = np.cumsum(rng.choice([0.0, 0.5, 6.0], size=(T, 2), p=[0.5, 0.3, 0.2]), 0)
    enemy, first = rng.random(T) < 0.03, rng.random(T) < 0.01
    hist, n = deque(maxlen=18), 0
    for t in range(T):
        hist.append((pos[t, 0], pos[t, 1], bool(enemy[t] or first[t])))
        n += stuck_now(hist)
    assert n == stuck_steps([tuple(x) for x in pos], list(enemy), list(first)) > 0


def test_explore_rewards_a_cell_left_long_ago_and_not_staying_in_it():
    visits = {}
    assert explore_step(visits, 10.0, 10.0, 0) == 1.0
    assert explore_step(visits, 20.0, 10.0, 1) == 0.0  # the same cell
    assert explore_step(visits, CELL + 1.0, 10.0, 2) == 1.0
    assert explore_step(visits, 10.0, 10.0, 3) == 0.0  # back after 3 steps
    assert explore_step(visits, 10.0, 10.0, 3 + REVISIT_STEPS + 1) == 1.0  # back after 30 s


def test_runner_explore_and_linger_parts():
    # 60 steps in one spot (lingering after 5 s, stuck after 2 s), then a run east through new cells
    rows = [row(px=0.0 if t < 60 else 200.0 * (t - 59), first=t == 0) for t in range(80)]
    r = Runner(DM, [BASE], (7,), 0, env_factory=lambda cfg, v: FakeMatchEnv(rows), teacher=StubTeacher((0, 0, 0, 0)))
    r.start()
    parts = np.stack([r.step((0, 0, 0, 0))['parts'] for _ in range(79)])
    lin, exp, stk = (parts[:, PARTS.index(k)] for k in ('linger', 'explore', 'stuck'))
    assert lin[:LINGER_STEPS - 1].sum() == 0 and lin[LINGER_STEPS - 1:59].all() and not lin[60:].any()
    assert stk[16:59].all() and exp[:59].sum() == 0 and exp[59:].all()


def test_territory_counts_holding_a_patch_even_with_bots_in_sight():
    # 150 steps strafing within 100 units with a bot always in sight, then a long run east
    rows = [row(px=50.0 * (t % 3) if t < 150 else 300.0 * (t - 149), bot=True, first=t == 0) for t in range(180)]
    r = Runner(DM, [BASE], (7,), 0, env_factory=lambda cfg, v: FakeMatchEnv(rows), teacher=StubTeacher((0, 0, 0, 0)))
    r.start()
    parts = np.stack([r.step((0, 0, 0, 0))['parts'] for _ in range(179)])
    ter, lin = parts[:, PARTS.index('territory')], parts[:, PARTS.index('linger')]
    assert not ter[:87].any() and ter[87:149].all() and not ter[155:].any() and not lin.any()


def test_critic_vec():
    v = critic_vec(priv(frags=3.0, player_frags=(3.0, 5.0) + (0.0,) * 6), 'chase', 525, True)
    assert v.shape == (PRIV_DIM,) and np.isfinite(v).all() and v[0] == 0.5 and v[3] == pytest.approx(-0.2)


def fake_rows(T=40):
    # a death at step 10 (frag before it), one more frag at 20, the match ends at T
    return [row(px=float(t), first=t in (0, 10), deaths=float(t >= 10), frags=float(t >= 5) + float(t >= 20))
            for t in range(T)]


def test_runner_records_and_match_numbers():
    r = Runner(DM, [BASE], (7,), 0, env_factory=lambda cfg, v: FakeMatchEnv(fake_rows()),
               teacher=StubTeacher((FORWARD, 0, 1, 0)))
    rec = r.start()
    assert rec['first'] and rec['frame'].shape == (DM.height, DM.width) and tuple(rec['label']) == (FORWARD, 0, 1, 0)
    assert rec['critic'].shape == (PRIV_DIM,) and not rec['map_ok'] and not rec['depth_ok']
    recs = [r.step((0, 0, 0, 0)) for _ in range(40)]
    end = recs[-1]
    assert end['done'] and end['first'] and end['match']['frags'] == 2.0 and end['match']['length'] == 40
    assert end['match']['parts']['frag'] == 2.0 and end['match']['parts']['death'] == 1.0
    assert sum(x['done'] for x in recs) == 1 and recs[9]['first']


def test_collect_and_update_train_the_policy_and_frozen_trains_the_critic_only():
    torch.manual_seed(0)
    cfg = replace(DM, seg_len=8)
    envs = local_envs(2, cfg, [BASE], (7,), env_factory=lambda c, v: FakeMatchEnv(fake_rows(30)),
                      teacher_factory=lambda: StubTeacher((FORWARD, 0, 1, 0)))
    model = DMNet(cfg, memory=True, aux=True, spatial=True)
    critic = Critic(model.head.in_features)
    pc = replace(PPOCfg(), horizon=16, mb_windows=2, epochs=1, lr=1e-3)
    col = Collector(envs, 2, cfg, 'cpu', True)
    roll = col.collect(model, critic, pc.horizon, REWARD)
    assert roll['fbuf'].shape[:2] == (2, cfg.stack - 1 + 16) and roll['h0'].shape == (2, 2, 512)
    assert roll['parts'].shape == (2, 16, N_PARTS) and np.isfinite(roll['reward']).all()
    opt_a, opt_c = torch.optim.Adam(model.parameters(), 1e-3), torch.optim.Adam(critic.parameters(), 1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = update(model, critic, opt_a, opt_c, roll, pc, 1.0, frozen=True, cfg=cfg)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()) and math.isfinite(out['v_loss'])
    roll = col.collect(model, critic, pc.horizon, REWARD)
    assert len(col.matches) == 2 and col.matches[0]['frags'] == 2.0
    prior = DMNet(cfg, memory=True, aux=True, spatial=True).eval()
    out = update(model, critic, opt_a, opt_c, roll, pc, 1.0, frozen=False, cfg=cfg, prior=prior, prior_w=0.1)
    assert not torch.equal(before['head.weight'], model.state_dict()['head.weight'])
    assert all(math.isfinite(out[k]) for k in ('pg', 'ks', 'aux', 'spatial', 'kl', 'agree_move', 'prior_kl'))


def test_schedule_warms_up_then_anneals():
    pc = PPOCfg()
    assert schedule(pc, 0) == (pc.lr, pc.ks, pc.prior)
    lr, ks, pw = schedule(pc, pc.steps)
    assert lr == pytest.approx(pc.lr * pc.lr_end) and ks == pytest.approx(pc.ks_end) and pw == pytest.approx(pc.prior_end)


def test_prior_kl_is_zero_at_the_start_and_positive_after():
    lg = torch.randn(3, 5, 28)
    assert prior_kl(lg, lg, 0.5).abs().max() < 1e-6 and (prior_kl(lg + torch.randn(3, 5, 28), lg) > 0).all()
