import inspect

import numpy as np
import pytest
import torch

from doomagent.dm.agent import DMAgent
from doomagent.dm.brain import render, trace
from doomagent.dm.config import DM
from doomagent.dm.env import DMPrivileged
from doomagent.dm.memory_view import MemoryTape, events_from
from doomagent.dm.model import DMNet
from doomagent.env import Label


def inputs(seed=0):
    g = np.random.default_rng(seed)
    return g.integers(0, 255, (4, DM.height, DM.width), dtype=np.uint8), g.random(2).astype(np.float32)


def bot_label():
    return Label(1, 3, 'DoomPlayer', 100.0, 50.0, 20.0, 30.0, 400.0, 0.0)


def test_m0_contributions_sum_to_the_logit():
    m = DMNet(DM, memory=False).eval()
    t = trace(m, *inputs())
    a = t['action']
    assert np.isclose(t['contrib'][a].sum() + t['vec_contrib'][a].sum() + t['bias'][a], t['logits'][a], atol=1e-4)


def test_m0_vec_contrib_is_the_linear_term():
    m = DMNet(DM, memory=False).eval()
    f, v = inputs()
    w = m.head.weight.detach().numpy()[:, 512:]
    assert np.allclose(trace(m, f, v)['vec_contrib'], w * v[None, :], atol=1e-5)


def test_m1_contributions_sum_to_the_logit():
    m = DMNet(DM, memory=True).eval()
    t = trace(m, *inputs(), h=torch.randn(1, 1, 512))
    a = t['action']
    assert np.isclose(t['contrib'][a].sum() + t['bias'][a], t['logits'][a], atol=1e-4)


def test_trace_leaves_h_unchanged():
    m = DMNet(DM, memory=True).eval()
    h = torch.randn(1, 1, 512)
    before = h.clone()
    trace(m, *inputs(), h=h)
    assert torch.equal(h, before)


@pytest.mark.parametrize('memory', [False, True])
def test_render_shape(memory):
    m = DMNet(DM, memory=memory).eval()
    f, v = inputs()
    img = render(trace(m, f, v), f, v)
    assert img.shape == (480, 1280, 3) and img.dtype == np.uint8


def test_trace_takes_only_model_inputs():
    assert list(inspect.signature(trace).parameters) == ['model', 'frames', 'vec', 'h']


def test_tape_uses_the_same_step_hidden():
    a = DMAgent(DMNet(DM, memory=True).eval())
    tape = MemoryTape()
    for i in range(3):
        info = a.explain(*inputs(i), i == 0)
        tape.push(info['h'], info['z'])
    assert np.allclose(tape.H[:, -1], info['h'])


def test_tape_keeps_the_last_20_seconds_and_places_events():
    tape = MemoryTape(steps=175)
    for i in range(200):
        tape.push(np.zeros(512), np.zeros(512), {'death'} if i == 190 else ())
    assert tape.H.shape == (512, 175) and tape.E[-10] == {'death'}
    assert tape.render().shape == (420, 900, 3)


def test_events_from():
    p0 = DMPrivileged(labels=[], ammo=5.0)
    p1 = DMPrivileged(labels=[bot_label()], ammo=15.0)
    assert events_from(p0, p1, False, False) == {'bot_seen', 'pickup'}
    assert events_from(p1, p0, True, False) == {'bot_lost', 'death'}
    assert events_from(None, p0, True, True) == set()
