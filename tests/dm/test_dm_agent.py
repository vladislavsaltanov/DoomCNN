import inspect
from dataclasses import astuple

import numpy as np

from dmfiles import Poison
from doomagent.dm.agent import DMAgent, DMChooser, agent_chooser
from doomagent.dm.config import DM
from doomagent.dm.env import DMObs
from doomagent.dm.model import DMNet


def ob(seed=0, first=False):
    g = np.random.default_rng(seed)
    return DMObs(g.integers(0, 255, (4, DM.height, DM.width), dtype=np.uint8),
                 g.random(2).astype(np.float32), first)


def test_agent_takes_pixels_vec_and_first_only():
    assert list(inspect.signature(DMAgent.act).parameters) == ['self', 'frames', 'vec', 'first']
    assert list(inspect.signature(DMAgent.explain).parameters) == ['self', 'frames', 'vec', 'first']
    m = DMNet(DM, memory=True)
    DMChooser(m.state_dict(), True)(ob(first=True), Poison())
    agent_chooser(DMAgent(m))(ob(first=True), Poison())


def test_first_resets_the_memory():
    m = DMNet(DM, memory=True).eval()
    a = DMAgent(m)
    for i in range(5):
        a.act(*astuple(ob(i, first=i == 0)))
    fresh = DMAgent(m).explain(*astuple(ob(9, first=True)))
    again = a.explain(*astuple(ob(9, first=True)))
    assert np.allclose(fresh['h'], again['h']) and fresh['action'] == again['action']


def test_chooser_resets_between_matches():
    m = DMNet(DM, memory=True)
    c = DMChooser(m.state_dict(), True)
    seq = [ob(i, first=i == 0) for i in range(8)]
    first_run = [c(o, None) for o in seq]
    c.reset()
    assert [c(o, None) for o in seq] == first_run == [DMChooser(m.state_dict(), True)(o, None) for o in seq]


def test_reset_every_step_equals_first_every_step():
    m = DMNet(DM, memory=True).eval()
    a, b = DMAgent(m, reset_every_step=True), DMAgent(m)
    for i in range(6):
        o = ob(i)
        assert a.act(o.frames, o.vec, False) == b.act(o.frames, o.vec, True)


def test_explain_chains_the_hidden_state():
    a = DMAgent(DMNet(DM, memory=True).eval())
    e1 = a.explain(*astuple(ob(0, first=True)))
    e2 = a.explain(*astuple(ob(1)))
    assert np.allclose(e2['h_prev'], e1['h']) and e1['z'].shape == (512,)


def test_chooser_trace_rows():
    m = DMNet(DM, memory=True)
    c = DMChooser(m.state_dict(), True, keep_trace=True)
    for i in range(5):
        c(ob(i, first=i == 0), None)
    t = c.take_trace()
    assert t['feat'].shape == (5, 512) and t['h'].shape == (5, 512) and c.take_trace()['feat'].shape[0] == 0
    assert c.steps == 0 or c.steps == 5
