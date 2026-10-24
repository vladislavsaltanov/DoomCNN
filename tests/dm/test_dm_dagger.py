import numpy as np
import pytest
import torch

from doomagent.dm.actions import VEC_DIM
from doomagent.dm.agent import DMChooser
from doomagent.dm.config import DM
from doomagent.dm.dagger import DAggerExecutor, stop_rule
from doomagent.dm.env import DMObs
from doomagent.dm.model import DMNet, init_from

LABEL = (1, 7, 1, 0)


def ob(seed=0, first=False):
    g = np.random.default_rng(seed)
    return DMObs(g.integers(0, 255, (4, DM.height, DM.width), dtype=np.uint8), g.random(VEC_DIM).astype(np.float32),
                 first)


def test_executor_beta_zero_is_the_student():
    m = DMNet(DM, memory=True)
    ex, ch = DAggerExecutor(m.state_dict(), True, 0.0), DMChooser(m.state_dict(), True)
    ex.reset(1)
    for i in range(6):
        o = ob(i, first=i == 0)
        assert ex(o, None, LABEL) == ch(o, None)
    assert not ex.in_burst


def test_executor_beta_one_executes_labels_but_the_student_still_runs():
    ex = DAggerExecutor(DMNet(DM, memory=True).state_dict(), True, 1.0)
    ex.reset(1)
    assert all(ex(ob(i, first=i == 0), None, LABEL) == LABEL for i in range(4))
    assert ex.chooser.steps == 4


def test_executor_reset_repeats():
    ex = DAggerExecutor(DMNet(DM, memory=True).state_dict(), True, 0.5)
    runs = []
    for _ in range(2):
        ex.reset(3)
        runs.append([ex(ob(i, first=i == 0), None, LABEL) for i in range(10)])
    assert runs[0] == runs[1] and LABEL in runs[0]


@pytest.mark.parametrize('scores, stop, best', [([10, 12, 12.2, 12.3], True, 3), ([10, 13], False, 1),
                                                ([10, 13, 16, 18], True, 3), ([10, 9, 9.5], True, 0)])
def test_stop_rule(scores, stop, best):
    assert stop_rule(scores, teacher_frags=10.0) == (stop, best)


def test_init_from_a_model_without_spatial_heads_keeps_its_weights():
    src = DMNet(DM, memory=True, aux=True)
    dst = DMNet(DM, memory=True, aux=True, spatial=True)
    init_from(dst, src.state_dict())
    assert torch.equal(dst.head.weight, src.head.weight) and torch.equal(dst.aux.weight, src.aux.weight)
    with pytest.raises(RuntimeError):
        init_from(DMNet(DM, memory=False), src.state_dict())  # the GRU and head shapes differ


def test_dagger_plan_plays_the_base_map_half_the_time(tmp_path):
    from doomagent.dm.config import dagger_plan
    import json
    from doomagent.variants import BASE
    for i in range(16):
        (tmp_path / f'train_{i}.json').write_text(json.dumps(dict(name=f'tex_train_{i:02d}', split='train', kind='tex')))
    plan = dagger_plan(1, root=str(tmp_path))
    base = sum(len(s) for v, s in plan if v == BASE)
    assert sum(len(s) for _, s in plan) == 40 and 12 <= base <= 28
    assert len({v.name for v, _ in plan}) == len(plan)  # one group per variant
