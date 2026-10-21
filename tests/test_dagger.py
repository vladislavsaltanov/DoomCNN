import numpy as np
import pytest
import torch

from doomagent.config import SHOOT, TURN_LEFT
from doomagent.dagger import student_executor


class FixedAgent:
    def act(self, obs):
        return TURN_LEFT


def test_executor_plays_the_student_or_the_label():
    rng = np.random.default_rng(0)
    assert all(student_executor(FixedAgent(), 0.0, rng)(None, None, SHOOT) == TURN_LEFT
               for _ in range(20))
    assert all(student_executor(FixedAgent(), 1.0, rng)(None, None, SHOOT) == SHOOT
               for _ in range(20))


@pytest.mark.vizdoom
def test_dagger_writes_teacher_labelled_episode(tmp_path):
    from doomagent.dagger import run_round
    from doomagent.model import StudentCNN
    from doomagent.variants import BASE
    torch.save(StudentCNN().state_dict(), tmp_path / 'm.pt')
    run_round(str(tmp_path / 'm.pt'), str(tmp_path / 'out'), [(BASE, (2000,))])
    z = np.load(tmp_path / 'out' / 'ep_base_2000.npz')
    assert len(z['actions']) > 0
    assert (tmp_path / 'out' / 'meta.json').exists()
