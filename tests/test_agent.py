import pickle

import numpy as np
import torch

from doomagent.agent import StudentChooser, load_agent
from doomagent.model import StudentCNN


def test_student_agent_round_trip(tmp_path):
    torch.manual_seed(0)
    m = StudentCNN()
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    agent = load_agent(str(tmp_path / 'm.pt'))
    obs = np.random.default_rng(0).integers(0, 255, (4, 120, 160), dtype=np.uint8)
    assert agent.act(obs) == m.act(obs)
    assert agent.explain(obs)['action'] == agent.act(obs)


def test_student_chooser_pickles_and_acts_like_the_model():
    torch.manual_seed(0)
    m = StudentCNN()
    chooser = pickle.loads(pickle.dumps(StudentChooser(m.state_dict())))
    rng = np.random.default_rng(1)
    for _ in range(5):
        obs = rng.integers(0, 255, (4, 120, 160), dtype=np.uint8)
        assert chooser(obs, None) == m.act(obs)
