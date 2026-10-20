import numpy as np
import torch

from doomagent.agent import load_agent
from doomagent.model import StudentCNN


def test_student_agent_round_trip(tmp_path):
    torch.manual_seed(0)
    m = StudentCNN()
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    agent = load_agent(str(tmp_path / 'm.pt'))
    obs = np.random.default_rng(0).integers(0, 255, (4, 120, 160), dtype=np.uint8)
    assert agent.act(obs) == m.act(obs)
    assert agent.explain(obs)['action'] == agent.act(obs)
