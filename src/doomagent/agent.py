import numpy as np
import torch

from doomagent.config import Cfg
from doomagent.model import StudentCNN


class StudentAgent:
    def __init__(self, path, cfg: Cfg = Cfg()):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = StudentCNN(cfg).to(dev)
        self.model.load_state_dict(torch.load(path, map_location=dev))
        self.model.eval()

    def act(self, obs):
        return self.model.act(obs)

    def explain(self, obs):
        return self.model.explain(obs)


class PPOAgent:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path)

    def act(self, obs):
        return int(self.model.predict(obs, deterministic=True)[0])

    @torch.no_grad()
    def explain(self, obs):
        t = self.model.policy.obs_to_tensor(np.asarray(obs))[0]
        probs = self.model.policy.get_distribution(t).distribution.probs[0].cpu().numpy()
        return dict(action=int(probs.argmax()), probs=probs, vis=None, off=None, seg=None)


def load_agent(path, cfg: Cfg = Cfg()):
    return PPOAgent(path) if str(path).endswith('.zip') else StudentAgent(path, cfg)
