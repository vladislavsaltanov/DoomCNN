import numpy as np
import torch

from doomagent.dm.config import DM
from doomagent.dm.model import DMNet, gru_gates, load_model


def pick_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class DMAgent:
    # what an agent sees: the frame stack, [health, rockets] and whether a life just started
    def __init__(self, model, reset_every_step=False):
        self.model, self.reset_every_step = model.eval(), reset_every_step
        self.dev = next(model.parameters()).device
        self.reset()

    @classmethod
    def from_path(cls, path, cfg=DM, **kw):
        dev = pick_device()
        return cls(load_model(path, cfg, map_location=dev).to(dev), **kw)

    def reset(self):
        self.h = None

    @torch.no_grad()
    def _step(self, frames, vec, first):
        if first or self.reset_every_step:
            self.h = None  # a new life starts with an empty memory
        f = torch.from_numpy(np.ascontiguousarray(frames))[None, None].to(self.dev)
        v = torch.as_tensor(np.asarray(vec, np.float32))[None, None].to(self.dev)
        m = self.model
        feat, hid, h_out = m.features(f, v, self.h)
        z = None
        if m.memory:
            prev = self.h[0] if self.h is not None else torch.zeros(1, 512, device=self.dev)
            _, z, _, _ = gru_gates(m.gru, torch.cat([feat[:, 0], v[:, 0]], -1), prev)
            logits = m.head(hid)[0, 0]
        else:
            prev = None
            logits = m.head(torch.cat([feat, v], -1))[0, 0]
        self.h = h_out
        return logits, feat[0, 0], hid, z, prev

    def act(self, frames, vec, first):
        return int(self._step(frames, vec, first)[0].argmax())

    def explain(self, frames, vec, first):
        logits, feat, hid, z, prev = self._step(frames, vec, first)
        probs = torch.softmax(logits, 0).cpu().numpy()
        cpu = lambda t: None if t is None else t.reshape(-1).cpu().numpy()  # noqa: E731
        return dict(action=int(probs.argmax()), probs=probs, feat=cpu(feat), h=cpu(hid), h_prev=cpu(prev),
                    z=cpu(z))


class DMChooser:
    # a picklable chooser(obs, priv) for worker processes: the weights travel, the model is built in
    # the worker on first use; reads obs.frames, obs.vec and obs.first only
    def __init__(self, state, memory, cfg=DM, reset_every_step=False, keep_trace=False):
        self.state = {k: v.detach().cpu() for k, v in state.items()}
        self.memory, self.cfg, self.reset_every_step, self.keep_trace = memory, cfg, reset_every_step, keep_trace
        self.agent, self.steps, self._feat, self._h = None, 0, [], []

    def __getstate__(self):
        return dict(self.__dict__, agent=None)

    def reset(self):
        if self.agent is not None:
            self.agent.reset()
        self.steps = 0

    def __call__(self, obs, priv):
        if self.agent is None:
            dev = pick_device()
            m = DMNet(self.cfg, self.memory)
            m.load_state_dict(self.state)
            self.agent = DMAgent(m.to(dev), self.reset_every_step)
        self.steps += 1
        if not self.keep_trace:
            return self.agent.act(obs.frames, obs.vec, obs.first)
        info = self.agent.explain(obs.frames, obs.vec, obs.first)
        self._feat.append(info['feat'].astype(np.float16))
        if info['h'] is not None:
            self._h.append(info['h'].astype(np.float16))
        return info['action']

    def take_trace(self):
        out = dict(feat=np.array(self._feat, np.float16).reshape(-1, 512),
                   h=np.array(self._h, np.float16).reshape(-1, 512) if self.memory else None)
        self._feat, self._h = [], []
        return out


def agent_chooser(agent):
    class Chooser:
        def reset(self):
            agent.reset()

        def __call__(self, obs, priv):
            return agent.act(obs.frames, obs.vec, obs.first)

    return Chooser()
