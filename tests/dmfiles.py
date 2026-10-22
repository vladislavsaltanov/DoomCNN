import numpy as np

from doomagent.dm.config import DM
from doomagent.dm.env import DMObs, DMPrivileged
from doomagent.env import Label

BOT_LABEL = Label(1, 7, 'DoomPlayer', DM.width / 2 - 20.0, 40.0, 40.0, 40.0, 500.0, 0.0)  # centered, 500 units east


def row(px=0.0, py=0.0, ammo=15.0, frags=0.0, deaths=0.0, first=False, bot=False, health=100.0,
        items=0.0, hits=0.0, done=False):
    # one scripted step for FakeMatchEnv: the state the agent sees before acting
    return dict(px=px, py=py, ammo=ammo, frags=frags, deaths=deaths, first=first, bot=bot,
                health=health, items=items, hits=hits, done=done)


class FakeMatchEnv:
    # plays scripted rows without ViZDoom: rows[t] is the state before action t; a match has one
    # step per row, and stepping into a row with done=True ends it early. Frame t is filled with t.
    def __init__(self, rows, player_frags=(0.0,) * 8):
        self.rows, self.player_frags, self.t, self.actions = rows, tuple(player_frags), 0, []

    def _priv(self):
        r = self.rows[self.t]
        return DMPrivileged(labels=[BOT_LABEL] if r['bot'] else [], px=r['px'], py=r['py'],
                            health=r['health'], ammo=r['ammo'], attack_ready=True, frags=r['frags'],
                            deaths=r['deaths'], hits=r['hits'], items=r['items'],
                            players=len(self.player_frags), player_frags=self.player_frags,
                            tic=4 * self.t)

    def _obs(self):
        r = self.rows[self.t]
        return DMObs(np.full((DM.stack, DM.height, DM.width), self.t % 256, np.uint8),
                     np.array([r['health'] / 100.0, r['ammo'] / 50.0], np.float32), bool(r['first']))

    def reset(self, seed):
        self.t, self.actions = 0, []
        return self._obs()

    def step(self, action):
        self.actions.append(int(action))
        prev = self.rows[self.t]['frags']
        if self.t + 1 < len(self.rows):
            self.t += 1
            done = self.rows[self.t]['done']
        else:
            done = True
        return self._obs(), float(self.rows[self.t]['frags'] - prev), bool(done)

    def privileged(self):
        return self._priv()

    def screen(self):
        return np.zeros((480, 640, 3), np.uint8)

    def close(self):
        pass
