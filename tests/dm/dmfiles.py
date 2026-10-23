import numpy as np

from doomagent.dm.actions import VEC_DIM, make_vec
from doomagent.dm.config import DM
from doomagent.dm.env import DMObs, DMPrivileged
from doomagent.env import Label

BOT_LABEL = Label(1, 7, 'DoomPlayer', DM.width / 2 - 20.0, 40.0, 40.0, 40.0, 500.0, 0.0)  # centered, 500 units east


def row(px=0.0, py=0.0, ammo=15.0, frags=0.0, deaths=0.0, first=False, bot=False, health=100.0,
        items=0.0, hits=0.0, done=False, weapon=5):
    # one scripted step for FakeMatchEnv: the state the agent sees before acting; ammo is the weapon in hand's
    return dict(px=px, py=py, ammo=ammo, frags=frags, deaths=deaths, first=first, bot=bot,
                health=health, items=items, hits=hits, done=done, weapon=weapon)


class FakeMatchEnv:
    # plays scripted rows without ViZDoom: rows[t] is the state before action t; a match has one
    # step per row, and stepping into a row with done=True ends it early. Frame t is filled with t.
    def __init__(self, rows, player_frags=(0.0,) * 8):
        self.rows, self.player_frags, self.t, self.actions = rows, tuple(player_frags), 0, []
        from doomagent.variants import BASE
        self.variant = BASE

    def _priv(self):
        r = self.rows[self.t]
        return DMPrivileged(labels=[BOT_LABEL] if r['bot'] else [], px=r['px'], py=r['py'],
                            health=r['health'], ammo=r['ammo'], attack_ready=True, frags=r['frags'],
                            deaths=r['deaths'], hits=r['hits'], items=r['items'],
                            players=len(self.player_frags), player_frags=self.player_frags,
                            tic=4 * self.t, weapon=r['weapon'], weapon_ammo=r['ammo'])

    def _obs(self):
        r = self.rows[self.t]
        return DMObs(np.full((DM.stack, DM.height, DM.width), self.t % 256, np.uint8),
                     make_vec(r['health'], 0.0, r['weapon'], r['ammo']), bool(r['first']))

    def reset(self, seed):
        self.t, self.actions = 0, []
        return self._obs()

    def step(self, action):
        self.actions.append(tuple(int(x) for x in action))
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


class StubTeacher:
    # always the same label (an action tuple); stands in for DMTeacher in recorder tests
    def __init__(self, action):
        self.action, self.mode = action, 'patrol'

    def reset(self):
        pass

    def __call__(self, obs, priv):
        return self.action


class Poison:
    # a privileged-data stand-in that fails on any read: the agent must not touch it
    def __getattr__(self, name):
        raise AssertionError(f'the agent read privileged data: {name}')


def make_match(d, seed, T=40, first_at=(0,), variant='base', aux=False):
    # a synthetic match in the recorder format; frame t is filled with t % 256; aux: with the auxiliary
    # targets (enemy column t % 9 - 1, items, depth) and a pose that walks east across the map
    import os
    os.makedirs(d, exist_ok=True)
    pre = os.path.join(str(d), f'ep_{variant}_{seed}')
    np.save(pre + '.frames.npy', np.stack([np.full((DM.height, DM.width), t % 256, np.uint8)
                                           for t in range(T)]))
    first = np.zeros(T, bool)
    first[list(first_at)] = True
    acts = np.stack([np.arange(T) % 9, np.arange(T) % 15, np.arange(T) % 2, np.zeros(T, int)], 1).astype(np.int64)
    np.savez(pre + '.npz', actions=acts, executed=acts.copy(), weapon=np.full(T, 2, np.int8),
             vec=np.zeros((T, VEC_DIM), np.float32), first=first, rewards=np.zeros(T, np.float32),
             modes=np.zeros(T, np.int8), pose=np.zeros((T, 3), np.float32),
             stats=np.zeros((T, 4), np.float32), counters=np.zeros((T, 3), np.float32),
             bot_visible=np.zeros(T, bool), seed=seed, variant=variant, scenario='deathmatch')
    if aux:
        z = dict(np.load(pre + '.npz'))
        z['enemy_x'] = (np.arange(T) % 9 - 1).astype(np.int8)
        z['bot_visible'] = z['enemy_x'] >= 0
        z['items'] = np.stack([np.arange(T) % 2 == 0] * 4, 1)
        z['depth'] = np.tile(np.arange(16, dtype=np.uint8), (T, 8, 1))
        z['pose'] = np.stack([np.linspace(-700, 1900, T), np.zeros(T), np.zeros(T)], 1).astype(np.float32)
        np.savez(pre + '.npz', **z)
    return pre
