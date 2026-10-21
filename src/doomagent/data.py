import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from doomagent.config import FINE_LEFT, FINE_RIGHT, TURN_LEFT, TURN_RIGHT


def stack_at(frames, t, stack=4):
    # same rule as DoomEnv: oldest..newest, padded with the first frame
    idx = [max(t - j, 0) for j in range(stack - 1, -1, -1)]
    return frames[idx]


class EpisodeDataset(Dataset):
    # labels in RAM; frames and masks through memmaps opened lazily in each worker
    def __init__(self, dirs, stack=4):
        self.stack, self.eps, self.index, self._mm = stack, [], [], {}
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(d, 'ep_*.npz'))):
                z = np.load(p)
                self.eps.append(dict(prefix=p[:-4], **{k: z[k] for k in
                                                       ('actions', 'vis', 'off', 'valid')}))
        for i, e in enumerate(self.eps):
            self.index += [(i, t) for t in range(len(e['actions']))]

    def __getstate__(self):
        s = self.__dict__.copy()
        s['_mm'] = {}
        return s

    def _arrays(self, i):
        if i not in self._mm:
            pre = self.eps[i]['prefix']
            self._mm[i] = (np.load(pre + '.frames.npy', mmap_mode='r'),
                           np.load(pre + '.masks.npy', mmap_mode='r'))
        return self._mm[i]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        e = self.eps[i]
        frames, masks = self._arrays(i)
        obs = np.ascontiguousarray(stack_at(frames, t, self.stack))
        mask = np.unpackbits(stack_at(masks, t, self.stack), axis=-1)
        return dict(obs=torch.from_numpy(obs), mask=torch.from_numpy(mask),
                    action=int(e['actions'][t]), vis=int(e['vis'][t]), off=float(e['off'][t]),
                    valid=bool(e['valid'][t]))


def hflip_batch(b):
    out = dict(b)
    a = b['action']
    flipped = a.clone()
    flipped[a == TURN_LEFT] = TURN_RIGHT
    flipped[a == TURN_RIGHT] = TURN_LEFT
    flipped[a == FINE_LEFT] = FINE_RIGHT
    flipped[a == FINE_RIGHT] = FINE_LEFT
    out['action'] = flipped
    out['obs'] = torch.flip(b['obs'], dims=[-1])
    out['mask'] = torch.flip(b['mask'], dims=[-1])
    out['off'] = -b['off']
    return out
