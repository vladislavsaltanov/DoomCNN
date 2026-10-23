import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from doomagent.dm.actions import MOVE_MIRROR, TURN_MIRROR
from doomagent.dm.config import DM

MOVE_T, TURN_T = torch.tensor(MOVE_MIRROR), torch.tensor(TURN_MIRROR)


def life_starts(first):
    # for every step, the index of the step that started its life
    idx = np.where(np.asarray(first, bool), np.arange(len(first)), 0)
    return np.maximum.accumulate(idx)


def stack_at(frames, t, start, stack=4):
    # same rule as DeathmatchEnv: oldest..newest, padded with the first frame of the life
    return frames[[max(t - j, start) for j in range(stack - 1, -1, -1)]]


def windows(first, seg_len=32, burn_in=8):
    # M1 windows inside lives: starts every seg_len - burn_in steps; a window that starts a life
    # has an exact zero memory and trains on all its steps, the others warm up for burn_in steps,
    # so every step is in exactly one window's loss
    first = np.asarray(first, bool)
    starts = list(np.flatnonzero(first)) + [len(first)]
    out = []
    for a, b in zip(starts, starts[1:]):
        s = a
        while s < b:
            lf = s if s == a else s + burn_in
            e = min(s + seg_len, b)
            if lf >= e:
                break
            out.append((s, e, lf))
            s += seg_len - burn_in
    return out


def match_files(dirs):
    return [p for d in dirs for p in sorted(glob.glob(os.path.join(d, 'ep_*.npz')))]


def split_holdout(dirs, share=0.05):
    # the last files of the first dir are held out for offline loss, accuracy and confusion
    first = sorted(glob.glob(os.path.join(dirs[0], 'ep_*.npz')))
    k = max(1, int(len(first) * share)) if len(first) >= 2 else 0
    hold = first[len(first) - k:] if k else []
    return [p for p in match_files(dirs) if p not in set(hold)], hold


class _Matches(Dataset):
    # labels in RAM; frames through memmaps opened lazily in each loader process
    def __init__(self, files, stack):
        self.stack, self.files, self._mm = stack, list(files), {}
        self.eps = []
        for p in self.files:
            z = np.load(p)
            self.eps.append(dict(actions=z['actions'], vec=z['vec'].astype(np.float32),
                                 first=z['first'], starts=life_starts(z['first'])))

    def __getstate__(self):
        return dict(self.__dict__, _mm={})

    def _frames(self, i):
        if i not in self._mm:
            self._mm[i] = np.load(self.files[i][:-4] + '.frames.npy', mmap_mode='r')
        return self._mm[i]


class MatchSteps(_Matches):
    # M0: one sample per step
    def __init__(self, files, stack=DM.stack):
        super().__init__(files, stack)
        self.index = [(i, t) for i, e in enumerate(self.eps) for t in range(len(e['actions']))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        e = self.eps[i]
        frames = np.ascontiguousarray(stack_at(self._frames(i), t, e['starts'][t], self.stack))
        return dict(frames=torch.from_numpy(frames), vec=torch.from_numpy(e['vec'][t]),
                    action=torch.from_numpy(e['actions'][t].astype(np.int64)), loss=True)


class MatchSegments(_Matches):
    # M1: one window of seg_len steps inside a life, padded at the end
    def __init__(self, files, seg_len=DM.seg_len, burn_in=DM.burn_in, stack=DM.stack):
        super().__init__(files, stack)
        self.seg_len = seg_len
        self.index = [(i, s, e, lf) for i, ep in enumerate(self.eps)
                      for s, e, lf in windows(ep['first'], seg_len, burn_in)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s, e, lf = self.index[k]
        ep, fr, L = self.eps[i], self._frames(i), self.seg_len
        frames = np.zeros((L, self.stack) + fr.shape[1:], np.uint8)
        for j in range(e - s):
            frames[j] = stack_at(fr, s + j, ep['starts'][s + j], self.stack)
        vec = np.zeros((L, ep['vec'].shape[1]), np.float32)
        vec[:e - s] = ep['vec'][s:e]
        action = np.zeros((L, ep['actions'].shape[1]), np.int64)
        action[:e - s] = ep['actions'][s:e]
        valid = np.arange(L) < e - s
        loss = valid & (np.arange(L) >= lf - s)
        return dict(frames=torch.from_numpy(frames), vec=torch.from_numpy(vec),
                    action=torch.from_numpy(action), loss=torch.from_numpy(loss),
                    valid=torch.from_numpy(valid))


def mirror_batch(b):
    # a horizontally flipped batch: frames flipped, left and right moves swapped, turns negated
    out = dict(b)
    out['frames'] = torch.flip(b['frames'], dims=[-1])
    a = b['action'].clone()
    a[..., 0] = MOVE_T.to(a.device)[a[..., 0]]
    a[..., 1] = TURN_T.to(a.device)[a[..., 1]]
    out['action'] = a
    return out
