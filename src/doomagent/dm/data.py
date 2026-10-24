import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from doomagent.dm.actions import MOVE_MIRROR, TURN_MIRROR
from doomagent.dm.auxtargets import (AUX_SIZES, DEPTH_SHAPE, ENEMY_X_BINS, ITEM_KINDS, heading_bin,
                                     pos_cell)
from doomagent.dm.config import DM
from doomagent.dm.spatial import GOAL_BINS, MAP_N, MOTION_MIRROR, ODO_K, spatial_targets

MOVE_T, TURN_T = torch.tensor(MOVE_MIRROR), torch.tensor(TURN_MIRROR)
ODO_T, FUT_T = torch.tensor(np.tile(MOTION_MIRROR, len(ODO_K))), torch.tensor(MOTION_MIRROR)


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


def _aux(z):
    # targets of the auxiliary heads (dm.auxtargets); files recorded before them have none of the
    # per-step targets (-1 and not-ok), the map cell and heading come from the pose every file has
    T = len(z['actions'])
    pose = z['pose']
    a = dict(aux_enemy=z['bot_visible'].astype(np.float32),
             aux_x=z['enemy_x'].astype(np.int64) if 'enemy_x' in z.files else np.full(T, -1, np.int64),
             aux_pos=np.array([pos_cell(x, y) for x, y in pose[:, :2]], np.int64),
             aux_head=np.array([heading_bin(h) for h in pose[:, 2]], np.int64))
    a['items_ok'] = 'items' in z.files
    a['aux_items'] = z['items'].astype(np.float32) if a['items_ok'] else np.zeros((T, len(ITEM_KINDS)), np.float32)
    a['depth_ok'] = 'depth' in z.files
    a['aux_depth'] = z['depth'].reshape(T, -1) if a['depth_ok'] else np.zeros((T, AUX_SIZES['depth']), np.uint8)
    return a


AUX_STEP = ('aux_enemy', 'aux_x', 'aux_pos', 'aux_head', 'aux_items')
# per-step targets of the spatial heads (dm.spatial); the depth after the next steps is read at its index
SP_STEP = ('sp_map', 'sp_map_ok', 'sp_odo', 'sp_odo_ok', 'sp_fut', 'sp_fut_ok', 'sp_fut_move', 'sp_fut_turn',
           'sp_goal_dir', 'sp_goal_dist', 'sp_mode')


class _Matches(Dataset):
    # labels in RAM; frames through memmaps opened lazily in each loader process
    def __init__(self, files, stack):
        self.stack, self.files, self._mm = stack, list(files), {}
        self.eps = []
        for p in self.files:
            z = np.load(p)
            self.eps.append(dict(actions=z['actions'], vec=z['vec'].astype(np.float32),
                                 first=z['first'], starts=life_starts(z['first']), **_aux(z),
                                 **spatial_targets(z)))

    def __getstate__(self):
        return dict(self.__dict__, _mm={})

    def _spatial(self, e, ts):
        # the spatial targets of steps ts as float32 / int64 / bool arrays
        out = {k: e[k][ts] for k in SP_STEP}
        out['sp_map'] = out['sp_map'].astype(np.float32)
        out['sp_fut_depth'] = e['aux_depth'][e['sp_fut_idx'][ts]].astype(np.float32) / 255.0
        out['sp_fut_depth_ok'] = out['sp_fut_ok'] & e['depth_ok']
        return out

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
        out = dict(frames=torch.from_numpy(frames), vec=torch.from_numpy(e['vec'][t]),
                   action=torch.from_numpy(e['actions'][t].astype(np.int64)), loss=True,
                   aux_depth=torch.from_numpy(e['aux_depth'][t].astype(np.float32) / 255.0),
                   aux_items_ok=e['items_ok'], aux_depth_ok=e['depth_ok'])
        out.update({k: torch.as_tensor(e[k][t]) for k in AUX_STEP})
        out.update({k: torch.as_tensor(np.asarray(v)) for k, v in self._spatial(e, t).items()})
        return out


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
        out = dict(frames=torch.from_numpy(frames), vec=torch.from_numpy(vec),
                   action=torch.from_numpy(action), loss=torch.from_numpy(loss),
                   valid=torch.from_numpy(valid), aux_items_ok=ep['items_ok'], aux_depth_ok=ep['depth_ok'])
        for k in AUX_STEP + ('aux_depth',):
            src = ep[k][s:e]
            pad = np.full((L,) + src.shape[1:], -1 if src.dtype == np.int64 else 0, src.dtype)
            pad[:e - s] = src
            out[k] = torch.from_numpy(pad.astype(np.float32) / 255.0 if k == 'aux_depth' else pad)
        for k, src in self._spatial(ep, np.arange(s, e)).items():
            pad = np.full((L,) + src.shape[1:], -1 if src.dtype == np.int64 else 0, src.dtype)
            pad[:e - s] = src
            out[k] = torch.from_numpy(pad)
        return out


def mirror_batch(b):
    # a horizontally flipped batch: frames flipped, left and right moves swapped, turns negated
    out = dict(b)
    out['frames'] = torch.flip(b['frames'], dims=[-1])
    a = b['action'].clone()
    a[..., 0] = MOVE_T.to(a.device)[a[..., 0]]
    a[..., 1] = TURN_T.to(a.device)[a[..., 1]]
    out['action'] = a
    if 'aux_x' in b:  # the enemy column flips; the map cell and heading of a mirrored world do not exist
        out['aux_x'] = torch.where(b['aux_x'] >= 0, ENEMY_X_BINS - 1 - b['aux_x'], b['aux_x'])
        out['aux_pos'] = torch.full_like(b['aux_pos'], -1)
        out['aux_head'] = torch.full_like(b['aux_head'], -1)
        out['aux_depth'] = _flip_depth(b['aux_depth'])
    if 'sp_map' in b:  # the egomap and the depth flip, left motion and turns change sign, moves and turns mirror
        mp = b['sp_map']
        out['sp_map'] = torch.flip(mp.reshape(*mp.shape[:-1], MAP_N, MAP_N), dims=[-1]).reshape(mp.shape)
        out['sp_odo'] = b['sp_odo'] * ODO_T.to(mp.device)
        out['sp_fut'] = b['sp_fut'] * FUT_T.to(mp.device)
        out['sp_fut_depth'] = _flip_depth(b['sp_fut_depth'])
        for k, table in (('sp_fut_move', MOVE_T), ('sp_fut_turn', TURN_T)):
            v = b[k]
            out[k] = torch.where(v >= 0, table.to(v.device)[v.clamp_min(0)], v)
        g = b['sp_goal_dir']
        out['sp_goal_dir'] = torch.where(g >= 0, (-g) % GOAL_BINS, g)
    return out


def _flip_depth(d):
    return torch.flip(d.reshape(*d.shape[:-1], *DEPTH_SHAPE), dims=[-1]).reshape(d.shape)
