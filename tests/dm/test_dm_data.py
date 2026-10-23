from dataclasses import replace

import numpy as np
import pytest
import torch

from dmfiles import make_match
from doomagent.dm.augment import frame_augment
from doomagent.dm.actions import BACK_LEFT, BACK_RIGHT, FORWARD, STRAFE_LEFT, STRAFE_RIGHT, TURN_DEG, VEC_DIM
from doomagent.dm.config import DM
from doomagent.dm.data import (MatchSegments, MatchSteps, life_starts, match_files, mirror_batch,
                               split_holdout, stack_at, windows)


def test_life_starts():
    assert life_starts(np.array([1, 0, 0, 1, 0], bool)).tolist() == [0, 0, 0, 3, 3]


def test_stack_at_pads_with_the_life_start():
    frames = np.arange(8, dtype=np.uint8)[:, None, None].repeat(2, 1).repeat(2, 2)
    assert stack_at(frames, 6, 5)[:, 0, 0].tolist() == [5, 5, 5, 6]
    assert stack_at(frames, 3, 0)[:, 0, 0].tolist() == [0, 1, 2, 3]


def test_windows_tile_every_step_once():
    rng = np.random.default_rng(0)
    first = np.zeros(1000, bool)
    first[np.r_[0, np.sort(rng.choice(np.arange(1, 1000), 12, replace=False))]] = True
    starts = set(np.flatnonzero(first))
    covered = []
    for s, e, lf in windows(first, 32, 8):
        assert e - s <= 32 and not starts & set(range(s + 1, e))
        assert lf == s if s in starts else lf == s + 8
        covered += range(lf, e)
    assert sorted(covered) == list(range(1000))


def test_steps_and_segments_cover_the_same_steps(tmp_path):
    for seed, fa in ((1, (0, 17, 50)), (2, (0,))):
        make_match(tmp_path, seed, T=90, first_at=fa)
    files = match_files([str(tmp_path)])
    seg = MatchSegments(files)
    assert len(MatchSteps(files)) == sum(int(seg[i]['loss'].sum()) for i in range(len(seg)))


def test_segment_frames_equal_step_frames(tmp_path):
    make_match(tmp_path, 1, T=70, first_at=(0, 40))
    files = match_files([str(tmp_path)])
    steps, seg = MatchSteps(files), MatchSegments(files)
    item, (s, e, lf) = seg[1], seg.index[1][1:]
    for k in range(lf - s, e - s):
        assert torch.equal(item['frames'][k], steps[s + k]['frames'])


def test_padding_is_neither_valid_nor_loss(tmp_path):
    make_match(tmp_path, 1, T=20)
    it = MatchSegments(match_files([str(tmp_path)]))[0]
    assert it['frames'].shape == (32, 4, DM.height, DM.width)
    assert not it['valid'][20:].any() and not it['loss'][20:].any()


def test_mirror_batch_heads():
    act = torch.tensor([[BACK_LEFT, TURN_DEG.index(-20.0), 1, 0], [STRAFE_RIGHT, TURN_DEG.index(3.0), 0, 1]])
    b = dict(frames=torch.arange(6.0).view(1, 1, 1, 6).expand(2, 4, 1, 6).clone(), action=act)
    m = mirror_batch(b)
    assert m['frames'][0, 0, 0].tolist() == [5, 4, 3, 2, 1, 0]
    assert m['action'].tolist() == [[BACK_RIGHT, TURN_DEG.index(20.0), 1, 0], [STRAFE_LEFT, TURN_DEG.index(-3.0), 0, 1]]
    assert mirror_batch(dict(b, action=act.view(1, 2, 4)))['action'].shape == (1, 2, 4)


def test_items_carry_four_heads(tmp_path):
    make_match(tmp_path, 1, T=20)
    files = match_files([str(tmp_path)])
    assert MatchSteps(files)[3]['action'].tolist() == [3, 3, 1, 0]
    seg = MatchSegments(files)[0]
    assert seg['action'].shape == (32, 4) and seg['vec'].shape == (32, VEC_DIM)


def test_holdout_from_the_first_dir_only(tmp_path):
    for s in range(40):
        make_match(tmp_path / 'a', s, T=5)
    for s in range(10):
        make_match(tmp_path / 'b', 100 + s, T=5)
    train, hold = split_holdout([str(tmp_path / 'a'), str(tmp_path / 'b')])
    assert len(hold) == 2 and all('/a/' in h.replace('\\', '/') for h in hold)
    assert not set(train) & set(hold) and len(train) == 48


def test_frame_augment_keeps_shape_and_range():
    x = torch.rand(3, 8, DM.height, DM.width)
    y = frame_augment(x)
    assert y.shape == x.shape and float(y.min()) >= 0.0 and float(y.max()) <= 1.0


@pytest.mark.vizdoom
def test_dataset_stack_equals_env_stack():
    from doomagent.dm.env import DeathmatchEnv
    env = DeathmatchEnv(replace(DM, timelimit=0.1))
    o, frames, first, stacks, done, t = env.reset(38002), [], [], [], False, 0
    while not done:
        frames.append(o.frames[-1])
        first.append(o.first)
        stacks.append(o.frames)
        if t == 20:
            env.game.send_game_command('kill')
        o, _, done = env.step((FORWARD, 0, 0, 0))
        t += 1
    env.close()
    frames, first = np.stack(frames), np.array(first)
    ls = life_starts(first)
    assert first.sum() >= 2
    assert all(np.array_equal(stack_at(frames, i, ls[i]), stacks[i]) for i in range(len(frames)))


def test_aux_targets_in_steps_and_segments(tmp_path):
    from doomagent.dm.auxtargets import pos_cell
    make_match(tmp_path, 1, T=40, aux=True)
    files = sorted(str(p) for p in tmp_path.glob('ep_*.npz'))
    it = MatchSteps(files)[3]
    assert it['aux_enemy'] and it['aux_x'] == 2 and it['aux_items_ok'] and it['aux_depth_ok']
    assert it['aux_depth'].shape == (128,) and float(it['aux_depth'].max()) == pytest.approx(15 / 255)
    z = np.load(files[0])
    assert it['aux_pos'] == pos_cell(*z['pose'][3, :2]) and it['aux_head'] == 0
    seg = MatchSegments(files, seg_len=32, burn_in=8)[0]
    assert seg['aux_x'].shape == (32,) and seg['aux_depth'].shape == (32, 128) and seg['aux_x'][3] == 2


def test_aux_targets_missing_in_old_files(tmp_path):
    make_match(tmp_path, 1, T=10)
    it = MatchSteps(sorted(str(p) for p in tmp_path.glob('ep_*.npz')))[0]
    assert it['aux_x'] == -1 and not it['aux_items_ok'] and not it['aux_depth_ok'] and it['aux_pos'] >= 0


def test_mirror_batch_flips_aux_and_drops_the_map_targets():
    import torch
    b = dict(frames=torch.zeros(2, 4, 3, 5), vec=torch.zeros(2, 22), action=torch.zeros(2, 4, dtype=torch.long),
             aux_x=torch.tensor([0, -1]), aux_pos=torch.tensor([5, 6]), aux_head=torch.tensor([1, 2]),
             aux_depth=torch.arange(256.).reshape(2, 128))
    m = mirror_batch(b)
    assert m['aux_x'].tolist() == [7, -1] and m['aux_pos'].tolist() == [-1, -1] and m['aux_head'].tolist() == [-1, -1]
    assert m['aux_depth'][0, :16].tolist() == list(range(15, -1, -1))
