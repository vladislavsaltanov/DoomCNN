import pickle

import numpy as np
import pytest
import torch
from epfiles import make_ep

from doomagent.config import FINE_LEFT, FINE_RIGHT, SHOOT, TURN_LEFT, TURN_RIGHT
from doomagent.data import EpisodeDataset, hflip_batch, stack_at


def test_stack_at_pads_with_first_frame():
    frames = np.stack([np.full((2, 2), t, np.uint8) for t in range(6)])
    assert stack_at(frames, 0, 4)[:, 0, 0].tolist() == [0, 0, 0, 0]
    assert stack_at(frames, 1, 4)[:, 0, 0].tolist() == [0, 0, 0, 1]
    assert stack_at(frames, 5, 4)[:, 0, 0].tolist() == [2, 3, 4, 5]


def test_dataset_len_and_item(tmp_path):
    make_ep(tmp_path, 1)
    make_ep(tmp_path, 2, vis=1)
    ds = EpisodeDataset([str(tmp_path)])
    assert len(ds) == 12
    item = ds[7]
    assert item['obs'].shape == (4, 120, 160) and item['obs'].dtype == torch.uint8
    assert item['obs'][-1, 0, 0].item() == 1
    assert item['mask'].shape == (4, 120, 160) and item['mask'][-1, 15, 35].item() == 1
    assert ds[0]['valid'] is True and ds[5]['valid'] is False


def test_dataset_pickles_without_memmaps(tmp_path):
    make_ep(tmp_path, 1)
    ds = EpisodeDataset([str(tmp_path)])
    ds[0]
    assert len(pickle.dumps(ds)) < 50_000


def test_hflip_swaps_turns_and_negates_offset():
    obs = torch.arange(24, dtype=torch.uint8).reshape(2, 4, 1, 3)
    b = dict(obs=obs, mask=obs % 2, action=torch.tensor([TURN_LEFT, SHOOT]),
             off=torch.tensor([0.5, -0.2]), vis=torch.tensor([1, 1]))
    f = hflip_batch(b)
    assert f['action'].tolist() == [TURN_RIGHT, SHOOT]
    assert torch.allclose(f['off'], torch.tensor([-0.5, 0.2]))
    assert torch.equal(f['obs'], torch.flip(obs, dims=[-1]))
    assert torch.equal(f['mask'], torch.flip(obs % 2, dims=[-1]))
    r = hflip_batch(f)
    assert torch.equal(r['obs'], obs) and r['action'].tolist() == b['action'].tolist()


def test_hflip_swaps_fine_turns():
    a = torch.tensor([FINE_LEFT, FINE_RIGHT, TURN_LEFT, SHOOT])
    b = dict(obs=torch.zeros(4, 1, 1, 2), mask=torch.zeros(4, 1, 1, 2), action=a,
             off=torch.zeros(4), vis=torch.zeros(4))
    assert hflip_batch(b)['action'].tolist() == [FINE_RIGHT, FINE_LEFT, TURN_RIGHT, SHOOT]


@pytest.mark.vizdoom
def test_dataset_stack_matches_env_stack():
    from doomagent.env import DoomEnv
    env = DoomEnv()
    obs = env.reset(5000)
    frames, stacks = [obs[-1].copy()], [obs.copy()]
    for _ in range(6):
        obs, _, _ = env.step(TURN_RIGHT)
        frames.append(obs[-1].copy())
        stacks.append(obs.copy())
    env.close()
    frames = np.stack(frames)
    for t, s in enumerate(stacks):
        assert (stack_at(frames, t, 4) == s).all()
