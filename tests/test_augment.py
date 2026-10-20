import torch

from doomagent.augment import _vshift, augment


def batch():
    torch.manual_seed(0)
    obs = torch.rand(3, 4, 120, 160)
    mask = torch.zeros(3, 4, 120, 160, dtype=torch.bool)
    mask[:, :, 40:60, 70:90] = True
    return obs, mask


def test_shapes_and_range():
    obs, mask = batch()
    x, m = augment(obs, mask)
    assert x.shape == obs.shape and m.shape == mask.shape and m.dtype == torch.bool
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_background_changes_and_enemy_pixels_stay():
    obs, mask = batch()
    x, m = augment(obs, mask, p_bg=1.0, p_enemy=0.0, global_aug=False, max_shift=0)
    assert torch.equal(x[mask], obs[mask])
    assert (x[~mask] != obs[~mask]).any()
    assert torch.equal(m, mask)


def test_vertical_shift_moves_rows_down():
    t = torch.arange(6.0).reshape(1, 1, 6, 1)
    s = _vshift(t, 2)
    assert s[0, 0, :, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
