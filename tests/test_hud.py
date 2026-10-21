import numpy as np


def test_hud_draws_a_status_bar_at_the_bottom_only():
    from doomagent.hud import Hud
    img = np.zeros((480, 640, 3), np.uint8)
    out = Hud().draw(img, health=100, ammo=26)
    assert out.shape == img.shape
    bar_h = 640 * 32 // 320
    assert out[-bar_h:].any() and not out[:-bar_h].any()


def test_hud_numbers_follow_the_game():
    # ammo and health digits change the pixels of their own fields only
    from doomagent.hud import Hud
    hud, img = Hud(), np.zeros((480, 640, 3), np.uint8)
    a, b = hud.draw(img, 100, 26), hud.draw(img, 100, 3)
    diff = np.argwhere((a != b).any(axis=2))
    assert len(diff) and diff[:, 1].max() < 44 * 2 + 2  # ammo field: x < 44 in 320-wide units


def test_hud_face_reacts():
    from doomagent.hud import Hud
    hud, img = Hud(), np.zeros((480, 640, 3), np.uint8)
    calm = hud.draw(img, 100, 26)
    assert (hud.draw(img, 100, 26, killed=True) != calm).any()
    assert (hud.draw(img, 20, 26) != calm).any()
    assert (hud.draw(img, 0, 26) != calm).any()
