from doomagent.dm.weapons import WEAPONS, best_weapon

OWNED_ALL = (0, 1, 1, 1, 1, 1, 1, 0)
AMMO = (0, 0, 100, 20, 100, 10, 0, 0)  # AMMO0..7: pistol and chaingun share bullets


def test_table():
    assert WEAPONS[5].speed == 20.0 and WEAPONS[5].lo == 160.0 and WEAPONS[6].speed == 25.0
    assert sorted(WEAPONS) == list(range(1, 8)) and WEAPONS[3].hi == 384.0


def test_best_weapon_by_distance():
    assert best_weapon(OWNED_ALL, AMMO, 200.0) == 3      # super shotgun up close
    assert best_weapon(OWNED_ALL, AMMO, 1000.0) == 5     # rocket launcher far away
    assert best_weapon(OWNED_ALL, AMMO, 100.0) == 3      # no rocket closer than 160


def test_best_weapon_skips_empty():
    assert best_weapon((0, 1, 1, 1, 0, 1, 0, 0), (0, 0, 50, 0, 0, 0, 0, 0), 200.0) == 2


def test_best_weapon_out_of_ammo_falls_back_to_the_fist():
    assert best_weapon((0, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0), 300.0) == 1


def test_rocket_too_close_and_nothing_else():
    # only a rocket launcher with ammo and the target at 100: the pistol, not a rocket into one's own face
    assert best_weapon((0, 1, 1, 0, 0, 1, 0, 0), (0, 0, 50, 0, 0, 10, 0, 0), 100.0) == 2
