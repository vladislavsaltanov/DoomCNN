"""Doom weapons by slot for the teacher: projectile speed, splash, the distances where each is the best
choice, and the choice itself. slot_ammo and owned are AMMO0..7 and WEAPON0..7 (index = slot)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Weapon:
    slot: int
    name: str
    speed: float   # projectile map units per tic, 0 for hitscan
    splash: float  # blast radius that hurts the shooter too
    lo: float      # best from this distance...
    hi: float      # ...to this one
    rank: int      # strength when several fit
    cost: int = 1  # ammo per shot


WEAPONS = {w.slot: w for w in (
    Weapon(1, 'fist', 0.0, 0.0, 0.0, 64.0, 0, 0),
    Weapon(2, 'pistol', 0.0, 0.0, 0.0, 4096.0, 1),
    Weapon(3, 'super shotgun', 0.0, 0.0, 0.0, 384.0, 6, 2),
    Weapon(4, 'chaingun', 0.0, 0.0, 0.0, 4096.0, 3),
    Weapon(5, 'rocket launcher', 20.0, 128.0, 160.0, 4096.0, 5),
    Weapon(6, 'plasma rifle', 25.0, 0.0, 0.0, 4096.0, 4),
    Weapon(7, 'BFG9000', 25.0, 128.0, 192.0, 4096.0, 7, 40),
)}


def usable(owned, slot_ammo):
    # slots in hand's reach: owned with ammo for one shot; the fist always
    return [s for s, w in WEAPONS.items()
            if s == 1 or (s < len(owned) and owned[s] and slot_ammo[s] >= w.cost)]


def best_weapon(owned, slot_ammo, dist):
    # the strongest usable weapon whose range holds dist; else the strongest one that is not too close
    # (no rocket into one's own face); else the fist
    ok = usable(owned, slot_ammo)
    fits = [s for s in ok if WEAPONS[s].lo <= dist <= WEAPONS[s].hi]
    if fits:
        return max(fits, key=lambda s: WEAPONS[s].rank)
    safe = [s for s in ok if dist >= WEAPONS[s].lo]
    return max(safe, key=lambda s: WEAPONS[s].rank) if safe else 1
