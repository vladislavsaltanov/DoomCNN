import math
from dataclasses import dataclass

import numpy as np

from doomagent.config import FINE_LEFT, FINE_RIGHT, SHOOT, TURN_LEFT, TURN_RIGHT, Cfg

# Doom monster classes. scripts/discover_labels.py fails on any name that is in neither set:
# classify it here instead of letting it pass silently.
MONSTERS = {'ZombieMan', 'ShotgunGuy', 'ChaingunGuy', 'DoomImp', 'Demon', 'Spectre',
            'LostSoul', 'Cacodemon', 'HellKnight', 'BaronOfHell', 'Arachnotron',
            'PainElemental', 'Revenant', 'Fatso', 'Archvile', 'Cyberdemon',
            'SpiderMastermind', 'WolfensteinSS'}
NON_ENEMIES = {'DoomPlayer', 'Blood', 'BulletPuff', 'Clip', 'TeleportFog', 'DoomImpBall',
               'MarineChainsawVzd'}
# the coarse direction of each turn action: scanning always goes coarse, towards the last turn
COARSE_OF = {TURN_LEFT: TURN_LEFT, FINE_LEFT: TURN_LEFT,
             TURN_RIGHT: TURN_RIGHT, FINE_RIGHT: TURN_RIGHT}


def is_enemy_name(name):
    return not name.startswith('Dead') and name in MONSTERS


@dataclass(frozen=True)
class Enemy:
    id: int
    left: float      # box edges normalized to [-1, 1], negative = left of the screen center
    right: float
    dist: float      # world distance to the player
    width_px: float  # box width at network resolution

    @property
    def offset(self):
        return (self.left + self.right) / 2.0


def enemies_from_labels(labels, px, py, screen_w):
    out = []
    for lb in labels:
        if not is_enemy_name(lb.object_name):
            continue
        left = 2.0 * lb.x / screen_w - 1.0
        right = 2.0 * (lb.x + lb.width) / screen_w - 1.0
        dist = math.hypot(lb.object_position_x - px, lb.object_position_y - py)
        out.append(Enemy(int(lb.object_id), left, right, dist, float(lb.width)))
    return out


def enemy_mask(priv, cfg):
    # (height, width) bool: a cell is set if any of its full-resolution pixels shows an enemy
    if priv.labels_buffer is None:
        return np.zeros((cfg.height, cfg.width), bool)
    values = [lb.value for lb in priv.labels if is_enemy_name(lb.object_name)]
    full = np.isin(priv.labels_buffer, values)
    fy, fx = full.shape[0] // cfg.height, full.shape[1] // cfg.width
    return full.reshape(cfg.height, fy, cfg.width, fx).any(axis=(1, 3))


def is_aimed(e, margin):
    m = margin * (e.right - e.left)
    return e.left + m <= 0.0 <= e.right - m


def _towards(offset, fallback, fine_zone=0.0):
    # a coarse step would overshoot inside the fine zone, so the teacher steps finely there
    fine = abs(offset) <= fine_zone
    if offset < 0:
        return FINE_LEFT if fine else TURN_LEFT
    if offset > 0:
        return FINE_RIGHT if fine else TURN_RIGHT
    return fallback


def pick_target(enemies, lock_id=None, keep=1.25):
    # nearest enemy, but keep the locked one unless another is clearly closer
    nearest = min(enemies, key=lambda e: e.dist)
    for e in enemies:
        if e.id == lock_id and e.dist <= keep * nearest.dist:
            return e
    return nearest


def teacher_action(enemies, attack_ready, margin=0.2, scan=TURN_RIGHT, lock_id=None,
                   skip_id=None, min_w=0.0, fine_zone=0.0):
    if not enemies:
        return scan, None
    pool = enemies
    if not attack_ready and skip_id is not None:
        # the weapon is cycling after a shot at skip_id: pre-aim at the next target
        pool = [e for e in enemies if e.id != skip_id] or enemies
    target = pick_target(pool, lock_id)
    if is_aimed(target, margin) and attack_ready and target.width_px >= min_w:
        return SHOOT, target.id
    return _towards(target.offset, scan, fine_zone), target.id


def nearest_aux(enemies):
    if not enemies:
        return 0, 0.0
    e = min(enemies, key=lambda x: x.dist)
    return 1, e.offset


class Teacher:
    def __init__(self, cfg: Cfg = Cfg()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.lock, self.skip, self.scan = None, None, TURN_RIGHT

    def __call__(self, obs, priv):
        c = self.cfg
        enemies = enemies_from_labels(priv.labels, priv.px, priv.py, c.width)
        if priv.attack_ready:
            self.skip = None
        a, tid = teacher_action(enemies, priv.attack_ready, c.aim_margin, self.scan, self.lock,
                                self.skip, c.min_target_w, c.fine_zone)
        self.lock = tid
        if a == SHOOT:
            self.skip = tid
        else:
            self.scan = COARSE_OF[a]
        return a
