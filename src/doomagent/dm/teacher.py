import math
from collections import deque

from dataclasses import dataclass

from doomagent.dm.config import (BARREL, BOT, DM, DMCfg, FINE_LEFT, FINE_RIGHT, FORWARD, FORWARD_LEFT,
                                 FORWARD_RIGHT, ITEM_KIND, MODES, ROCKET_SPEED, SHOOT, STRAFE_LEFT,
                                 STRAFE_RIGHT, TIC_RATE, TURN_LEFT, TURN_RIGHT)
from doomagent.dm.nav import lookahead, nav_for

ARRIVE_R = 64.0       # a goal counts as reached this close
PICKUP_R = 24.0       # a known item this close is forgotten: picked up or gone
LOOKAHEAD = 64.0      # steer towards the path point this far ahead
REPLAN_STEPS = 8
STUCK_STEPS = 8       # moving less than STUCK_MOVE over this many navigation steps is stuck
STUCK_MOVE = 16.0
BARREL_MARGIN = 32.0  # a barrel in the line of fire closer than safe_dist + this blocks the shot
LEFT_ACTIONS = {TURN_LEFT, FINE_LEFT, FORWARD_LEFT}


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


def is_aimed(e, margin):
    m = margin * (e.right - e.left)
    return e.left + m <= 0.0 <= e.right - m


def pick_target(enemies, lock_id=None, keep=1.25):
    # nearest bot, but keep the locked one unless another is clearly closer
    nearest = min(enemies, key=lambda e: e.dist)
    for e in enemies:
        if e.id == lock_id and e.dist <= keep * nearest.dist:
            return e
    return nearest


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def bearing(px, py, angle, x, y):
    # degrees from the view direction to (x, y), + is left
    return wrap(math.degrees(math.atan2(y - py, x - px)) - angle)


def x_ndc(px, py, angle, x, y):
    # screen position of a world point at 90 degrees of horizontal view: -1 left edge, 1 right edge
    rel = max(-89.0, min(89.0, bearing(px, py, angle, x, y)))
    return -math.tan(math.radians(rel))


def visible_bots(labels, px, py, screen_w):
    out = []
    for lb in labels:
        if lb.object_name != BOT:
            continue
        dist = math.hypot(lb.object_position_x - px, lb.object_position_y - py)
        if dist < 1.0:  # the player's own label, if the env let one through
            continue
        out.append(Enemy(int(lb.object_id), 2.0 * lb.x / screen_w - 1.0,
                         2.0 * (lb.x + lb.width) / screen_w - 1.0, dist, float(lb.width)))
    return out


def lead_shift(track, px, py, angle, tx, ty):
    # screen shift from the target to where a rocket meets it: velocity from the track's ends
    if len(track) < 2 or track[-1][0] == track[0][0]:
        return 0.0
    dt = track[-1][0] - track[0][0]
    vx, vy = (track[-1][1] - track[0][1]) / dt, (track[-1][2] - track[0][2]) / dt
    t = math.hypot(tx - px, ty - py) / ROCKET_SPEED
    lx, ly = tx + vx * t, ty + vy * t
    t = math.hypot(lx - px, ly - py) / ROCKET_SPEED  # one refinement on the lead point's range
    lx, ly = tx + vx * t, ty + vy * t
    return x_ndc(px, py, angle, lx, ly) - x_ndc(px, py, angle, tx, ty)


def turn_towards(offset, fallback, fine_zone):
    fine = abs(offset) <= fine_zone
    if offset < 0:
        return FINE_LEFT if fine else TURN_LEFT
    if offset > 0:
        return FINE_RIGHT if fine else TURN_RIGHT
    return fallback


def steer(rel):
    if abs(rel) > 45.0:
        return TURN_LEFT if rel > 0 else TURN_RIGHT
    if abs(rel) > 7.0:
        return FORWARD_LEFT if rel > 0 else FORWARD_RIGHT
    return FORWARD


class DMTeacher:
    # scripted player on privileged data, honest about what it knows: items and bots only from
    # labels seen in this match, chase only to where a bot was seen; objects_info is never read
    def __init__(self, cfg: DMCfg = DM, keep_trace=False):
        self.cfg, self.keep_trace = cfg, keep_trace
        self.nav, self.ring = None, []
        self.reset()

    def reset(self):
        self.known, self.modes = [], []
        self.mode, self.goal = 'patrol', None
        self._new_life(None)

    def _new_life(self, me):
        self.tracks, self.lock, self.target, self.lead, self.last_seen = {}, None, None, None, None
        self.path, self._path_goal, self._path_age = [], None, 0
        self.scan, self.unstick, self.unstick_side = TURN_RIGHT, 0, STRAFE_LEFT
        self.moves = deque(maxlen=STUCK_STEPS + 1)
        self.ring_i = 0
        if me is not None and self.ring:
            self.ring_i = min(range(len(self.ring)),
                              key=lambda i: math.hypot(self.ring[i][0] - me[0], self.ring[i][1] - me[1]))

    def take_trace(self):
        out, self.modes = dict(modes=list(self.modes)), []
        return out

    def __call__(self, obs, priv):
        c = self.cfg
        me = (priv.px, priv.py)
        if self.nav is None:
            self.nav = nav_for(priv.sectors)
            self.ring = self.nav.patrol_ring()
            self._new_life(me)
        if obs.first:
            self._new_life(me)
        self._update_items(priv.labels, me)
        bots = visible_bots(priv.labels, priv.px, priv.py, c.width)
        seen = {e.id for e in bots}
        self.tracks = {k: v for k, v in self.tracks.items() if k in seen}
        for lb in priv.labels:
            if int(lb.object_id) in seen:
                self.tracks.setdefault(int(lb.object_id), deque(maxlen=4)).append(
                    (priv.tic, lb.object_position_x, lb.object_position_y))
        if bots:
            near = min(bots, key=lambda e: e.dist)
            _, x, y = self.tracks[near.id][-1]
            self.last_seen = (x, y, priv.tic)
        action = self._decide(priv, me, bots)
        if self.keep_trace:
            self.modes.append(MODES.index(self.mode))
        return action

    def _update_items(self, labels, me):
        for lb in labels:
            kind = ITEM_KIND.get(lb.object_name)
            if kind and not any(math.hypot(x - lb.object_position_x, y - lb.object_position_y) < 16
                                for _, x, y in self.known):
                self.known.append((kind, lb.object_position_x, lb.object_position_y))
        self.known = [k for k in self.known if math.hypot(k[1] - me[0], k[2] - me[1]) >= PICKUP_R]

    def _needed_item(self, priv, me):
        c = self.cfg
        for kind, low, value in (('health', c.low_health, priv.health), ('ammo', c.low_ammo, priv.ammo)):
            if value < low:
                cands = [k for k in self.known if k[0] == kind]
                if cands:
                    return min(cands, key=lambda k: math.hypot(k[1] - me[0], k[2] - me[1]))
        return None

    def _decide(self, priv, me, bots):
        c = self.cfg
        if priv.ammo > 0 and bots:
            self.mode = 'combat'
            return self._combat(priv, me, bots)
        self.target, self.lead = None, None
        if self.last_seen is not None:
            x, y, tic = self.last_seen
            if priv.tic - tic > c.chase_s * TIC_RATE or math.hypot(x - me[0], y - me[1]) < ARRIVE_R:
                self.last_seen = None
        if priv.ammo > 0 and self.last_seen is not None:
            self.mode, self.goal = 'chase', self.last_seen[:2]
            a = self._navigate(priv, me)
            if a is not None:
                return a
            self.last_seen = None
        item = self._needed_item(priv, me)
        while item is not None:
            self.mode, self.goal = 'items', item[1:]
            a = self._navigate(priv, me)
            if a is not None:
                return a
            self.known.remove(item)  # no path to it: forget it, try the next one
            item = self._needed_item(priv, me)
        self.mode = 'patrol'
        goal = self.ring[self.ring_i]
        if math.hypot(goal[0] - me[0], goal[1] - me[1]) < ARRIVE_R:
            self.ring_i = (self.ring_i + 1) % len(self.ring)
            goal = self.ring[self.ring_i]
        self.goal = goal
        a = self._navigate(priv, me)
        return TURN_RIGHT if a is None else a

    def _combat(self, priv, me, bots):
        c = self.cfg
        t = pick_target(bots, self.lock)
        self.lock, self.target = t.id, t
        _, tx, ty = self.tracks[t.id][-1]
        self.lead = lead_shift(self.tracks[t.id], priv.px, priv.py, priv.angle, tx, ty)
        aim = Enemy(t.id, t.left + self.lead, t.right + self.lead, t.dist, t.width_px)
        safe = (t.dist >= c.safe_dist
                and self.nav.wall_dist(priv.px, priv.py, priv.angle) >= c.safe_dist
                and not self._barrel_in_line(priv, me))
        if is_aimed(aim, c.aim_margin) and priv.attack_ready and safe and t.width_px >= c.min_target_w:
            return SHOOT
        if t.dist < c.safe_dist:
            left = self.nav.wall_dist(priv.px, priv.py, priv.angle + 90.0)
            right = self.nav.wall_dist(priv.px, priv.py, priv.angle - 90.0)
            return STRAFE_LEFT if left >= right else STRAFE_RIGHT
        a = turn_towards(aim.offset, self.scan, c.fine_zone)
        if a != SHOOT:
            self.scan = TURN_LEFT if a in LEFT_ACTIONS else TURN_RIGHT
        return a

    def _barrel_in_line(self, priv, me):
        w = self.cfg.width
        for lb in priv.labels:
            if lb.object_name != BARREL:
                continue
            if 2.0 * lb.x / w - 1.0 <= 0.0 <= 2.0 * (lb.x + lb.width) / w - 1.0:
                d = math.hypot(lb.object_position_x - me[0], lb.object_position_y - me[1])
                if d < self.cfg.safe_dist + BARREL_MARGIN:
                    return True
        return False

    def _navigate(self, priv, me):
        # one step along the A* path to self.goal; None when there is no path
        if self.unstick > 0:
            self.unstick -= 1
            return self.unstick_side
        off_path = (not self.path or
                    min(math.hypot(x - me[0], y - me[1]) for x, y in self.path) > LOOKAHEAD)
        if self.goal != self._path_goal or self._path_age >= REPLAN_STEPS or off_path:
            self.path, self._path_goal, self._path_age = self.nav.path(me, self.goal), self.goal, 0
        self._path_age += 1
        if not self.path:
            return None
        self.moves.append(me)
        if len(self.moves) == self.moves.maxlen and math.hypot(
                self.moves[-1][0] - self.moves[0][0], self.moves[-1][1] - self.moves[0][1]) < STUCK_MOVE:
            self.moves.clear()
            self.unstick = 1
            self.unstick_side = STRAFE_RIGHT if self.unstick_side == STRAFE_LEFT else STRAFE_LEFT
            self._path_age = REPLAN_STEPS
            return self.unstick_side
        wp = lookahead(self.path, me[0], me[1], LOOKAHEAD)
        return steer(bearing(priv.px, priv.py, priv.angle, wp[0], wp[1]))
