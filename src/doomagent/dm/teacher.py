"""The cheater teacher for cig MAP02 (spec v2 §4): reflexes, not knowledge.

It knows the map: its geometry (A* on the sector lines) and where weapons, ammo, medikits and armor lie
(read from objects on the first step of a match only). Bots and flying projectiles it knows only from labels,
i.e. what is on screen; then it remembers where it saw a bot. Every step it returns one value per head
(move, turn, fire, weapon): it aims with exact turns and a lead for the weapon in hand, fires when aimed and
safe, strafes and dodges at the same time, and takes the best weapon for the distance.
"""
import math
from collections import deque
from dataclasses import dataclass

from doomagent.dm.actions import (BACK_LEFT, BACK_RIGHT, FORWARD, FORWARD_LEFT, FORWARD_RIGHT, STAY, STRAFE_LEFT,
                                  STRAFE_RIGHT, TURN_DEG)
from doomagent.dm.aim import dodge_move, half_width_deg, lead_point, pick_turn, target_angle
from doomagent.dm.config import (AMMO_ITEMS, BARREL, BOT, DM, DMCfg, ITEM_KIND, MODES, PROJECTILES, TIC_RATE,
                                 WEAPON_ITEMS)
from doomagent.dm.nav import lookahead, nav_for
from doomagent.dm.weapons import WEAPONS, best_weapon, usable

ARRIVE_R = 64.0       # a goal counts as reached this close
PICKUP_R = 32.0       # an item spot this close is taken: it is skipped for ITEM_BACK
ITEM_BACK = 30 * TIC_RATE
REPLAN_STEPS = 8
STUCK_STEPS = 8       # moving less than STUCK_MOVE over this many navigation steps is stuck
STUCK_MOVE = 16.0
NAV_TOL = 3.0         # degrees: no turn while walking if the path point is this close to straight ahead
WALK_MAX = 45.0       # degrees: turn in place first when the path point is farther off
SWITCH_STEPS = 3      # a weapon change takes about 3 steps; no new press before
SIDE_ROOM = 48.0      # a strafe turns back before a wall this close on its side
RANGE_MARGIN = 32.0   # back off this far beyond a weapon's minimum range
NAV_RANGE = 512.0     # the distance the weapon in hand is chosen for when nobody is in sight
BARREL_R = 160.0      # a barrel this close in the line of fire blows up the shooter too
SIDE = {(0, 1): STRAFE_LEFT, (0, -1): STRAFE_RIGHT, (1, 1): FORWARD_LEFT, (1, -1): FORWARD_RIGHT,
        (-1, 1): BACK_LEFT, (-1, -1): BACK_RIGHT}


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


def pick_target(enemies, lock_id=None, keep=1.25):
    # nearest bot, but keep the locked one unless another is clearly closer
    nearest = min(enemies, key=lambda e: e.dist)
    for e in enemies:
        if e.id == lock_id and e.dist <= keep * nearest.dist:
            return e
    return nearest


def bearing(px, py, angle, x, y):
    # degrees from the view direction to (x, y), + is left (kept for scripts of version 1.1)
    return target_angle(px, py, angle, x, y)


def visible_bots(labels, px, py, screen_w, vis_px=None, min_px=0.0):
    # bots on screen; with min_px, not those showing fewer pixels (unknown counts pass)
    out = []
    for lb in labels:
        if lb.object_name != BOT:
            continue
        if min_px > 0.0 and vis_px and vis_px.get(int(lb.object_id), math.inf) < min_px:
            continue
        dist = math.hypot(lb.object_position_x - px, lb.object_position_y - py)
        if dist < 1.0:  # the player's own label, if the env let one through
            continue
        out.append(Enemy(int(lb.object_id), 2.0 * lb.x / screen_w - 1.0,
                         2.0 * (lb.x + lb.width) / screen_w - 1.0, dist, float(lb.width)))
    return out


class DMTeacher:
    def __init__(self, cfg: DMCfg = DM, keep_trace=False):
        self.cfg, self.keep_trace = cfg, keep_trace
        self.nav, self.ring = None, []
        self.reset()

    def reset(self):
        # a new match: the map's items are read again on the first step
        self.spots, self.known, self.used, self.modes = None, [], {}, []
        self.mode, self.goal, self.why = 'patrol', None, 'no_combat'
        self._new_life(None)

    def _new_life(self, me):
        self.tracks, self.shots, self.lock, self.target, self.aim, self.last_seen = {}, {}, None, None, None, None
        self.lead = None  # version 1.1 viewers read it
        self.path, self._path_goal, self._path_age = [], None, 0
        self.unstick, self.unstick_side = 0, STRAFE_LEFT
        self.side, self.side_n = 1, 0
        self.fired_last, self.since_switch = False, SWITCH_STEPS
        self.skip = set()  # item spots without a path, this life
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
        if self.spots is None:  # the map's items: read once per match, never later
            self.spots = [(ITEM_KIND[o.name], o.name, o.x, o.y) for o in priv.objects if o.name in ITEM_KIND]
            self.known = [(k, x, y) for k, _, x, y in self.spots]
        if obs.first:
            self._new_life(me)
        for _, _, x, y in self.spots:
            if math.hypot(x - me[0], y - me[1]) < PICKUP_R:
                self.used[(x, y)] = priv.tic
        bots = visible_bots(priv.labels, priv.px, priv.py, c.width, priv.vis_px, c.min_vis_px)
        seen = {e.id for e in bots}
        self.tracks = {k: v for k, v in self.tracks.items() if k in seen}
        threats, shots = [], {}
        for lb in priv.labels:
            i, x, y = int(lb.object_id), lb.object_position_x, lb.object_position_y
            if i in seen:
                self.tracks.setdefault(i, deque(maxlen=4)).append((priv.tic, x, y))
            elif lb.object_name in PROJECTILES:
                shots[i] = (priv.tic, x, y)
                if i in self.shots and priv.tic > self.shots[i][0]:
                    t0, x0, y0 = self.shots[i]
                    threats.append((x, y, (x - x0) / (priv.tic - t0), (y - y0) / (priv.tic - t0)))
        self.shots = shots
        if bots:
            near = min(bots, key=lambda e: e.dist)
            _, x, y = self.tracks[near.id][-1]
            self.last_seen = (x, y, priv.tic)
        move, turn, fire, weapon = self._decide(priv, me, bots)
        if c.dodge:
            d = dodge_move(threats, priv.px, priv.py, priv.angle)
            if d is not None:
                move = d
        self.fired_last = bool(fire)
        self.since_switch = 0 if weapon else self.since_switch + 1
        if self.keep_trace:
            self.modes.append(MODES.index(self.mode))
        return move, turn, fire, weapon

    # -------------------------------------------------------------- decisions

    def _switch(self, priv, best):
        return int(priv.weapon != best and self.since_switch >= SWITCH_STEPS)

    def _decide(self, priv, me, bots):
        c = self.cfg
        if bots:
            near = min(bots, key=lambda e: e.dist)
            weak = c.weapon_first > 0.0 and near.dist > c.weapon_first and self._gear_spot(priv, me) is not None
            hurt = priv.health < c.retreat_health and self._item_spot(priv, me) is not None
            if (any(s != 1 for s in usable(priv.owned, priv.slot_ammo)) or near.dist < WEAPONS[1].hi) and not weak                     and not hurt:
                self.mode = 'combat'
                return self._combat(priv, me, bots)
        self.target, self.aim, self.why = None, None, 'no_combat'
        weapon = self._switch(priv, best_weapon(priv.owned, priv.slot_ammo, NAV_RANGE))
        if self.last_seen is not None:
            x, y, tic = self.last_seen
            if priv.tic - tic > c.chase_s * TIC_RATE or math.hypot(x - me[0], y - me[1]) < ARRIVE_R:
                self.last_seen = None
        for mode, spot in (('gear', self._gear_spot(priv, me)), ('items', self._item_spot(priv, me))):
            while spot is not None:
                self.mode, self.goal = mode, spot
                a = self._navigate(priv, me)
                if a is not None:
                    return a + (0, weapon)
                self.skip.add(spot)  # no path to it: try the next one
                spot = self._gear_spot(priv, me) if mode == 'gear' else self._item_spot(priv, me)
        if self.last_seen is not None:
            self.mode, self.goal = 'chase', self.last_seen[:2]
            a = self._navigate(priv, me)
            if a is not None:
                return a + (0, weapon)
            self.last_seen = None
        self.mode = 'patrol'
        goal = self.ring[self.ring_i]
        if math.hypot(goal[0] - me[0], goal[1] - me[1]) < ARRIVE_R:
            self.ring_i = (self.ring_i + 1) % len(self.ring)
            goal = self.ring[self.ring_i]
        self.goal = goal
        a = self._navigate(priv, me)
        return (a if a is not None else (STAY, TURN_DEG.index(20.0))) + (0, weapon)

    def _nearest(self, me, tic, names):
        cands = [(x, y) for _, n, x, y in self.spots if n in names and (x, y) not in self.skip
                 and tic - self.used.get((x, y), -ITEM_BACK) >= ITEM_BACK]
        return min(cands, key=lambda p: math.hypot(p[0] - me[0], p[1] - me[1])) if cands else None

    def _gear_spot(self, priv, me):
        # with only weak weapons in hand (rank below gear_rank): the nearest stronger weapon not owned yet
        c = self.cfg
        if max(WEAPONS[s].rank for s in usable(priv.owned, priv.slot_ammo)) >= c.gear_rank:
            return None
        names = {n for n, s in WEAPON_ITEMS.items()
                 if WEAPONS[s].rank >= c.gear_rank and not (s < len(priv.owned) and priv.owned[s])}
        return self._nearest(me, priv.tic, names)

    def _item_spot(self, priv, me):
        # a medikit when hurt; ammo for an owned strong weapon that ran dry
        c = self.cfg
        if priv.health < c.low_health:
            spot = self._nearest(me, priv.tic, {n for n, k in ITEM_KIND.items() if k == 'health'})
            if spot is not None:
                return spot
        if priv.armor < c.armor_below:
            spot = self._nearest(me, priv.tic, {n for n, k in ITEM_KIND.items() if k == 'armor'})
            if spot is not None:
                return spot
        dry = [s for s, w in WEAPONS.items() if w.rank >= c.gear_rank and s < len(priv.owned) and priv.owned[s]
               and priv.slot_ammo[s] < max(w.cost, c.low_ammo)]
        if dry:
            return self._nearest(me, priv.tic, {n for n, s in AMMO_ITEMS.items() if s in dry or s == 0})
        return None

    def _combat(self, priv, me, bots):
        c = self.cfg
        t = pick_target(bots, self.lock)
        self.lock, self.target = t.id, t
        weapon = self._switch(priv, best_weapon(priv.owned, priv.slot_ammo, t.dist))
        w = WEAPONS.get(priv.weapon, WEAPONS[2])
        self.aim = lead_point(self.tracks[t.id], priv.px, priv.py, w.speed)
        rel = target_angle(priv.px, priv.py, priv.angle, *self.aim)
        tol = c.aim_tol * half_width_deg(t.dist)
        turn = pick_turn(self._ahead(priv, t.id) if c.track_ahead else rel, tol)
        safe = t.dist >= w.lo and (w.splash == 0.0 or
                                   self.nav.shot_dist(priv.px, priv.py, priv.angle) >= c.safe_dist)
        if c.barrel_check and self._barrel_in_line(priv):
            safe = False
        # the rocket launcher fires on a new press only; the other weapons repeat while the key is held
        ready = priv.attack_ready or (self.fired_last and priv.weapon != 5)
        fresh = not (self.fired_last and priv.weapon == 5)
        fire_tol = max(tol, c.hitscan_tol) if w.speed == 0.0 else tol
        fire = int(abs(rel) <= fire_tol and safe and ready and fresh and weapon == 0 and priv.weapon_ammo >= w.cost)
        # the first failed condition, for the diagnostics
        self.why = ('shoot' if fire else 'switching' if weapon else 'not_aimed' if abs(rel) > fire_tol else
                    'unsafe' if not safe else 'no_ammo' if priv.weapon_ammo < w.cost else 'reloading')
        fwd = 0
        if t.dist < w.lo + RANGE_MARGIN or (w.splash and t.dist < c.safe_dist + RANGE_MARGIN):
            fwd = -1
        elif t.dist > w.hi:
            fwd = 1
        return SIDE[(fwd, self._strafe_side(priv))], turn, fire, weapon

    def _ahead(self, priv, tid):
        # the angle to the aim point moved by the target's velocity over this step: the turn ends on the target
        track = self.tracks[tid]
        if len(track) < 2 or track[-1][0] == track[0][0]:
            return target_angle(priv.px, priv.py, priv.angle, *self.aim)
        dt = track[-1][0] - track[0][0]
        vx, vy = (track[-1][1] - track[0][1]) / dt, (track[-1][2] - track[0][2]) / dt
        step = self.cfg.frame_skip
        return target_angle(priv.px, priv.py, priv.angle, self.aim[0] + vx * step, self.aim[1] + vy * step)

    def _barrel_in_line(self, priv):
        w = self.cfg.width
        for lb in priv.labels:
            if lb.object_name == BARREL and 2.0 * lb.x / w - 1.0 <= 0.0 <= 2.0 * (lb.x + lb.width) / w - 1.0 and                     math.hypot(lb.object_position_x - priv.px, lb.object_position_y - priv.py) < BARREL_R:
                return True
        return False

    def _strafe_side(self, priv):
        # +1 left, -1 right: strafe_period steps one way, then the other; a wall close on that side flips it
        self.side_n += 1
        room = self.nav.wall_dist(priv.px, priv.py, priv.angle + 90.0 * self.side)
        if self.side_n > self.cfg.strafe_period or room < SIDE_ROOM:
            self.side, self.side_n = -self.side, 1
        return self.side

    def _navigate(self, priv, me):
        # (move, turn) one step along the A* path to self.goal; None when there is no path
        if self.unstick > 0:
            self.unstick -= 1
            return self.unstick_side, 0
        off_path = (not self.path or
                    min(math.hypot(x - me[0], y - me[1]) for x, y in self.path) > ARRIVE_R)
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
            return self.unstick_side, 0
        x, y = lookahead(self.path, me[0], me[1])
        rel = target_angle(priv.px, priv.py, priv.angle, x, y)
        return (FORWARD if abs(rel) < WALK_MAX else STAY), pick_turn(rel, NAV_TOL)
