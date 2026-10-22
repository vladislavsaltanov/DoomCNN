import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import vizdoom as vzd

from doomagent.dm.config import AMMO_SCALE, BOT, BOT_NAMES, DM, HEALTH_SCALE, DMCfg
from doomagent.env import DELTA_SIGN, RESOLUTIONS, Label, to_net
from doomagent.variants import BASE, Variant
from doomagent.vzseed import vizdoom_seed

GV = vzd.GameVariable
BOTS_CFG = os.path.join(os.path.dirname(vzd.__file__), 'bots.cfg')
GAME_ARGS = ('-host 1 -deathmatch +timelimit {timelimit} +sv_forcerespawn 1 +sv_noautoaim 1 '
             '+sv_respawnprotect 1 +sv_spawnfarthest 1 +viz_respawn_delay 0 '
             '+viz_bots_path {bots}')
BUTTONS = (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
           vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK)  # + TURN_LEFT_RIGHT_DELTA
VARS = (GV.HEALTH, GV.ARMOR, GV.AMMO5, GV.ATTACK_READY, GV.POSITION_X, GV.POSITION_Y, GV.ANGLE,
        GV.FRAGCOUNT, GV.DEATHCOUNT, GV.HITCOUNT, GV.DAMAGECOUNT, GV.ITEMCOUNT, GV.PLAYER_COUNT)
MAX_DEAD_TICS = 70  # the dead state lasts a tic or two with sv_forcerespawn; a guard only
SELF_DIST = 1.0     # the player's own DoomPlayer label sits exactly at its position


def unthrottle_children():
    # Windows: opt every ViZDoom process this one started (a new one per match) out of power
    # throttling, with the first section's helpers; importing doomagent.env already set the 1 ms
    # timer and unthrottled this process
    if os.name != 'nt':
        return False
    from doomagent import env as base
    done = False
    for pid in base._child_pids('vizdoom.exe'):
        h = base._k32.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
        if h:
            done = base._unthrottle(h) or done
            base._k32.CloseHandle(h)
    return done


def action_buttons(cfg: DMCfg):
    # one button vector per action in dm.config order; fine turns go through the delta button
    f = DELTA_SIGN * cfg.fine_turn_delta
    return ([1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, -f],
            [0, 0, 0, 0, 0, 0, f], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0])


@dataclass(frozen=True)
class DMObs:
    frames: np.ndarray  # (stack, 120, 160) uint8, oldest..newest
    vec: np.ndarray     # (2,) float32: health / 100, rockets / 50
    first: bool         # first step of a life: match start or just after a respawn


@dataclass(frozen=True)
class MapObject:
    id: int
    name: str
    x: float
    y: float
    angle: float


@dataclass
class DMPrivileged:
    labels: list = field(default_factory=list)   # net coordinates, the player's own label removed
    objects: list = field(default_factory=list)  # every map object but the player: viewer, evaluator
    sectors: list = None                         # the static map, same list every step
    px: float = 0.0
    py: float = 0.0
    angle: float = 0.0
    health: float = 0.0
    armor: float = 0.0
    ammo: float = 0.0
    attack_ready: bool = False
    frags: float = 0.0   # match counters
    deaths: float = 0.0
    hits: float = 0.0
    damage: float = 0.0
    items: float = 0.0
    players: int = 0
    player_frags: tuple = ()  # PLAYER1..N_FRAGCOUNT, the agent first
    tic: int = 0


def _is_self(name, x, y, px, py):
    return name == BOT and math.hypot(x - px, y - py) < SELF_DIST


class DeathmatchEnv:
    # cig.wad MAP01 against named ViZDoom bots. In host mode a match repeats bit for bit only if
    # the seed is set before init() and the match is the game's first (probed), so every reset
    # starts a fresh game. The agent-facing API returns DMObs; privileged data is separate.
    def __init__(self, cfg: DMCfg = DM, variant: Variant = BASE, visible: bool = False):
        self.cfg, self.variant, self.visible = cfg, variant, visible
        self.game, self.unthrottled, self.sectors = None, False, []
        self._buttons, self._noop = action_buttons(cfg), [0] * 7
        self._vars = VARS + tuple(getattr(GV, f'PLAYER{i}_FRAGCOUNT') for i in range(1, cfg.bots + 2))
        self._frames = deque(maxlen=cfg.stack)
        self._rgb = np.zeros((cfg.render_h, cfg.render_w, 3), np.uint8)
        self._small = np.zeros((cfg.height, cfg.width), np.uint8)
        self._priv = DMPrivileged()

    def _new_game(self, seed):
        if self.game is not None:
            self.game.close()
        c = self.cfg
        assert ' ' not in BOTS_CFG, f'ViZDoom splits game args on spaces: {BOTS_CFG}'
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, 'cig.cfg'))
        g.set_doom_map('map01')
        g.add_game_args(GAME_ARGS.format(timelimit=c.timelimit, bots=BOTS_CFG))
        self.variant.apply(g)
        g.set_mode(vzd.Mode.PLAYER)
        g.set_window_visible(self.visible)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(RESOLUTIONS[(c.render_w, c.render_h)])
        g.set_render_hud(c.hud)
        g.set_render_crosshair(False)
        g.set_labels_buffer_enabled(True)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_depth_buffer_enabled(False)
        g.clear_available_buttons()
        for b in BUTTONS:
            g.add_available_button(b)
        g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)  # fine turns only
        g.clear_available_game_variables()
        for v in self._vars:
            g.add_available_game_variable(v)
        g.set_seed(vizdoom_seed(int(seed)))  # before init(); half of raw seeds would play one match
        g.init()
        self.game = g
        self.unthrottled = unthrottle_children()
        for name in BOT_NAMES[:c.bots]:
            g.send_game_command('addbot ' + name)
        g.make_action(self._noop, 1)  # the bots join on this tic

    def _read(self):
        g, c = self.game, self.cfg
        st = g.get_state()
        if st is None:  # match over: keep the last frame and labels, refresh the counters
            values = [g.get_game_variable(v) for v in self._vars]
        else:
            values = list(st.game_variables)
            self._rgb = st.screen_buffer
            self._small = to_net(self._rgb, c)
        (health, armor, ammo, ready, px, py, angle, frags, deaths, hits, damage, items,
         players) = values[:len(VARS)]
        p = self._priv
        labels, objects = p.labels, p.objects
        if st is not None:
            sx, sy = c.width / c.render_w, c.height / c.render_h
            labels = [Label(int(lb.value), int(lb.object_id), lb.object_name, lb.x * sx, lb.y * sy,
                            lb.width * sx, lb.height * sy, lb.object_position_x, lb.object_position_y)
                      for lb in st.labels
                      if not _is_self(lb.object_name, lb.object_position_x, lb.object_position_y, px, py)]
            objects = [MapObject(int(o.id), o.name, o.position_x, o.position_y, o.angle)
                       for o in st.objects if not _is_self(o.name, o.position_x, o.position_y, px, py)]
        self._priv = DMPrivileged(labels, objects, self.sectors, px, py, angle, health, armor, ammo,
                                  bool(ready), frags, deaths, hits, damage, items, int(players),
                                  tuple(values[len(VARS):]), int(g.get_episode_time()))

    def _obs(self, first):
        p = self._priv
        vec = np.array([p.health / HEALTH_SCALE, p.ammo / AMMO_SCALE], np.float32)
        return DMObs(np.stack(self._frames), vec, bool(first))

    def _restack(self):
        self._frames.clear()
        for _ in range(self.cfg.stack):
            self._frames.append(self._small)

    def _skip_death(self, on_tic=None):
        # the agent never acts on the death screen: respawn and tic until alive
        g = self.game
        if g.is_episode_finished() or not g.is_player_dead():
            return False
        g.respawn_player()
        for _ in range(MAX_DEAD_TICS):
            if g.is_episode_finished() or not g.is_player_dead():
                break
            g.make_action(self._noop, 1)
            if on_tic is not None and not g.is_episode_finished():
                self._read()
                on_tic(self._rgb, self._priv)
        return True

    def reset(self, seed: int) -> DMObs:
        self._new_game(seed)
        self._skip_death()  # a bot that joins on the player's spawn spot telefrags it
        st = self.game.get_state()
        self.sectors = [(float(s.floor_height), float(s.ceiling_height),
                         tuple((float(ln.x1), float(ln.y1), float(ln.x2), float(ln.y2),
                                bool(ln.is_blocking)) for ln in s.lines)) for s in st.sectors]
        self._read()
        self._restack()
        return self._obs(True)

    def step(self, action: int, on_tic=None):
        # on_tic(rgb, privileged) runs after every tic, the death skip included: the viewer's path
        g, c = self.game, self.cfg
        frags0, deaths0 = self._priv.frags, self._priv.deaths
        buttons = list(self._buttons[action])
        if on_tic is None:
            g.make_action(buttons, c.frame_skip)
        else:
            for _ in range(c.frame_skip):
                g.make_action(buttons, 1)
                if g.is_episode_finished():
                    break
                self._read()
                on_tic(self._rgb, self._priv)
        died = self._skip_death(on_tic)
        done = bool(g.is_episode_finished())
        self._read()
        # a death and the forced respawn can both fall inside one frame-skip step
        new_life = (died or self._priv.deaths > deaths0) and not done
        if new_life:
            self._restack()
        else:
            self._frames.append(self._small)
        return self._obs(new_life), float(self._priv.frags - frags0), done

    def privileged(self) -> DMPrivileged:
        return self._priv

    def screen(self):
        # full RGB frame for the demo and videos only; never passed to the agent
        return self._rgb

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
