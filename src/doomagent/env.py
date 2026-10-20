import os
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import vizdoom as vzd

from doomagent.config import Cfg
from doomagent.variants import BASE, Variant

GV = vzd.GameVariable
RESOLUTIONS = {(640, 480): vzd.ScreenResolution.RES_640X480,
               (800, 600): vzd.ScreenResolution.RES_800X600,
               (320, 240): vzd.ScreenResolution.RES_320X240}
DELTA_SIGN = 1.0  # flip to -1.0 if test_turn_directions[delta] shows that TURN_LEFT turns right


@dataclass(frozen=True)
class Label:
    # a labels-buffer entry with the box scaled to network coordinates
    value: int
    object_id: int
    object_name: str
    x: float
    y: float
    width: float
    height: float
    object_position_x: float
    object_position_y: float


@dataclass
class Privileged:
    labels: list = field(default_factory=list)
    labels_buffer: np.ndarray = None  # (render_h, render_w) uint8, full resolution
    attack_ready: bool = False
    health: float = 0.0
    ammo: float = 0.0
    px: float = 0.0
    py: float = 0.0
    angle: float = 0.0


def to_net(rgb, cfg: Cfg):
    # the only downscale: recording, training, PPO, evaluation and demo all go through it
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)


class DoomEnv:
    # agent-facing API returns pixels only; privileged data and the full frame are separate
    def __init__(self, cfg: Cfg = Cfg(), variant: Variant = BASE, visible: bool = False):
        self.cfg, self.variant = cfg, variant
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, 'defend_the_center.cfg'))
        variant.apply(g)
        g.set_window_visible(visible)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_screen_resolution(RESOLUTIONS[(cfg.render_w, cfg.render_h)])
        g.set_render_hud(cfg.hud)
        g.set_render_crosshair(False)
        g.set_labels_buffer_enabled(True)
        g.set_depth_buffer_enabled(False)
        g.clear_available_buttons()
        if cfg.turn_mode == 'binary':
            for b in (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK):
                g.add_available_button(b)
            self._buttons = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
        elif cfg.turn_mode == 'delta':
            g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)
            g.add_available_button(vzd.Button.ATTACK)
            d = DELTA_SIGN * cfg.turn_delta
            self._buttons = ([-d, 0], [d, 0], [0, 1])
        else:
            raise ValueError(cfg.turn_mode)
        g.clear_available_game_variables()
        for v in (GV.HEALTH, GV.AMMO2, GV.ATTACK_READY, GV.POSITION_X, GV.POSITION_Y,
                  GV.ANGLE):
            g.add_available_game_variable(v)
        g.init()
        self.game = g
        self._frames = deque(maxlen=cfg.stack)
        self._rgb = np.zeros((cfg.render_h, cfg.render_w, 3), np.uint8)
        self._small = np.zeros((cfg.height, cfg.width), np.uint8)
        self._priv = Privileged()

    def _read(self):
        st = self.game.get_state()
        if st is None:  # terminal: keep the last frame, privileged data becomes empty
            self._priv = Privileged()
            return
        c = self.cfg
        self._rgb = st.screen_buffer
        self._small = to_net(self._rgb, c)
        sx, sy = c.width / c.render_w, c.height / c.render_h
        labels = [Label(int(lb.value), int(lb.object_id), lb.object_name, lb.x * sx, lb.y * sy,
                        lb.width * sx, lb.height * sy, lb.object_position_x,
                        lb.object_position_y) for lb in st.labels]
        g = self.game.get_game_variable
        self._priv = Privileged(labels, st.labels_buffer, bool(g(GV.ATTACK_READY)),
                                g(GV.HEALTH), g(GV.AMMO2), g(GV.POSITION_X), g(GV.POSITION_Y),
                                g(GV.ANGLE))

    @property
    def obs(self):
        return np.stack(self._frames)

    def reset(self, seed: int):
        self.game.set_seed(int(seed))
        self.game.new_episode()
        self._read()
        self._frames.clear()
        for _ in range(self.cfg.stack):
            self._frames.append(self._small)
        return self.obs

    def step(self, action: int):
        r = self.game.make_action(list(self._buttons[action]), self.cfg.frame_skip)
        done = bool(self.game.is_episode_finished())
        self._read()
        self._frames.append(self._small)
        return self.obs, float(r), done

    def privileged(self) -> Privileged:
        return self._priv

    def screen(self):
        # full RGB frame for the demo and videos only; never passed to the agent
        return self._rgb

    def close(self):
        self.game.close()
