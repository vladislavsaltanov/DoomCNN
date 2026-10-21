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

if os.name == 'nt':
    # ViZDoom's IPC waits on the system timer; at the default 15.6 ms a 640x480 step takes
    # ~32 ms instead of ~5.5 ms. Sync mode, so the game itself is unchanged.
    import ctypes
    from ctypes import wintypes

    class _Throttling(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                    ('StateMask', wintypes.ULONG)]

    class _ProcEntry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                    ('th32ProcessID', wintypes.DWORD), ('th32DefaultHeapID', ctypes.c_void_p),
                    ('th32ModuleID', wintypes.DWORD), ('cntThreads', wintypes.DWORD),
                    ('th32ParentProcessID', wintypes.DWORD), ('pcPriClassBase', ctypes.c_long),
                    ('dwFlags', wintypes.DWORD), ('szExeFile', ctypes.c_wchar * 260)]

    _k32 = ctypes.windll.kernel32
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                           wintypes.DWORD]

    def _unthrottle(handle):
        # Windows 11 stops honoring timeBeginPeriod for windowless background processes after
        # a few seconds (steps then take ~100 ms) and may put them on efficiency cores:
        # opt out of both (ProcessPowerThrottling; EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION)
        s = _Throttling(1, 0x1 | 0x4, 0)
        return bool(_k32.SetProcessInformation(handle, 4, ctypes.byref(s), ctypes.sizeof(s)))

    def _child_pids(exe):
        snap, e, out = _k32.CreateToolhelp32Snapshot(0x2, 0), _ProcEntry(), []
        e.dwSize = ctypes.sizeof(e)
        ok = _k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            if e.th32ParentProcessID == os.getpid() and e.szExeFile.lower() == exe:
                out.append(e.th32ProcessID)
            ok = _k32.Process32NextW(snap, ctypes.byref(e))
        _k32.CloseHandle(snap)
        return out

    _unthrottle(_k32.GetCurrentProcess())
    ctypes.windll.winmm.timeBeginPeriod(1)


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
        d, f = DELTA_SIGN * cfg.turn_delta, DELTA_SIGN * cfg.fine_turn_delta
        if cfg.turn_mode == 'binary':
            for b in (vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK):
                g.add_available_button(b)
            g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)  # fine turns only
            self._buttons = ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -f], [0, 0, 0, f])
        elif cfg.turn_mode == 'delta':
            g.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)
            g.add_available_button(vzd.Button.ATTACK)
            self._buttons = ([-d, 0], [d, 0], [0, 1], [-f, 0], [f, 0])
        else:
            raise ValueError(cfg.turn_mode)
        g.clear_available_game_variables()
        for v in (GV.HEALTH, GV.AMMO2, GV.ATTACK_READY, GV.POSITION_X, GV.POSITION_Y,
                  GV.ANGLE):
            g.add_available_game_variable(v)
        g.init()
        self.game = g
        self.unthrottled = False
        if os.name == 'nt':
            for pid in _child_pids('vizdoom.exe'):  # this env's game process (and older ones)
                h = _k32.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
                if h:
                    self.unthrottled = _unthrottle(h) or self.unthrottled
                    _k32.CloseHandle(h)
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

    def step(self, action: int, on_tic=None):
        # on_tic(rgb, privileged) runs after every tic of the frame skip: the viewer's path only
        if on_tic is None:
            r = self.game.make_action(list(self._buttons[action]), self.cfg.frame_skip)
        else:
            r = 0.0
            for _ in range(self.cfg.frame_skip):
                r += self.game.make_action(list(self._buttons[action]), 1)
                if self.game.is_episode_finished():
                    break
                self._read()
                on_tic(self._rgb, self._priv)
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
