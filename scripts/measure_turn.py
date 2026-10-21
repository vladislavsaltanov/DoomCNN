import math

from doomagent.config import FINE_LEFT, FINE_RIGHT, TURN_LEFT, TURN_RIGHT, Cfg
from doomagent.env import DoomEnv


def deltas(env, actions):
    out = []
    for a in actions:
        a0 = env.privileged().angle
        env.step(a)
        out.append(round((env.privileged().angle - a0 + 180.0) % 360.0 - 180.0, 2))
    return out


for mode in ('binary', 'delta'):
    cfg = Cfg(turn_mode=mode)
    env = DoomEnv(cfg)
    env.reset(5000)
    hold = deltas(env, [TURN_LEFT] * 8)
    alternate = deltas(env, [TURN_LEFT, TURN_RIGHT] * 4)
    fine_hold = deltas(env, [FINE_LEFT] * 8)
    fine_alternate = deltas(env, [FINE_LEFT, FINE_RIGHT] * 4)
    env.close()
    worst = max(abs(d) for d in hold + alternate)
    px = math.tan(math.radians(worst)) * cfg.width / 2
    fine_worst = max(abs(d) for d in fine_hold + fine_alternate)
    fine_px = math.tan(math.radians(fine_worst)) * cfg.width / 2
    print(f'{mode}: hold left {hold}')
    print(f'{mode}: alternate {alternate}')
    print(f'{mode}: largest step {worst:.1f} deg = {px:.1f} px at {cfg.width} px width; '
          f'targets narrower than about {px:.0f} px can be stepped over')
    print(f'{mode}: fine hold left {fine_hold}')
    print(f'{mode}: fine alternate {fine_alternate}')
    print(f'{mode}: largest fine step {fine_worst:.1f} deg = {fine_px:.1f} px at '
          f'{cfg.width} px width')
