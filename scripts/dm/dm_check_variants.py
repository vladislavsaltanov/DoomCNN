"""The teacher on every deathmatch graphics variant; variants it cannot play are switched off.

    python scripts/dm/dm_check_variants.py [--workers 4]

A variant is disabled (disabled, reason in its JSON) if the teacher's frags fall below 0.7 of the
base on CHECK_SEEDS, or if a tex or mix variant changes the network frame by less than 8 on average.
Writes artifacts/dm/e3_variants.png (one 320x240 frame per variant with a bot on screen) and
reports/dm/e3_variants.md.
"""
import argparse
import json
import os

import cv2
import numpy as np

from doomagent.dm.actions import FORWARD, STAY, turn_index
from doomagent.dm.config import BOT, CHECK_SEEDS, DM, VARIANTS_DIR
from doomagent.dm.env import DeathmatchEnv
from doomagent.dm.evaluate import evaluate_plan, summarize
from doomagent.dm.teacher import DMTeacher
from doomagent.variants import BASE, load_variants

DIFF_STEPS = 20
BOT_STEPS = 400  # the snapshot looks for the widest bot in the first 45 s
WALK = ((FORWARD, 0, 0, 0), (STAY, turn_index(20.0), 0, 0))  # forward, forward, turn right 20 degrees


def net_frames(v):
    # the network frames of a fixed walk: same seed, same actions, so only the graphics differ
    env = DeathmatchEnv(DM, v)
    obs, out = env.reset(CHECK_SEEDS[0]), []
    for i in range(DIFF_STEPS):
        obs, _, _ = env.step(WALK[0] if i % 3 else WALK[1])
        out.append(obs.frames[-1].astype(np.int16))
    env.close()
    return np.stack(out)


def bot_frame(v, steps=BOT_STEPS):
    # the frame with the widest bot on screen in the first steps, the teacher playing; frames right
    # after a hit are skipped (the red pain flash covers the screen)
    env, teacher = DeathmatchEnv(DM, v), DMTeacher(DM)
    obs, done, step, best = env.reset(CHECK_SEEDS[0]), False, 0, (-1, None)
    hurt, health = 0, None
    while not done and step < steps:
        p = env.privileged()
        hurt = 6 if health is not None and p.health < health else max(hurt - 1, 0)
        health = p.health
        w = max((lb.width for lb in p.labels if lb.object_name == BOT), default=0)
        if (w > best[0] and not hurt) or best[1] is None:
            best = (w, env.screen().copy())
        obs, _, done = env.step(teacher(obs, p))
        step += 1
    env.close()
    return best[1]


def disable(v, reason):
    path = os.path.join(VARIANTS_DIR, v.name + '.json')
    with open(path) as fh:
        d = json.load(fh)
    d.update(disabled=True, reason=reason)
    with open(path, 'w') as fh:
        json.dump(d, fh, indent=1)


def grid(tiles, out, cols=6):
    rows = []
    for i in range(0, len(tiles), cols):
        row = []
        for name, rgb in tiles[i:i + cols]:
            t = cv2.cvtColor(cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            cv2.rectangle(t, (0, 0), (320, 24), (25, 26, 26), -1)
            cv2.putText(t, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (236, 239, 240), 1, cv2.LINE_AA)
            row.append(t)
        row += [np.full_like(row[0], 25)] * (cols - len(row))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(out, np.concatenate(rows, axis=0))


if __name__ == '__main__':  # evaluation workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    variants = load_variants(root=VARIANTS_DIR)  # variants already disabled are skipped
    plan = [(BASE, CHECK_SEEDS)] + [(v, CHECK_SEEDS) for v in variants]
    res = evaluate_plan(plan, DMTeacher(DM), DM, workers=a.workers)
    by = {name: summarize([r for r in res if r['variant'] == name]) for name in ['base'] + [v.name for v in variants]}
    base = by['base']['frags']['mean']
    base_net = net_frames(BASE)
    tiles, rows = [('base', bot_frame(BASE))], []
    for v in variants:
        f = by[v.name]['frags']
        diff = float(np.abs(net_frames(v) - base_net).mean())
        reason = None
        if f['mean'] < 0.7 * base:
            reason = f'teacher frags {f["mean"]:.2f} < 0.7 of base {base:.2f}'
        elif v.kind in ('tex', 'mix') and diff < 8:
            reason = f'looks like the base graphics (mean frame diff {diff:.1f})'
        if reason:
            disable(v, reason)
        tiles.append((v.name, bot_frame(v)))
        rows.append(f'| {v.name} | {v.split} | {v.kind} | {f["mean"]:.2f} [{f["lo"]:.2f}, {f["hi"]:.2f}] | '
                    f'{f["mean"] / base:.2f} | {by[v.name]["kd_total"]:.2f} | {diff:.1f} | {reason or "ok"} |')
        print(rows[-1], flush=True)
    os.makedirs('artifacts/dm', exist_ok=True)
    grid(tiles, 'artifacts/dm/e3_variants.png')
    b = by['base']
    head = ['# E3: graphics variants of cig MAP02', '',
            f'Teacher on CHECK_SEEDS ({len(CHECK_SEEDS)} matches per variant). Base: frags '
            f'{base:.2f} [{b["frags"]["lo"]:.2f}, {b["frags"]["hi"]:.2f}], K/D {b["kd_total"]:.2f}. '
            'A variant is switched off below 0.7 of the base frags, or (tex, mix) with a mean network-frame '
            f'difference from the base below 8 (first {DIFF_STEPS} steps of a fixed walk).', '',
            '| variant | split | kind | frags [95% CI] | of base | K/D | frame diff | status |',
            '|---|---|---|---|---|---|---|---|']
    ok = sum(r.endswith('| ok |') for r in rows)
    with open('reports/dm/e3_variants.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(head + rows + ['', f'{ok} of {len(rows)} variants kept.',
                                          '', '![variants](../../artifacts/dm/e3_variants.png)', '']))
    print(f'{ok} of {len(rows)} variants kept; written reports/dm/e3_variants.md, artifacts/dm/e3_variants.png')
