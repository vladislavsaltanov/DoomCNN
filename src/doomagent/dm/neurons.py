"""What single neurons of a deathmatch network respond to, from its own matches.

record_activations() plays the student on a plan with a shadow teacher (it labels the teacher's mode
only), keeping the 512 units of every step: the GRU state for M1, the fc features for M0. Then:
place maps (mean activation over the player's position), selectivity to game conditions, and a
t-SNE map of the states colored by the teacher's mode.
"""
import numpy as np

from doomagent.dm.config import DM, MODES
from doomagent.dm.evaluate import STUCK_STEPS
from doomagent.dm.weapons import WEAPONS

CONDITIONS = ('bot_visible', 'bot_recent', 'health_low', 'gear', 'items', 'patrol', 'weapon_strong')


class ShadowChooser:
    # the student plays from its observation alone; the teacher sees (obs, priv) only to name its mode
    def __init__(self, student, teacher):
        self.student, self.teacher, self.modes = student, teacher, []

    def reset(self):
        self.student.reset()
        self.teacher.reset()

    def __call__(self, obs, priv):
        self.teacher(obs, priv)
        self.modes.append(MODES.index(self.teacher.mode))
        return self.student(obs, None)

    def take_trace(self):
        out = self.student.take_trace() if hasattr(self.student, 'take_trace') else {}
        out['modes'] = np.array(self.modes, np.int8)
        self.modes = []
        return out


def _recent(enemy, first, k=STUCK_STEPS):
    # a bot was on screen in the last k steps of this life, but not now
    out, last = np.zeros(len(enemy), bool), -10 ** 9
    for t in range(len(enemy)):
        if first[t]:
            last = -10 ** 9
        if enemy[t]:
            last = t
        else:
            out[t] = t - last <= k
    return out


def activations_from(results, memory):
    # the per-step arrays of evaluate_plan(..., trace=True) results joined over matches
    parts = {k: [] for k in ('act', 'feat', 'px', 'py', 'mode', 'bot_visible', 'bot_recent', 'health_low', 'weapon_strong',
                             'match')}
    for i, r in enumerate(results):
        t, c = r['trace'], r['chooser']
        parts['act'].append(c['h'] if memory else c['feat'])
        parts['feat'].append(c['feat'])
        parts['px'].append(t['px'])
        parts['py'].append(t['py'])
        parts['mode'].append(c['modes'])
        parts['bot_visible'].append(t['enemy'].astype(bool))
        parts['bot_recent'].append(_recent(t['enemy'].astype(bool), t['first'].astype(bool)))
        parts['health_low'].append(t['health'] < DM.low_health)
        parts['weapon_strong'].append(np.array([WEAPONS.get(int(w), WEAPONS[2]).rank >= 3 for w in t['weapon']], bool))
        parts['match'].append(np.full(len(t['px']), i))
    return {k: np.concatenate(v) for k, v in parts.items()}


def record_activations(model_path, plan, cfg=DM, workers=4):
    from doomagent.dm.agent import DMChooser
    from doomagent.dm.evaluate import evaluate_plan
    from doomagent.dm.model import load_model
    from doomagent.dm.teacher import DMTeacher
    m = load_model(model_path, cfg)
    student = DMChooser(m.state_dict(), m.memory, cfg, keep_trace=True)
    res = evaluate_plan(plan, ShadowChooser(student, DMTeacher(cfg)), cfg, workers=workers, trace=True)
    acts = activations_from(res, m.memory)
    np.savez_compressed(str(model_path).replace('.pt', '') + '.acts.npz', **acts)
    return acts


def place_maps(acts, top=8, cell=64.0, min_visits=5):
    # mean activation per cell of the player's position; units ranked by the share of their variance
    # the cells explain (spatial information); cells visited fewer than min_visits times are NaN
    act = np.asarray(acts['act'], np.float32)
    px, py = np.asarray(acts['px']), np.asarray(acts['py'])
    x0, y0 = px.min(), py.min()
    cx, cy = ((px - x0) // cell).astype(int), ((py - y0) // cell).astype(int)
    nx, ny = cx.max() + 1, cy.max() + 1
    idx = cy * nx + cx
    visits = np.bincount(idx, minlength=nx * ny)
    sums = np.zeros((nx * ny, act.shape[1]), np.float64)
    np.add.at(sums, idx, act)
    means = sums / np.maximum(visits, 1)[:, None]
    ok = visits >= min_visits
    keep = ok[idx]
    var = act[keep].var(0)
    between = (visits[ok, None] * (means[ok] - act[keep].mean(0)) ** 2).sum(0) / max(keep.sum(), 1)
    info = np.where(var > 1e-8, between / np.maximum(var, 1e-8), 0.0)
    out = []
    for u in np.argsort(-info)[:top]:
        grid = np.where(ok, means[:, u], np.nan).reshape(ny, nx)
        out.append((int(u), grid, dict(spatial_info=float(info[u]), x0=float(x0), y0=float(y0), cell=cell)))
    return out


def condition_masks(acts):
    mode = np.asarray(acts['mode'])
    return dict(bot_visible=np.asarray(acts['bot_visible'], bool), bot_recent=np.asarray(acts['bot_recent'], bool),
                health_low=np.asarray(acts['health_low'], bool), gear=mode == MODES.index('gear'),
                items=mode == MODES.index('items'), patrol=mode == MODES.index('patrol'),
                weapon_strong=np.asarray(acts['weapon_strong'], bool))


def selectivity(acts, conditions=CONDITIONS):
    # Pearson correlation of every unit with every binary condition
    a = np.asarray(acts['act'], np.float64)
    masks = condition_masks(acts)
    az = (a - a.mean(0)) / np.maximum(a.std(0), 1e-8)
    corr = np.zeros((a.shape[1], len(conditions)))
    for k, name in enumerate(conditions):
        c = masks[name].astype(np.float64)
        if c.std() < 1e-8:
            continue
        corr[:, k] = az.T @ ((c - c.mean()) / c.std()) / len(c)
    strength = np.abs(corr).max(1)
    return np.abs(corr).argmax(1), strength, corr


def state_map(acts, n=3000, seed=0):
    from sklearn.manifold import TSNE
    a, mode = np.asarray(acts['act'], np.float32), np.asarray(acts['mode'])
    g = np.random.default_rng(seed)
    pick = g.choice(len(a), size=min(n, len(a)), replace=False)
    x = a[pick]
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    xy = TSNE(2, perplexity=30, init='pca', random_state=seed).fit_transform(x)
    return xy, mode[pick]
