from types import SimpleNamespace as NS

from doomagent.config import SHOOT, TURN_LEFT, TURN_RIGHT
from doomagent.variants import Variant
from doomagent.evaluate import (bootstrap_ci, evaluate_plan, paired_diff_ci, pixel_only,
                                run_episode, summarize, val_score)


class FakeEnv:
    # ammo[t] is the ammo after t steps
    def __init__(self, rewards, ammo):
        self.rewards, self.ammo, self.t = list(rewards), list(ammo), 0

    def reset(self, seed):
        self.t = 0
        return 'obs'

    def privileged(self):
        return NS(ammo=self.ammo[self.t])

    def step(self, a):
        r = self.rewards[self.t]
        self.t += 1
        return 'obs', r, self.t >= len(self.rewards)

    def close(self):
        pass


def test_run_episode_counts_real_shots_by_ammo():
    env = FakeEnv([0.0, 1.0, 0.0, 1.0, -1.0], [5, 5, 4, 4, 4, 4])
    actions = iter([TURN_LEFT, SHOOT, SHOOT, TURN_RIGHT, SHOOT])
    res = run_episode(env, lambda obs, priv: next(actions), seed=0)
    assert res['reward'] == 1.0 and res['kills'] == 2.0
    assert res['shots'] == 1 and res['pressed'] == 3
    assert res['length'] == 5 and res['first_shot'] == 1


def test_run_episode_resets_stateful_choosers():
    class C:
        resets = 0

        def reset(self):
            C.resets += 1

        def __call__(self, obs, priv):
            return TURN_LEFT

    run_episode(FakeEnv([0.0], [1, 1]), C(), 0)
    assert C.resets == 1


def test_pixel_only_hides_privileged():
    calls = []
    chooser = pixel_only(lambda obs: calls.append(obs) or 0)
    chooser('o', 'SECRET')
    assert calls == ['o']


def test_bootstrap_ci_constant():
    m, lo, hi = bootstrap_ci([5.0] * 10)
    assert m == lo == hi == 5.0


def test_summarize_idle_and_means():
    rs = [dict(reward=2.0, kills=3.0, shots=3, pressed=4, length=10, first_shot=1),
          dict(reward=-1.0, kills=0.0, shots=0, pressed=0, length=10, first_shot=None)]
    s = summarize(rs)
    assert s['idle_rate'] == 0.5 and s['reward']['mean'] == 0.5
    assert s['kills_per_shot'] == 1.0 and s['pressed_per_ep'] == 2.0


def test_paired_diff_ci_sign():
    assert paired_diff_ci([3.0, 4.0, 5.0], [1.0, 2.0, 3.0]) == (2.0, 2.0, 2.0)


def test_evaluate_plan_tags_variants_and_val_score_weights_them_equally():
    rewards = {'a': [1.0], 'b': [3.0]}
    plan = [(NS(name='a'), [0, 1, 2]), (NS(name='b'), [3])]
    res = evaluate_plan(plan, lambda obs, priv: TURN_LEFT,
                        env_factory=lambda cfg, v: FakeEnv(rewards[v.name], [0, 0]))
    assert [r['variant'] for r in res] == ['a', 'a', 'a', 'b']
    score, per = val_score(res)
    assert per == {'a': 1.0, 'b': 3.0} and score == 2.0


class SeedEnv(FakeEnv):
    # module level so worker processes can unpickle it; the reward depends on the seed
    def __init__(self, cfg, variant):
        super().__init__([], [0])
        self.base = 100.0 if variant.name == 'b' else 0.0

    def reset(self, seed):
        self.rewards, self.ammo, self.t = [self.base + seed, 1.0], [0, 0, 0], 0
        return 'obs'


def turn_left(obs, priv):
    return TURN_LEFT


def test_parallel_evaluate_plan_matches_serial():
    plan = [(Variant('a'), tuple(range(7))), (Variant('b'), (40, 41, 42))]
    serial = evaluate_plan(plan, turn_left, env_factory=SeedEnv)
    parallel = evaluate_plan(plan, turn_left, env_factory=SeedEnv, workers=3)
    assert parallel == serial
    assert [(r['variant'], r['seed']) for r in parallel] == [
        ('a', s) for s in range(7)] + [('b', s) for s in (40, 41, 42)]
