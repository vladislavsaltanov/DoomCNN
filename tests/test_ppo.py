import json

import pytest
import torch


@pytest.mark.vizdoom
def test_gym_env_spaces_and_empty_info():
    from doomagent.ppo import DoomGymEnv
    env = DoomGymEnv((5000,))
    obs, info = env.reset()
    assert obs.shape == (4, 120, 160) and info == {}
    obs, r, term, trunc, info = env.step(0)
    assert info == {} and trunc is False
    env.close()


@pytest.mark.vizdoom
def test_ppo_starts_as_the_student():
    from doomagent.env import DoomEnv
    from doomagent.model import StudentCNN
    from doomagent.ppo import make_ppo, make_vec
    from doomagent.variants import BASE
    torch.manual_seed(0)
    student = StudentCNN()
    venv = make_vec((5000,), (BASE,), 1)
    model = make_ppo(venv, student, lr=1e-5)
    env = DoomEnv()
    obs = env.reset(5000)
    assert int(model.predict(obs, deterministic=True)[0]) == student.act(obs)
    env.close()
    venv.close()


@pytest.mark.vizdoom
def test_finetune_smoke_keeps_a_best_checkpoint(tmp_path):
    from doomagent.model import StudentCNN
    from doomagent.ppo import finetune
    from doomagent.variants import BASE
    torch.save(StudentCNN().state_dict(), tmp_path / 'bc.pt')
    best = finetune(str(tmp_path / 'bc.pt'), str(tmp_path / 'ft'), warmup_steps=512, steps=512,
                    variants=[BASE], n_envs=1, eval_freq=512, val=[(BASE, (5000,))])
    assert best.endswith('best_model.zip') and (tmp_path / 'ft' / 'best_model.zip').exists()
    log = json.loads((tmp_path / 'ft' / 'val_log.json').read_text())
    assert log[0]['timesteps'] == 0 and len(log) >= 2


def test_flip_penalty_only_for_reversals_with_no_enemy():
    # orchestrator decision: penalise left/right reversals while no enemy is on screen
    from doomagent.config import FINE_RIGHT, SHOOT, TURN_LEFT, TURN_RIGHT
    from doomagent.ppo import FlipPenalty
    f = FlipPenalty(0.1)
    assert f(TURN_LEFT, blind=True) == 0.0     # first turn: nothing to reverse
    assert f(FINE_RIGHT, blind=True) == -0.1   # reversal while blind
    assert f(TURN_RIGHT, blind=True) == 0.0    # same direction
    assert f(SHOOT, blind=True) == 0.0
    assert f(TURN_LEFT, blind=False) == 0.0    # an enemy is visible: free to turn back
    assert f(TURN_RIGHT, blind=True) == 0.0    # the visible step reset the direction
    f.reset()
    assert f(TURN_LEFT, blind=True) == 0.0


@pytest.mark.vizdoom
def test_policy_as_student_acts_like_ppo():
    # parallel evaluation runs the PPO actor as a StudentCNN: same layers, same weights
    from doomagent.env import DoomEnv
    from doomagent.model import StudentCNN
    from doomagent.ppo import make_ppo, make_vec, policy_state
    from doomagent.variants import BASE
    torch.manual_seed(1)
    venv = make_vec((5000,), (BASE,), 1)
    model = make_ppo(venv, None, lr=1e-5)
    s = StudentCNN()
    s.load_state_dict(policy_state(model, s.state_dict()))
    env = DoomEnv()
    obs = env.reset(5000)
    for a in (0, 1, 1, 2, 0):
        assert int(model.predict(obs, deterministic=True)[0]) == s.act(obs)
        obs, _, _ = env.step(a)
    env.close()
    venv.close()
