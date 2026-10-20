import inspect

import numpy as np
import torch

from doomagent.config import FINE_LEFT, FINE_RIGHT, N_ACTIONS, SHOOT, TURN_RIGHT
from doomagent.model import StudentCNN, bc_loss, seg_target


def make_batch(n, m):
    torch.manual_seed(0)
    return dict(obs=torch.randint(0, 255, (n, 4, 120, 160), dtype=torch.uint8),
                action=torch.randint(0, N_ACTIONS, (n,)), vis=torch.randint(0, 2, (n,)),
                off=torch.rand(n) * 2 - 1, valid=torch.ones(n, dtype=torch.bool),
                seg=torch.zeros(n, *m.seg_hw))


def test_forward_shapes():
    m = StudentCNN()
    out = m(torch.zeros(3, 4, 120, 160, dtype=torch.uint8))
    assert m.seg_hw == (11, 16)
    assert out['logits'].shape == (3, N_ACTIONS) and out['seg'].shape == (3, 11, 16)
    assert out['vis'].shape == (3,) and out['off'].shape == (3,)


def test_uint8_and_float_inputs_agree():
    m = StudentCNN()
    x = torch.randint(0, 255, (2, 4, 120, 160), dtype=torch.uint8)
    assert torch.allclose(m(x)['logits'], m(x.float() / 255.0)['logits'])


def test_act_and_explain_take_only_pixels():
    m = StudentCNN()
    assert list(inspect.signature(m.act).parameters) == ['obs']
    assert list(inspect.signature(m.explain).parameters) == ['obs']
    obs = np.zeros((4, 120, 160), np.uint8)
    info = m.explain(obs)
    assert info['action'] == m.act(obs) and abs(info['probs'].sum() - 1.0) < 1e-5
    assert info['seg'].shape == (11, 16) and 0.0 <= info['vis'] <= 1.0


def test_overfits_tiny_batch():
    m = StudentCNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    b = make_batch(16, m)
    for _ in range(400):
        loss, _ = bc_loss(m(b['obs']), b, shoot_weight=1.0, lam=0.5, lam_seg=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_seg_head_learns_a_bright_square():
    torch.manual_seed(0)
    m = StudentCNN()
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    b = make_batch(8, m)
    b['obs'] = torch.zeros_like(b['obs'])
    mask = torch.zeros(8, 120, 160, dtype=torch.bool)
    for i in range(8):
        y, x = 10 + 10 * i, 15 * i
        b['obs'][i, :, y:y + 20, x:x + 20] = 255
        mask[i, y:y + 20, x:x + 20] = True
    b['seg'] = seg_target(mask, m.seg_hw)
    first = None
    for _ in range(150):
        loss, parts = bc_loss(m(b['obs']), b, lam=0.0, lam_seg=1.0)
        first = parts['seg'] if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert parts['seg'] < 0.5 * first


def test_shoot_weight_changes_loss():
    out = {'logits': torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0, 0.0]]),
           'vis': torch.zeros(2), 'off': torch.zeros(2), 'seg': torch.zeros(2, 11, 16)}
    b = dict(action=torch.tensor([SHOOT, TURN_RIGHT]), vis=torch.zeros(2, dtype=torch.long),
             off=torch.zeros(2), valid=torch.ones(2, dtype=torch.bool),
             seg=torch.zeros(2, 11, 16))
    l1, _ = bc_loss(out, b, shoot_weight=1.0, lam=0.0, lam_seg=0.0)
    l2, _ = bc_loss(out, b, shoot_weight=3.0, lam=0.0, lam_seg=0.0)
    assert not torch.isclose(l1, l2)


def test_invalid_steps_do_not_affect_the_action_loss():
    out = {'logits': torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
           'vis': torch.zeros(2), 'off': torch.zeros(2), 'seg': torch.zeros(2, 11, 16)}
    b = dict(action=torch.tensor([0, 0]), vis=torch.zeros(2, dtype=torch.long),
             off=torch.zeros(2), valid=torch.tensor([True, False]), seg=torch.zeros(2, 11, 16))
    l1, _ = bc_loss(out, b, lam=0.0, lam_seg=0.0)
    b['action'] = torch.tensor([0, 2])
    l2, _ = bc_loss(out, b, lam=0.0, lam_seg=0.0)
    assert torch.isclose(l1, l2)


def test_fine_turn_labels_have_unit_weight():
    out = {'logits': torch.zeros(2, N_ACTIONS), 'vis': torch.zeros(2), 'off': torch.zeros(2),
           'seg': torch.zeros(2, 11, 16)}
    b = dict(action=torch.tensor([FINE_LEFT, FINE_RIGHT]), vis=torch.zeros(2, dtype=torch.long),
             off=torch.zeros(2), valid=torch.ones(2, dtype=torch.bool), seg=torch.zeros(2, 11, 16))
    loss, _ = bc_loss(out, b, shoot_weight=3.0, lam=0.0, lam_seg=0.0)
    assert torch.isclose(loss, torch.log(torch.tensor(float(N_ACTIONS))))
