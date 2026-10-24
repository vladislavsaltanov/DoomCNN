import pytest
import torch

import torch.nn.functional as F

from doomagent.dm.actions import HEAD_SIZES, HEADS, N_LOGITS, VEC_DIM
from doomagent.dm.config import DM
from doomagent.dm.model import DMNet, bc_loss, gru_gates, load_conv, load_model, save_model, split_logits

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def batch(B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 255, (B, T, 4, DM.height, DM.width), generator=g, dtype=torch.uint8),
            torch.rand(B, T, VEC_DIM, generator=g))


def test_forward_shapes():
    f, v = batch()
    logits, h = DMNet(DM, memory=False)(f, v)
    assert logits.shape == (2, 3, N_LOGITS) and h is None
    logits, h = DMNet(DM, memory=True)(f, v)
    assert logits.shape == (2, 3, N_LOGITS) and h.shape == (1, 2, 512)


def random_actions(B, T, g=None):
    return torch.stack([torch.randint(0, k, (B, T), generator=g) for k in HEAD_SIZES], -1)


def test_split_logits_sizes():
    parts = split_logits(torch.zeros(2, 3, N_LOGITS))
    assert [p.shape[-1] for p in parts] == list(HEAD_SIZES)


def test_bc_loss_sums_heads():
    torch.manual_seed(0)
    logits, a = torch.randn(2, 3, N_LOGITS), random_actions(2, 3)
    loss, parts = bc_loss(logits, a, torch.ones(2, 3, dtype=torch.bool))
    ref = sum(F.cross_entropy(l.reshape(-1, l.shape[-1]), a[..., k].reshape(-1)) for k, l in enumerate(split_logits(logits)))
    assert torch.isclose(loss, ref, atol=1e-5) and set(f'acc_{h}' for h in HEADS) <= set(parts)


def test_uint8_and_float_inputs_agree():
    f, v = batch()
    m = DMNet(DM, memory=True)
    assert torch.allclose(m(f, v)[0], m(f.float() / 255.0, v)[0], atol=1e-5)


def test_m1_step_equals_sequence():
    torch.manual_seed(0)
    m = DMNet(DM, memory=True).eval()
    f, v = batch(1, 12)
    seq, _ = m(f, v)
    h, steps = None, []
    for t in range(12):
        out, h = m(f[:, t:t + 1], v[:, t:t + 1], h)
        steps.append(out)
    assert torch.allclose(seq, torch.cat(steps, 1), atol=1e-5)


def test_hidden_state_changes_m1_output():
    m = DMNet(DM, memory=True).eval()
    f, v = batch(1, 1)
    assert not torch.allclose(m(f, v)[0], m(f, v, torch.randn(1, 1, 512))[0])


@pytest.mark.parametrize('memory, B, T', [(False, 16, 1), (True, 2, 8)])
def test_overfits_a_tiny_set(memory, B, T):
    torch.manual_seed(0)
    m = DMNet(DM, memory=memory).to(DEV)
    f, v = (x.to(DEV) for x in batch(B, T))
    a = random_actions(B, T).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(300):
        loss, _ = bc_loss(m(f, v)[0], a, torch.ones(B, T, dtype=torch.bool, device=DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, parts = bc_loss(m(f, v)[0], a, torch.ones(B, T, dtype=torch.bool, device=DEV))
    assert loss.item() < 0.2 and parts['acc'] == 1.0


def test_bc_loss_ignores_masked_steps():
    logits = torch.zeros(1, 2, N_LOGITS)
    logits[0, 0, 3] = 5.0
    mask = torch.tensor([[True, False]])
    l1, _ = bc_loss(logits, torch.tensor([[[3, 0, 0, 0], [0, 0, 0, 0]]]), mask)
    l2, _ = bc_loss(logits, torch.tensor([[[3, 0, 0, 0], [7, 9, 1, 1]]]), mask)
    assert torch.isclose(l1, l2)


def test_load_conv_copies_only_the_body(tmp_path):
    from doomagent.model import StudentCNN
    s = StudentCNN()
    torch.save(s.state_dict(), tmp_path / 's.pt')
    m = DMNet(DM, memory=True)
    head = m.head.weight.clone()
    assert load_conv(m, str(tmp_path / 's.pt')) == 6
    assert all(torch.equal(a, b) for a, b in zip(m.body.state_dict().values(), s.body.state_dict().values()))
    assert torch.equal(m.head.weight, head)


def test_gru_gates_reproduce_the_gru():
    m = DMNet(DM, memory=True)
    x, h = torch.randn(1, 1, 512 + VEC_DIM), torch.randn(1, 1, 512)
    _, ref = m.gru(x, h)
    *_, h_new = gru_gates(m.gru, x[:, 0], h[0])
    assert torch.allclose(h_new, ref[0], atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = DMNet(DM, memory=True).eval()
    save_model(m, tmp_path / 'm.pt')
    f, v = batch(1, 2)
    assert torch.allclose(m(f, v)[0], load_model(tmp_path / 'm.pt').eval()(f, v)[0])


def _aux_batch(B, T, mirrored=False):
    from doomagent.dm.auxtargets import AUX_SIZES
    g = torch.Generator().manual_seed(1)
    return dict(aux_enemy=torch.randint(0, 2, (B, T), generator=g).float(),
                aux_x=torch.randint(-1, 8, (B, T), generator=g), aux_pos=torch.randint(-1, 99, (B, T), generator=g),
                aux_head=torch.randint(0, 8, (B, T), generator=g), aux_items=torch.randint(0, 2, (B, T, 4), generator=g).float(),
                aux_items_ok=torch.ones(B, dtype=torch.bool), aux_depth_ok=torch.ones(B, dtype=torch.bool),
                aux_depth=torch.rand(B, T, AUX_SIZES['depth'], generator=g))


@pytest.mark.parametrize('memory', [False, True])
def test_aux_heads_shapes_and_loss(memory):
    from doomagent.dm.auxtargets import AUX_SIZES
    from doomagent.dm.model import aux_loss
    f, v = batch()
    net = DMNet(DM, memory=memory, aux=True)
    logits, h, aux = net.forward_aux(f, v)
    assert logits.shape == (2, 3, N_LOGITS) and {k: a.shape[-1] for k, a in aux.items()} == AUX_SIZES
    b = _aux_batch(2, 3)
    loss, parts = aux_loss(aux, b, torch.ones(2, 3, dtype=torch.bool))
    assert loss.item() > 0 and {'aux_enemy_acc', 'aux_x_acc', 'aux_pos_acc', 'aux_head_acc', 'aux_depth_mse'} <= set(parts)
    loss.backward()
    assert net.aux.weight.grad is not None


def test_aux_loss_skips_missing_targets():
    from doomagent.dm.model import aux_loss
    aux = {k: torch.zeros(1, 2, n) for k, n in __import__('doomagent.dm.auxtargets', fromlist=['AUX_SIZES']).AUX_SIZES.items()}
    b = _aux_batch(1, 2)
    b.update(aux_x=torch.full((1, 2), -1), aux_pos=torch.full((1, 2), -1), aux_head=torch.full((1, 2), -1),
             aux_items_ok=torch.zeros(1, dtype=torch.bool), aux_depth_ok=torch.zeros(1, dtype=torch.bool))
    b['aux_enemy'] = torch.full((1, 2), 0.5)  # BCE on logit 0 -> log 2 only from the enemy head
    loss, _ = aux_loss(aux, b, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() == pytest.approx(float(torch.log(torch.tensor(2.0))), rel=1e-4)


def test_save_load_keeps_the_aux_heads(tmp_path):
    from doomagent.dm.agent import DMChooser
    net = DMNet(DM, memory=True, aux=True)
    save_model(net, str(tmp_path / 'a.pt'))
    m = load_model(str(tmp_path / 'a.pt'))
    assert m.aux_on and m.memory
    DMChooser(net.state_dict(), True)  # builds the same net from a state with aux heads


def _sp_batch(B, T):
    from doomagent.dm.spatial import FUT_K, GOAL_BINS, MAP_N
    g = torch.Generator().manual_seed(1)
    return dict(sp_map=torch.randint(0, 2, (B, T, MAP_N * MAP_N), generator=g).float(),
                sp_map_ok=torch.ones(B, T, dtype=torch.bool), sp_odo=torch.randn(B, T, 6, generator=g),
                sp_odo_ok=torch.ones(B, T, 2, dtype=torch.bool), sp_fut=torch.randn(B, T, 3, generator=g),
                sp_fut_ok=torch.ones(B, T, dtype=torch.bool), sp_fut_depth=torch.rand(B, T, 128, generator=g),
                sp_fut_depth_ok=torch.ones(B, T, dtype=torch.bool),
                sp_fut_move=torch.randint(0, 9, (B, T, FUT_K), generator=g),
                sp_fut_turn=torch.randint(0, 15, (B, T, FUT_K), generator=g),
                sp_goal_dir=torch.randint(-1, GOAL_BINS, (B, T), generator=g),
                sp_goal_dist=torch.randint(-1, 4, (B, T), generator=g), sp_mode=torch.randint(0, 5, (B, T), generator=g))


@pytest.mark.parametrize('memory', [False, True])
def test_spatial_heads_shapes_and_loss(memory):
    from doomagent.dm.model import fut_input, spatial_loss
    from doomagent.dm.spatial import FUT_OUT, SPATIAL_SIZES
    f, v = torch.zeros(2, 3, DM.stack, DM.height, DM.width), torch.zeros(2, 3, VEC_DIM)
    net = DMNet(DM, memory=memory, spatial=True)
    b = _sp_batch(2, 3)
    logits, h, out = net.forward_aux(f, v, fut=fut_input(b['sp_fut_move'], b['sp_fut_turn']))
    assert {k: out[k].shape[-1] for k in SPATIAL_SIZES} == SPATIAL_SIZES and out['fwd'].shape == (2, 3, FUT_OUT)
    loss, parts = spatial_loss(out, b, torch.ones(2, 3, dtype=torch.bool))
    assert loss.item() > 0
    assert {'sp_map_acc', 'sp_odo_err', 'sp_fut_err', 'sp_fut_depth_mse', 'sp_goal_dir_acc', 'sp_goal_dist_acc',
            'sp_mode_acc', 'spatial'} <= set(parts)
    loss.backward()
    assert net.spatial.weight.grad is not None and net.fwd[0].weight.grad is not None
    assert net.forward_aux(f, v)[2].keys() == SPATIAL_SIZES.keys()  # no forward model without the actions


def test_spatial_loss_skips_missing_targets():
    from doomagent.dm.model import spatial_loss
    from doomagent.dm.spatial import FUT_OUT, SPATIAL_SIZES
    out = {k: torch.zeros(1, 2, n) for k, n in SPATIAL_SIZES.items()}
    out['fwd'] = torch.zeros(1, 2, FUT_OUT)
    b = _sp_batch(1, 2)
    off = torch.zeros(1, 2, dtype=torch.bool)
    b.update(sp_map_ok=off, sp_odo_ok=torch.zeros(1, 2, 2, dtype=torch.bool), sp_fut_ok=off, sp_fut_depth_ok=off,
             sp_goal_dir=torch.full((1, 2), -1), sp_goal_dist=torch.full((1, 2), -1))
    b['sp_mode'] = torch.zeros(1, 2, dtype=torch.long)
    loss, parts = spatial_loss(out, b, torch.ones(1, 2, dtype=torch.bool))
    assert loss.item() == pytest.approx(__import__('math').log(5), rel=1e-4)  # only the mode head, uniform logits
    assert 'sp_map_acc' not in parts and 'sp_fut_err' not in parts


def test_fut_input_is_one_hot_and_ignores_padding():
    from doomagent.dm.model import fut_input
    from doomagent.dm.spatial import FUT_IN, FUT_K
    x = fut_input(torch.tensor([[1, 2, 3, 4], [-1] * FUT_K]), torch.tensor([[0, 1, 2, 3], [-1] * FUT_K]))
    assert x.shape == (2, FUT_IN) and x[0].sum() == 2 * FUT_K and x[1].sum() == 0


def test_save_load_keeps_the_spatial_heads(tmp_path):
    from doomagent.dm.agent import DMChooser
    net = DMNet(DM, memory=False, aux=True, spatial=True)
    save_model(net, tmp_path / 'm.pt')
    m = load_model(tmp_path / 'm.pt')
    assert m.spatial_on and m.aux_on and not m.memory
    import numpy as np
    from doomagent.dm.env import DMObs
    obs = DMObs(np.zeros((DM.stack, DM.height, DM.width), np.uint8), np.zeros(VEC_DIM, np.float32), True)
    assert len(DMChooser(net.state_dict(), False)(obs, None)) == 4  # builds the net from a state with them
