import pytest
import torch

from doomagent.dm.config import DM
from doomagent.dm.model import DMNet, bc_loss, gru_gates, load_conv, load_model, save_model

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def batch(B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 255, (B, T, 4, DM.height, DM.width), generator=g, dtype=torch.uint8),
            torch.rand(B, T, 2, generator=g))


def test_forward_shapes():
    f, v = batch()
    logits, h = DMNet(DM, memory=False)(f, v)
    assert logits.shape == (2, 3, 10) and h is None
    logits, h = DMNet(DM, memory=True)(f, v)
    assert logits.shape == (2, 3, 10) and h.shape == (1, 2, 512)


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
    a = torch.randint(0, 10, (B, T), device=DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(300):
        loss, _ = bc_loss(m(f, v)[0], a, torch.ones(B, T, dtype=torch.bool, device=DEV))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_bc_loss_ignores_masked_steps():
    logits = torch.zeros(1, 2, 10)
    logits[0, 0, 3] = 5.0
    mask = torch.tensor([[True, False]])
    l1, _ = bc_loss(logits, torch.tensor([[3, 0]]), mask)
    l2, _ = bc_loss(logits, torch.tensor([[3, 7]]), mask)
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
    x, h = torch.randn(1, 1, 514), torch.randn(1, 1, 512)
    _, ref = m.gru(x, h)
    *_, h_new = gru_gates(m.gru, x[:, 0], h[0])
    assert torch.allclose(h_new, ref[0], atol=1e-5)


def test_checkpoint_roundtrip(tmp_path):
    m = DMNet(DM, memory=True).eval()
    save_model(m, tmp_path / 'm.pt')
    f, v = batch(1, 2)
    assert torch.allclose(m(f, v)[0], load_model(tmp_path / 'm.pt').eval()(f, v)[0])
