import numpy as np
import torch

from doomagent.model import StudentCNN


def _obs(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (4, 120, 160), dtype=np.uint8)


def test_trace_layers_and_contributions_add_up():
    # the logit of each action is the sum of fc activation x weight plus the bias
    from doomagent.brain import trace
    torch.manual_seed(0)
    m = StudentCNN().eval()
    t = trace(m, _obs())
    assert t['conv'][0].shape == (32, 29, 39) and t['conv'][1].shape == (64, 13, 18)
    assert t['conv'][2].shape == (64, 11, 16) and t['fc'].shape == (512,)
    a = t['action']
    assert a == m.act(_obs())
    assert np.isclose(t['contrib'][a].sum() + m.pi.bias[a].item(), t['logits'][a], atol=1e-4)
    assert t['saliency'].shape == (120, 160) and t['saliency'].min() >= 0


def test_render_is_a_fixed_size_image():
    from doomagent.brain import render, trace
    m = StudentCNN().eval()
    img = render(trace(m, _obs()), _obs())
    assert img.shape == (430, 1280, 3) and img.dtype == np.uint8
