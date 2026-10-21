import pytest

from doomagent.config import VAL_SEEDS
from doomagent.evaluate import evaluate_plan, summarize
from doomagent.teacher import Teacher
from doomagent.variants import BASE

pytestmark = pytest.mark.vizdoom


def test_teacher_plays_sanely():
    res = evaluate_plan([(BASE, VAL_SEEDS[:3])], Teacher())
    s = summarize(res)
    assert s['reward']['mean'] > 10.0
    assert all(r['shots'] > 0 for r in res)
    assert s['kills_per_shot'] > 0.5
