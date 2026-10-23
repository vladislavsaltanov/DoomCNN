import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'dm'))

import dm_eval  # noqa: E402


def _res(frags, variant='base'):
    return [dict(variant=variant, seed=s, frags=f, deaths=1.0, rank=1 if f > 5 else 3, shots=10, length=1050,
                 weapon_share={k: 0.0 for k in range(1, 8)}) for s, f in enumerate(frags)]


def test_compare_pairs_with_the_teacher_and_m1_with_m0():
    out = dm_eval.compare({'teacher': _res([10, 12, 8]), 'a_m0': _res([4, 6, 2]), 'b_m1': _res([5, 6, 4])},
                          kinds={'a_m0': 'm0', 'b_m1': 'm1'})
    assert out['models']['a_m0']['share_of_teacher'] == 12 / 30
    assert out['vs_teacher']['a_m0']['mean'] == -6.0
    assert out['m1_vs_m0']['b_m1 - a_m0']['mean'] == 1.0
    assert 'teacher' not in out['vs_teacher']


def test_compare_without_teacher():
    out = dm_eval.compare({'a_m0': _res([1, 2])}, kinds={'a_m0': 'm0'})
    assert out['vs_teacher'] == {} and out['models']['a_m0']['share_of_teacher'] is None
