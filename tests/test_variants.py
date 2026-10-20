import json

from doomagent.config import DAGGER_SEEDS, VAL_OOD_EPISODES, VAL_SEEDS
from doomagent.variants import (BASE, Variant, dagger_plan, load_variants, spread_seeds,
                                val_plan)


def write(d, name, split, **kw):
    (d / f'{name}.json').write_text(json.dumps(dict(name=name, split=split, pwads=[], **kw)))


def test_load_variants_filters_split_and_disabled(tmp_path):
    write(tmp_path, 'a', 'train')
    write(tmp_path, 'b', 'val')
    write(tmp_path, 'c', 'train', disabled=True)
    assert [v.name for v in load_variants(root=str(tmp_path))] == ['a', 'b']
    assert [v.name for v in load_variants('val', str(tmp_path))] == ['b']


def test_spread_seeds_gives_disjoint_chunks():
    plan = spread_seeds([Variant('a'), Variant('b')], tuple(range(10)), 3)
    assert [s for _, s in plan] == [(0, 1, 2), (3, 4, 5)]


def test_val_plan_has_base_and_short_ood(tmp_path):
    write(tmp_path, 'v', 'val')
    plan = val_plan(str(tmp_path))
    assert plan[0] == (BASE, VAL_SEEDS)
    assert plan[1][0].name == 'v' and len(plan[1][1]) == VAL_OOD_EPISODES


def test_dagger_rounds_use_disjoint_seeds(tmp_path):
    write(tmp_path, 'a', 'train')
    r2 = [s for _, ss in dagger_plan(2, 100, str(tmp_path)) for s in ss]
    r3 = [s for _, ss in dagger_plan(3, 100, str(tmp_path)) for s in ss]
    assert len(r2) == len(r3) == 100 and not set(r2) & set(r3)
    assert set(r2) <= set(DAGGER_SEEDS)


def test_apply_sets_iwad_and_files(tmp_path):
    calls = []

    class G:
        def set_doom_game_path(self, p):
            calls.append(('iwad', p))

        def add_game_args(self, a):
            calls.append(('args', a))

    Variant('x', iwad=str(tmp_path / 'i.wad'), pwads=(str(tmp_path / 'p.wad'),)).apply(G())
    assert calls[0][0] == 'iwad' and calls[1][1].startswith('-file ')
