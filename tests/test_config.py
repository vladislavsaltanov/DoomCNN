from doomagent.config import (DAGGER_SEEDS, RL_SEEDS, TEST_SEEDS, TRAIN_SEEDS,
                              TRAIN_SEEDS_EXTRA, VAL_SEEDS, VARIANT_SEEDS, Cfg)


def test_seed_sets_are_disjoint():
    sets = [set(s) for s in (TRAIN_SEEDS, TRAIN_SEEDS_EXTRA, VARIANT_SEEDS, DAGGER_SEEDS,
                             RL_SEEDS, VAL_SEEDS, TEST_SEEDS)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j])


def test_seed_set_sizes():
    assert len(TRAIN_SEEDS) == 300 and len(VAL_SEEDS) == 30 and len(TEST_SEEDS) == 100


def test_render_is_an_integer_multiple_of_the_network_input():
    c = Cfg()
    assert c.render_w % c.width == 0 and c.render_h % c.height == 0
    assert c.render_w // c.width == c.render_h // c.height
