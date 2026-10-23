from doomagent.dm.vzseed import INT_MAX, first_mt19937, vizdoom_seed


def test_first_mt19937_matches_the_standard():
    assert first_mt19937(5489) == 3499211612  # std::mt19937 with its default seed
    assert first_mt19937(1) == 1791095845


def test_vizdoom_seed_is_valid_distinct_and_stable():
    seeds = list(range(30000, 30300)) + list(range(39000, 39040))
    mapped = [vizdoom_seed(s) for s in seeds]
    assert all(first_mt19937(m) <= INT_MAX for m in mapped)
    assert len(set(mapped)) == len(seeds)
    assert len({first_mt19937(m) for m in mapped}) == len(seeds)
    assert [vizdoom_seed(s) for s in seeds[:5]] == mapped[:5]
