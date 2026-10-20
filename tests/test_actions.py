from doomagent.config import (ACTION_NAMES, FINE_LEFT, FINE_RIGHT, N_ACTIONS, SHOOT, TURN_LEFT,
                              TURN_RIGHT)


def test_five_actions_with_fixed_indices():
    assert N_ACTIONS == 5
    assert (TURN_LEFT, TURN_RIGHT, SHOOT, FINE_LEFT, FINE_RIGHT) == (0, 1, 2, 3, 4)
    assert len(ACTION_NAMES) == N_ACTIONS
    assert ACTION_NAMES[FINE_LEFT] == 'fine_left' and ACTION_NAMES[FINE_RIGHT] == 'fine_right'
