from __future__ import annotations

import numpy as np
import pytest

from chu_pair.model import (
    PAYOFF_TENSOR,
    TRANSITION_TENSOR,
    Action,
    State,
    continuous_selected_update,
    edge_payoffs,
    next_state,
    payoff,
)


def test_complete_payoff_truth_table() -> None:
    expected = np.array(
        [
            [[1.0, 0.0], [0.1, 0.1]],
            [[1.0, -0.1], [1.2, 0.0]],
        ]
    )
    np.testing.assert_array_equal(PAYOFF_TENSOR, expected)
    for state in State:
        for own_action in Action:
            for opponent_action in Action:
                assert payoff(state, own_action, opponent_action) == expected[
                    state, own_action, opponent_action
                ]


def test_complete_transition_truth_table() -> None:
    expected = np.array(
        [
            [[State.SH, State.SH], [State.SH, State.PD]],
            [[State.SH, State.PD], [State.PD, State.PD]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(TRANSITION_TENSOR, expected)
    for old_state in State:
        for own_action in Action:
            for opponent_action in Action:
                assert next_state(old_state, own_action, opponent_action) == State(
                    int(expected[old_state, own_action, opponent_action])
                )


def test_payoff_orientation_for_both_endpoints() -> None:
    payoff_u, payoff_v = edge_payoffs(State.PD, Action.C, Action.D)
    assert payoff_u == pytest.approx(-0.1)
    assert payoff_v == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("action", "reward", "expected"),
    [
        (Action.C, 1.0, np.array([0.52, 0.8])),
        (Action.D, 0.0, np.array([0.2, 0.48])),
    ],
)
def test_continuous_update_changes_only_selected_coordinate(
    action: Action,
    reward: float,
    expected: np.ndarray,
) -> None:
    original = np.array([0.2, 0.8])
    updated = continuous_selected_update(original, action, reward, alpha=0.4)
    np.testing.assert_allclose(updated, expected, rtol=0.0, atol=1e-15)
    assert updated[1 - int(action)] == original[1 - int(action)]
    np.testing.assert_array_equal(original, np.array([0.2, 0.8]))


def test_continuous_update_is_not_projected() -> None:
    updated = continuous_selected_update([0.0, 0.0], Action.C, reward=0.123, alpha=0.4)
    assert updated[Action.C] == pytest.approx(0.0492)
    assert updated[Action.D] == 0.0

