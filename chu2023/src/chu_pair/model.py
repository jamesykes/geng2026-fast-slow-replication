"""Authoritative actions, states, payoffs, transitions, and Q update."""

from __future__ import annotations

from enum import IntEnum
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Action(IntEnum):
    C = 0
    D = 1


class State(IntEnum):
    SH = 0
    PD = 1


ACTION_ORDER = (Action.C, Action.D)
STATE_ORDER = (State.SH, State.PD)

# Axes: (state, own row action, opponent column action).
PAYOFF_TENSOR: NDArray[np.float64] = np.array(
    [
        [[1.0, 0.0], [0.1, 0.1]],
        [[1.0, -0.1], [1.2, 0.0]],
    ],
    dtype=np.float64,
)

# Axes: (old state, own action, opponent action).
TRANSITION_TENSOR: NDArray[np.int8] = np.array(
    [
        [[State.SH, State.SH], [State.SH, State.PD]],
        [[State.SH, State.PD], [State.PD, State.PD]],
    ],
    dtype=np.int8,
)

PAYOFF_TENSOR.setflags(write=False)
TRANSITION_TENSOR.setflags(write=False)


def payoff(state: State | int, own_action: Action | int, opponent_action: Action | int) -> float:
    """Return the payoff with the focal/own action in row orientation."""

    return float(PAYOFF_TENSOR[int(State(state)), int(Action(own_action)), int(Action(opponent_action))])


def edge_payoffs(
    state: State | int,
    action_u: Action | int,
    action_v: Action | int,
) -> tuple[float, float]:
    """Return endpoint payoffs, placing each endpoint's action in its own row."""

    payoff_u = payoff(state, action_u, action_v)
    payoff_v = payoff(state, action_v, action_u)
    return payoff_u, payoff_v


def next_state(
    old_state: State | int,
    own_action: Action | int,
    opponent_action: Action | int,
) -> State:
    """Apply the deterministic transition to an old state and joint action."""

    value = TRANSITION_TENSOR[
        int(State(old_state)), int(Action(own_action)), int(Action(opponent_action))
    ]
    return State(int(value))


def selected_coordinate_velocity(
    q: ArrayLike,
    action: Action | int,
    reward: float,
    alpha: float,
) -> float:
    """Return alpha * (reward - Q[action]) without any grid projection."""

    q_array = np.asarray(q, dtype=np.float64)
    if q_array.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q_array.shape}")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not math.isfinite(reward):
        raise ValueError("reward must be finite")
    action_index = int(Action(action))
    return float(q_learning_velocity(q_array[action_index], float(reward), alpha))


def q_learning_velocity(chosen_q, reward, alpha):
    """Backend-neutral selected-coordinate increment ``alpha * (reward - Q)``."""

    return alpha * (reward - chosen_q)


def continuous_selected_update(
    q: ArrayLike,
    action: Action | int,
    reward: float,
    alpha: float,
) -> NDArray[np.float64]:
    """Update only the selected coordinate; never quantise the result."""

    q_array = np.asarray(q, dtype=np.float64)
    if q_array.shape != (2,):
        raise ValueError(f"q must have shape (2,), got {q_array.shape}")
    updated = q_array.copy()
    action_index = int(Action(action))
    updated[action_index] += selected_coordinate_velocity(q_array, action_index, reward, alpha)
    return updated
