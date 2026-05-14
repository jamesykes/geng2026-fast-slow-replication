from dataclasses import dataclass
import numpy as np


@dataclass
class TwoStatePrisonersDilemma:
    b1: float = 5.0
    b2: float = 1.2
    c1: float = 0.5
    c2: float = 0.5
    p1: float = 0.8
    p2: float = 0.3

    COOPERATE: int = 0
    DEFECT: int = 1
    GOOD_STATE: int = 0
    BAD_STATE: int = 1

    def payoff_matrix(self, state: int) -> np.ndarray:
        """
        Returns the payoff matrix for the row player in the given state.

        Actions:
            0 = cooperate
            1 = defect

        Entry U[a_i, a_j] is the reward to player i when
        player i chooses a_i and player j chooses a_j.
        """
        if state == self.GOOD_STATE:
            b, c = self.b1, self.c1
        elif state == self.BAD_STATE:
            b, c = self.b2, self.c2
        else:
            raise ValueError(f"Unknown state: {state}")

        return np.array([
            [b - c, -c],
            [b,      0.0],
        ])

    def rewards(self, state: int, action_i: int, action_j: int) -> tuple[float, float]:
        """
        Returns rewards for both players.
        """
        U = self.payoff_matrix(state)

        reward_i = U[action_i, action_j]
        reward_j = U[action_j, action_i]

        return reward_i, reward_j

    def next_state(self, state: int, action_i: int, action_j: int, rng: np.random.Generator) -> int:
        """
        Returns the next state after the two agents interact.

        If both cooperate, transition to GOOD_STATE with probability p1.
        Otherwise, transition to GOOD_STATE with probability p2.

        The current state does not directly affect the transition probability
        in the Figure 3 setup.
        """
        both_cooperate = (
            action_i == self.COOPERATE
            and action_j == self.COOPERATE
        )

        prob_good = self.p1 if both_cooperate else self.p2

        if rng.random() < prob_good:
            return self.GOOD_STATE
        else:
            return self.BAD_STATE

rng = np.random.default_rng(seed=0)
game = TwoStatePrisonersDilemma()

# print(game.payoff_matrix(0))
# print(game.payoff_matrix(1))
#
# print(game.rewards(state=0, action_i=0, action_j=1))
# print(game.next_state(state=0, action_i=0, action_j=0, rng=rng))

def make_torus_lattice(width: int, height: int) -> list[tuple[int, int]]:
    """
    Create an undirected 2D torus lattice.

    Nodes are numbered 0, 1, ..., width * height - 1.

    Each node is connected to its right neighbor and down neighbor.
    Because of wraparound, this gives every node degree 4.

    Returns:
        A list of undirected edges (i, j), with i < j.
    """
    edges = set()

    def node_id(x: int, y: int) -> int:
        return y * width + x

    for y in range(height):
        for x in range(width):
            i = node_id(x, y)

            right = node_id((x + 1) % width, y)
            down = node_id(x, (y + 1) % height)

            edges.add(tuple(sorted((i, right))))
            edges.add(tuple(sorted((i, down))))

    return sorted(edges)

edges = make_torus_lattice(width=10, height=10)

# print(len(edges))
# print(edges[:])

def initialize_edge_states(
    num_edges: int,
    rng: np.random.Generator,
    prob_good: float = 0.5,
) -> np.ndarray:
    """
    Initialize each edge state.

    State 0 = good state s1.
    State 1 = bad state s2.
    """
    is_good = rng.random(num_edges) < prob_good
    edge_states = np.where(is_good, 0, 1)
    return edge_states

rng = np.random.default_rng(seed=0)

edges = make_torus_lattice(width=10, height=10)
edge_states = initialize_edge_states(len(edges), rng)

# print(f"Number of agents: {100}")
# print(f"Number of edges: {len(edges)}")
# print(f"Initial fraction in good state: {(edge_states == 0).mean():.3f}")

def initialize_q_values(
    num_agents: int,
    num_states: int = 2,
    num_actions: int = 2,
    initial_value: float = 0.0,
) -> np.ndarray:
    """
    Initialize Q-values.

    Q[i, s, a] is agent i's Q-value for action a in state s.
    """
    return np.full(
        shape=(num_agents, num_states, num_actions),
        fill_value=initial_value,
        dtype=float,
    )


def softmax_policy(Q: np.ndarray, beta: float) -> np.ndarray:
    """
    Convert Q-values into action probabilities using softmax.

    Input:
        Q has shape (num_agents, num_states, num_actions)

    Output:
        policy has the same shape.
        policy[i, s, a] is probability that agent i chooses action a in state s.
    """
    logits = beta * Q

    # Numerical stability trick:
    # subtract max before exponentiating.
    logits = logits - logits.max(axis=-1, keepdims=True)

    exp_logits = np.exp(logits)
    policy = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    return policy

num_agents = 100
Q = initialize_q_values(num_agents)

policy = softmax_policy(Q, beta=1.0)

# print(policy[0])

def sample_actions(
    policy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample one action for each agent in each state.

    Input:
        policy[i, s, a] is the probability of action a.

    Output:
        actions[i, s] is the sampled action.
    """
    num_agents, num_states, num_actions = policy.shape

    actions = np.empty((num_agents, num_states), dtype=int)

    for i in range(num_agents):
        for s in range(num_states):
            actions[i, s] = rng.choice(num_actions, p=policy[i, s])

    return actions

rng = np.random.default_rng(seed=0)

Q = initialize_q_values(num_agents=100)
policy = softmax_policy(Q, beta=1.0)
actions = sample_actions(policy, rng)

# print(actions.shape)
# print(actions[:5])

def simulation_step(
    Q: np.ndarray,
    edge_states: np.ndarray,
    edges: list[tuple[int, int]],
    game: TwoStatePrisonersDilemma,
    alpha: float,
    beta: float,
    gamma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one time step of the multi-agent Q-learning simulation.

    Returns:
        Q: updated Q-values
        edge_states: updated edge states
        policy: policy used during this step
    """
    num_agents, num_states, num_actions = Q.shape

    # 1. Compute policies and sample one action per agent per state.
    policy = softmax_policy(Q, beta)
    actions = sample_actions(policy, rng)

    # We accumulate TD errors here.
    # td_sums[i, s, a] stores total TD error for agent i's Q(s,a).
    # td_counts[i, s, a] stores how many edge interactions contributed.
    td_sums = np.zeros_like(Q)
    td_counts = np.zeros_like(Q)

    # 2. Process each edge interaction.
    for edge_idx, (i, j) in enumerate(edges):
        old_state = edge_states[edge_idx]

        action_i = actions[i, old_state]
        action_j = actions[j, old_state]

        reward_i, reward_j = game.rewards(
            state=old_state,
            action_i=action_i,
            action_j=action_j,
        )

        new_state = game.next_state(
            state=old_state,
            action_i=action_i,
            action_j=action_j,
            rng=rng,
        )

        edge_states[edge_idx] = new_state

        # TD error for i.
        td_i = (
            reward_i
            + gamma * Q[i, new_state].max()
            - Q[i, old_state, action_i]
        )

        # TD error for j.
        td_j = (
            reward_j
            + gamma * Q[j, new_state].max()
            - Q[j, old_state, action_j]
        )

        td_sums[i, old_state, action_i] += td_i
        td_counts[i, old_state, action_i] += 1

        td_sums[j, old_state, action_j] += td_j
        td_counts[j, old_state, action_j] += 1

    # 3. Average TD errors and update Q-values.
    has_update = td_counts > 0
    average_td = np.zeros_like(Q)
    average_td[has_update] = td_sums[has_update] / td_counts[has_update]

    Q[has_update] += alpha * average_td[has_update]

    return Q, edge_states, policy

rng = np.random.default_rng(seed=0)

num_agents = 100
edges = make_torus_lattice(width=10, height=10)
edge_states = initialize_edge_states(len(edges), rng)

game = TwoStatePrisonersDilemma()
Q = initialize_q_values(num_agents)

Q, edge_states, policy = simulation_step(
    Q=Q,
    edge_states=edge_states,
    edges=edges,
    game=game,
    alpha=0.001,
    beta=1.0,
    gamma=0.8,
    rng=rng,
)

# print(Q.shape)
# print(edge_states.shape)
# print(policy.shape)
# print(Q[0])

def summarize_system(
    Q: np.ndarray,
    edge_states: np.ndarray,
    beta: float,
) -> dict[str, float]:
    """
    Compute summary statistics for plotting.
    """
    policy = softmax_policy(Q, beta)

    return {
        "state_0_fraction": float((edge_states == 0).mean()),
        "state_1_fraction": float((edge_states == 1).mean()),
        "coop_prob_state_0": float(policy[:, 0, 0].mean()),
        "coop_prob_state_1": float(policy[:, 1, 0].mean()),
        "defect_prob_state_0": float(policy[:, 0, 1].mean()),
        "defect_prob_state_1": float(policy[:, 1, 1].mean()),
    }


def run_simulation(
    num_steps: int,
    seed: int = 0,
    width: int = 10,
    height: int = 10,
    alpha: float = 0.001,
    beta: float = 1.0,
    gamma: float = 0.8,
    initial_q_value: float = 0.0,
    initial_prob_good: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Run the full agent-based simulation.
    """
    rng = np.random.default_rng(seed)

    num_agents = width * height

    game = TwoStatePrisonersDilemma()
    edges = make_torus_lattice(width, height)
    edge_states = initialize_edge_states(
        num_edges=len(edges),
        rng=rng,
        prob_good=initial_prob_good,
    )
    Q = initialize_q_values(
        num_agents=num_agents,
        initial_value=initial_q_value,
    )

    history = {
        "state_0_fraction": [],
        "state_1_fraction": [],
        "coop_prob_state_0": [],
        "coop_prob_state_1": [],
        "defect_prob_state_0": [],
        "defect_prob_state_1": [],
    }

    # Record the initial condition before learning.
    summary = summarize_system(Q, edge_states, beta)
    for key, value in summary.items():
        history[key].append(value)

    for _ in range(num_steps):
        print(_)
        Q, edge_states, _ = simulation_step(
            Q=Q,
            edge_states=edge_states,
            edges=edges,
            game=game,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rng=rng,
        )

        summary = summarize_system(Q, edge_states, beta)
        for key, value in summary.items():
            history[key].append(value)

    return {
        key: np.array(values)
        for key, values in history.items()
    }

# history = run_simulation(
#     num_steps=20_000,
#     seed=0,
#     alpha=0.001,
#     beta=1.0,
#     gamma=0.8,
# )
#
# print(history["coop_prob_state_0"][:5])
# print(history["coop_prob_state_0"][-5:])
# print(history["coop_prob_state_1"][-5:])

import matplotlib.pyplot as plt


history = run_simulation(
    num_steps=100_000,
    seed=0,
    alpha=0.001,
    beta=1.0,
    gamma=0.8,
)

time = np.arange(len(history["coop_prob_state_0"]))

plt.figure(figsize=(8, 5))

plt.plot(
    time,
    history["coop_prob_state_0"],
    label="Cooperation probability in state s1",
)

plt.plot(
    time,
    history["coop_prob_state_1"],
    label="Cooperation probability in state s2",
)

plt.xlabel("Time step")
plt.ylabel("Mean probability of cooperation")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# plt.figure(figsize=(8, 5))
#
# plt.plot(
#     time,
#     history["state_0_fraction"],
#     label="Fraction of edges in state s1",
# )
#
# plt.plot(
#     time,
#     history["state_1_fraction"],
#     label="Fraction of edges in state s2",
# )
#
# plt.xlabel("Time step")
# plt.ylabel("Fraction of edges")
# plt.ylim(0, 1)
# plt.legend()
# plt.tight_layout()
# plt.show()