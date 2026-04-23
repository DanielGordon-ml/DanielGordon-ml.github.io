"""Single-Bit Coordination Dynamics for Pareto-Efficient Outcomes (SBC-PE)
on a 3-agent, 2-action coordination game.

Implements the exploration-phase rule of Kiremitci, Donmez & Sayin (2025),
"Achieving Pareto Optimality in Games via Single-bit Feedback"
(arXiv:2509.25921v2). Each round, every agent plays uniformly at random,
observes its local utility, and broadcasts a one-bit endorsement
    m_i^t = 1  with probability  eps ** (1 - w_i * u_i^t)   if u_i^t > lambda_i
            0  otherwise.
Per-action counters c_i(a_i) advance only on rounds where every agent
endorses (all bits = 1). After K rounds each agent commits to argmax_a c_i(a).

Reproduces the plot used as Figure 2 of the blog post
"One Bit to Coordinate Them All".

Run:  python sbc_pe.py
Out:  sbc-pe-welfare.svg  (same directory as this script)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

# --- Game definition --------------------------------------------------------
# 3 agents, 2 actions each (8 joint profiles).
# Optimum a* = (1, 1, 1) with u_i(a*) = 0.95 -> weighted welfare W(a*) = 0.95.
# All other profiles have u_i = 0.05 -> weighted welfare W = 0.05.
# A wide gap matters: SBC-PE's separation in theta scales as eps^(W(a*) - W(other)),
# so a small welfare gap demands either tiny eps or huge K to sort the counters
# above sampling noise.
# Weights w_i = 1/3 satisfy the paper's assumption  w_i u_i < 1.
# Local thresholds lambda_i = 0 -> every profile is feasible (A_lambda = A).
N_AGENTS = 3
N_ACTIONS = 2
WEIGHTS = np.full(N_AGENTS, 1.0 / N_AGENTS)
THRESHOLD = 0.0
OPTIMUM = (1, 1, 1)
U_OPT = 0.95
U_OTHER = 0.05


def utility(action_profile: tuple[int, ...]) -> np.ndarray:
    """Per-agent utility vector u_i(a) for the given joint action."""
    if action_profile == OPTIMUM:
        return np.full(N_AGENTS, U_OPT)
    return np.full(N_AGENTS, U_OTHER)


@dataclass
class Agent:
    """One SBC-PE agent during the exploration phase."""

    n_actions: int
    epsilon: float
    weight: float
    threshold: float
    counter: np.ndarray = field(init=False)
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.counter = np.zeros(self.n_actions, dtype=np.int64)

    def play(self) -> int:
        return self.rng.randrange(self.n_actions)

    def endorse(self, utility_value: float) -> int:
        """Single-bit endorsement m_i^t (Equation 3)."""
        if utility_value <= self.threshold:
            return 0
        accept_prob = self.epsilon ** (1.0 - self.weight * utility_value)
        return 1 if self.rng.random() < accept_prob else 0

    def increment(self, action: int) -> None:
        self.counter[action] += 1

    def commit(self) -> int:
        return int(np.argmax(self.counter))


def run_simulation(
    epsilon: float, n_steps: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one trial; return (committed_welfare_per_step, at_optimum_per_step)."""
    base_rng = random.Random(seed)
    agents = [
        Agent(
            n_actions=N_ACTIONS,
            epsilon=epsilon,
            weight=float(WEIGHTS[i]),
            threshold=THRESHOLD,
            rng=random.Random(base_rng.random()),
        )
        for i in range(N_AGENTS)
    ]
    welfare = np.empty(n_steps, dtype=float)
    at_opt = np.zeros(n_steps, dtype=bool)
    for t in range(n_steps):
        actions = tuple(agent.play() for agent in agents)
        u_t = utility(actions)
        bits = [agents[i].endorse(float(u_t[i])) for i in range(N_AGENTS)]
        if all(b == 1 for b in bits):
            for i, agent in enumerate(agents):
                agent.increment(actions[i])
        # "If the agents committed right now, what would the welfare be?"
        committed = tuple(agent.commit() for agent in agents)
        welfare[t] = float(utility(committed).mean())
        at_opt[t] = committed == OPTIMUM
    return welfare, at_opt


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    n_steps = 8000
    n_trials = 100
    window = 50

    epsilons = [0.1, 0.2, 0.4]
    colors = {0.1: "#3b82f6", 0.2: "#60a5fa", 0.4: "#94a3b8"}

    # --- Dark style to match the site palette ----------------------------
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#1e293b",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#e2e8f0",
            "axes.titlecolor": "#f8fafc",
            "xtick.color": "#94a3b8",
            "ytick.color": "#94a3b8",
            "grid.color": "#334155",
            "text.color": "#e2e8f0",
            "font.family": "sans-serif",
            "font.size": 11,
        }
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), dpi=120, sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18},
    )

    pareto_welfare = U_OPT
    other_welfare = U_OTHER

    for eps in epsilons:
        trials_w = np.empty((n_trials, n_steps), dtype=float)
        trials_o = np.empty((n_trials, n_steps), dtype=bool)
        for s in range(n_trials):
            w, o = run_simulation(eps, n_steps, seed=s)
            trials_w[s] = w
            trials_o[s] = o
        mean_welfare = trials_w.mean(axis=0)
        opt_fraction = trials_o.mean(axis=0)

        x = np.arange(len(smooth(mean_welfare, window))) + window // 2
        ax_top.plot(x, smooth(mean_welfare, window),
                    label=f"\u03b5 = {eps}", color=colors[eps], linewidth=2)
        ax_bot.plot(x, smooth(opt_fraction, window),
                    label=f"\u03b5 = {eps}", color=colors[eps], linewidth=2)

    ax_top.axhline(pareto_welfare, color="#22c55e", linestyle="--", linewidth=1,
                   label=f"W(a*) = {pareto_welfare}")
    ax_top.axhline(other_welfare, color="#ef4444", linestyle="--", linewidth=1,
                   label=f"W(other) = {other_welfare}")

    ax_top.set_ylabel("Welfare of running argmax (smoothed)")
    ax_top.set_title("SBC-PE on a 3-agent, 2-action coordination game")
    ax_top.set_ylim(0.0, 1.0)
    ax_top.grid(True, alpha=0.4)
    ax_top.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155")

    ax_bot.set_xlabel("Exploration round t")
    ax_bot.set_ylabel("Fraction of trials at a*")
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.grid(True, alpha=0.4)
    ax_bot.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sbc-pe-welfare.svg",
    )
    fig.tight_layout()
    fig.savefig(out_path, format="svg", facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
