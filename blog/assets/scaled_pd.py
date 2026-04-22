"""Scaled Prisoner's Dilemma under the Marden-Young-Pao (2014)
completely uncoupled payoff-based learning rule.

Reproduces the result used as Figure 2 of the blog post
"Pareto Optimality Without Communication".

Run:  python scaled_pd.py
Out:  welfare-plot.svg  (same directory as this script)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

# --- Scaled Prisoner's Dilemma payoff matrix -------------------------------
# Rows = action of agent 0, cols = action of agent 1.
# Action 0 = "A" (cooperate), Action 1 = "B" (defect).
# Payoffs taken from the post:
#   A,A -> (3/4, 3/4)      <- Pareto-optimal
#   A,B -> (0,   4/5)
#   B,A -> (4/5, 0)
#   B,B -> (1/3, 1/3)      <- unique pure Nash
PAYOFFS = np.array(
    [
        [[3 / 4, 3 / 4], [0.0, 4 / 5]],
        [[4 / 5, 0.0], [1 / 3, 1 / 3]],
    ]
)
N_AGENTS = 2
N_ACTIONS = 2


@dataclass
class Agent:
    """One agent running the content/discontent learning rule."""

    n_actions: int
    epsilon: float
    c: int
    benchmark_action: int = 0
    benchmark_payoff: float = 0.0
    mood: str = "C"  # "C" (content) or "D" (discontent)
    rng: random.Random = field(default_factory=random.Random)

    def choose_action(self) -> int:
        if self.mood == "D":
            return self.rng.randrange(self.n_actions)
        # Content: play benchmark with prob 1 - eps^c, else uniform over others.
        if self.rng.random() < 1.0 - self.epsilon ** self.c:
            return self.benchmark_action
        others = [a for a in range(self.n_actions) if a != self.benchmark_action]
        return self.rng.choice(others)

    def update(self, action: int, payoff: float) -> None:
        matches_benchmark = (
            action == self.benchmark_action
            and abs(payoff - self.benchmark_payoff) < 1e-12
        )
        if self.mood == "C" and matches_benchmark:
            return  # stay content, benchmark unchanged
        # Mismatch: set new benchmark, choose new mood probabilistically.
        self.benchmark_action = action
        self.benchmark_payoff = payoff
        accept_prob = self.epsilon ** (1.0 - payoff)
        self.mood = "C" if self.rng.random() < accept_prob else "D"


def run_simulation(epsilon: float, n_steps: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    agents = [
        Agent(
            n_actions=N_ACTIONS,
            epsilon=epsilon,
            c=N_AGENTS,  # c >= n per the paper; smallest valid choice.
            rng=random.Random(rng.random()),
        )
        for _ in range(N_AGENTS)
    ]
    welfare = np.empty(n_steps, dtype=float)
    for t in range(n_steps):
        actions = [agent.choose_action() for agent in agents]
        payoffs = PAYOFFS[actions[0], actions[1]]  # shape (2,)
        for i, agent in enumerate(agents):
            agent.update(actions[i], float(payoffs[i]))
        welfare[t] = float(payoffs.sum())
    return welfare


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    n_steps = 5000
    n_trials = 30
    window = 200

    epsilons = [0.05, 0.01]
    colors = {0.05: "#60a5fa", 0.01: "#3b82f6"}

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

    pareto_welfare = 2 * (3 / 4)  # joint welfare at (A, A)

    for eps in epsilons:
        trials = np.stack(
            [run_simulation(eps, n_steps, seed=s) for s in range(n_trials)]
        )
        mean_welfare = trials.mean(axis=0)
        # (A, A) is the only joint action with total welfare == 1.5 in this PD.
        pareto_fraction = (trials == pareto_welfare).mean(axis=0)

        x = np.arange(len(smooth(mean_welfare, window))) + window // 2
        ax_top.plot(x, smooth(mean_welfare, window),
                    label=f"ε = {eps}", color=colors[eps], linewidth=2)
        ax_bot.plot(x, smooth(pareto_fraction, window),
                    label=f"ε = {eps}", color=colors[eps], linewidth=2)

    nash_welfare = 2 * (1 / 3)
    ax_top.axhline(pareto_welfare, color="#22c55e", linestyle="--", linewidth=1,
                   label=f"Pareto optimum W = {pareto_welfare:.2f}")
    ax_top.axhline(nash_welfare, color="#ef4444", linestyle="--", linewidth=1,
                   label=f"Nash W = {nash_welfare:.2f}")

    ax_top.set_ylabel("Mean welfare (smoothed)")
    ax_top.set_title("Scaled PD under the MYP (2014) learning rule")
    ax_top.set_ylim(0.5, 1.75)
    ax_top.grid(True, alpha=0.4)
    ax_top.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155")

    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylabel("Fraction of trials at (A, A)")
    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.grid(True, alpha=0.4)
    ax_bot.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "welfare-plot.svg")
    fig.tight_layout()
    fig.savefig(out_path, format="svg", facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
