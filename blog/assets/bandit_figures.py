"""Dark-theme figures for the multi-armed bandit post. Mirrors Kuleshov & Precup (2014).

Self-contained: run `python bandit_figures.py` from blog/assets/.
Setting: K=10 Gaussian arms, means ~ U(0,1), reward std sigma=0.1, horizon T=1000,
averaged over R=4000 independent runs, seed=2024.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = os.path.dirname(os.path.abspath(__file__))
BG, GRID = "#1e293b", "#334155"
TXT, MUT = "#e2e8f0", "#94a3b8"
C = {"random": "#64748b", "greedy": "#f59e0b", "epsilon": "#3b82f6",
     "softmax": "#fb923c", "ucb1": "#a855f7"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "text.color": TXT, "axes.labelcolor": TXT, "axes.titlecolor": TXT,
    "xtick.color": MUT, "ytick.color": MUT, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.alpha": 0.4, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.labelcolor": TXT,
})


def vec_evaluate(algo, R=4000, T=1000, K=10, sigma=0.1, eps=0.01, tau=0.05, init=0.0, seed=2024):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.0, 1.0, size=(R, K)); best = mu.max(axis=1); opt_arm = mu.argmax(axis=1)
    rows = np.arange(R); Q = np.full((R, K), init, float); N = np.zeros((R, K))
    cum = np.zeros(T); pct = np.zeros(T); running = np.zeros(R)
    for t in range(1, T + 1):
        if algo == "random":
            a = rng.integers(K, size=R)
        elif algo == "greedy":
            a = (Q + 1e-9 * rng.random((R, K))).argmax(1)
        elif algo == "epsilon":
            g = (Q + 1e-9 * rng.random((R, K))).argmax(1); rnd = rng.integers(K, size=R)
            a = np.where(rng.random(R) < eps, rnd, g)
        elif algo == "softmax":
            z = Q / tau; gum = -np.log(-np.log(rng.random((R, K)))); a = (z + gum).argmax(1)
        elif algo == "ucb1":
            if t <= K:
                a = np.full(R, t - 1)
            else:
                a = (Q + np.sqrt(2.0 * np.log(t) / N)).argmax(1)
        else:
            raise ValueError(algo)
        r = rng.normal(mu[rows, a], sigma); N[rows, a] += 1; Q[rows, a] += (r - Q[rows, a]) / N[rows, a]
        running += best - mu[rows, a]; cum[t - 1] = running.mean(); pct[t - 1] = np.mean(a == opt_arm)
    return cum, pct, N.mean(0)


def smooth(y, w=15):
    pad = w // 2; yp = np.pad(y, pad, mode="edge"); k = np.ones(w) / w
    return np.convolve(yp, k, mode="valid")[:len(y)]


COMMON = dict(R=4000, K=10, T=1000, sigma=0.1, seed=2024)
T = 1000; tt = np.arange(1, T + 1); cache = {}


def ev(key, **kw):
    if key not in cache:
        cache[key] = vec_evaluate(**{**COMMON, **kw})
    return cache[key]


# FIG 1 — hidden odds (seed 7 for the displayed instance, matches draft caption)
rng = np.random.default_rng(7); mu = rng.uniform(0, 1, 10); b = int(mu.argmax())
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = ["#475569"] * 10; cols[b] = "#22c55e"
bars = ax.bar(np.arange(10), mu, color=cols, edgecolor=BG, linewidth=1.2)
ax.set_xticks(range(10)); ax.set_xticklabels([f"#{i+1}" for i in range(10)])
ax.set_ylabel(r"True win rate  $\mu_i$  (hidden)"); ax.set_xlabel("Arm"); ax.set_ylim(0, 1.0)
ax.set_title("A 10-armed bandit: each arm pays out at a different, unknown rate")
ax.annotate("best arm\n(the agent must discover this)", xy=(b, mu[b]), xytext=(b - 2.6, mu[b] + 0.06),
            fontsize=10.5, color="#4ade80", arrowprops=dict(arrowstyle="->", color="#4ade80"))
for i in range(10):
    ax.text(i, mu[i] + 0.015, f"{mu[i]:.2f}", ha="center", va="bottom", fontsize=9, color=MUT)
fig.tight_layout(); fig.savefig(f"{OUT}/bandit_fig1_setup.png", bbox_inches="tight"); plt.close(fig)

# FIG 2 — dilemma (standard init Q0=0)
rnd = ev("rand", algo="random", init=0.0); grd = ev("g0", algo="greedy", init=0.0)
e0 = ev("e0", algo="epsilon", eps=0.10, init=0.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for (cum, _, _), c, lab in [(rnd, C["random"], "Random (explore-only)"),
                            (grd, C["greedy"], "Greedy (exploit-only)"),
                            (e0, C["epsilon"], r"$\epsilon$-greedy ($\epsilon=0.1$)")]:
    a1.plot(tt, cum, color=c, lw=2.2, label=lab)
a1.set_xlabel("Pulls"); a1.set_ylabel("Cumulative regret")
a1.set_title("Regret over time (lower is better)"); a1.legend(loc="upper left")
for (_, opt, _), c in [(rnd, C["random"]), (grd, C["greedy"]), (e0, C["epsilon"])]:
    a2.plot(tt, smooth(opt), color=c, lw=2.2)
a2.axhline(1.0, ls=":", c=MUT, lw=1); a2.set_ylim(0, 1.02)
a2.yaxis.set_major_formatter(PercentFormatter(1.0))
a2.set_xlabel("Pulls"); a2.set_ylabel("P(playing the best arm)")
a2.set_title("How often the best arm is chosen")
fig.suptitle("The dilemma: pure exploration and pure exploitation both fail",
             fontsize=14, color=TXT, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/bandit_fig2_dilemma.png", bbox_inches="tight"); plt.close(fig)

# FIG 3 — head-to-head (optimistic init Q0=1, paper-faithful) -> note in subtitle
eh = ev("e1", algo="epsilon", eps=0.10, init=1.0); sh = ev("s1", algo="softmax", tau=0.05, init=1.0)
uc = ev("u", algo="ucb1", init=1.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
series = [(eh, C["epsilon"], r"$\epsilon$-greedy ($\epsilon=0.1$)"),
          (sh, C["softmax"], r"Softmax ($\tau=0.05$)"),
          (uc, C["ucb1"], "UCB1 (no tuning)")]
for (cum, _, _), c, lab in series:
    a1.plot(tt, cum, color=c, lw=2.4, label=lab)
a1.set_xlabel("Pulls"); a1.set_ylabel("Cumulative regret")
a1.set_title("Regret over time (lower is better)"); a1.legend(loc="upper left")
for (_, opt, _), c, _ in series:
    a2.plot(tt, smooth(opt), color=c, lw=2.4)
a2.axhline(1.0, ls=":", c=MUT, lw=1); a2.set_ylim(0, 1.02)
a2.yaxis.set_major_formatter(PercentFormatter(1.0))
a2.set_xlabel("Pulls"); a2.set_ylabel("P(playing the best arm)")
a2.set_title("How often the best arm is chosen")
fig.suptitle("Head-to-head: the simple heuristics beat the 'optimal' algorithm",
             fontsize=14, color=TXT, y=1.04)
fig.text(0.5, 0.965, r"all contenders use an optimistic start ($Q_0=1$) — see section 6",
         ha="center", fontsize=9.5, color=MUT, style="italic")
fig.tight_layout(); fig.savefig(f"{OUT}/bandit_fig3_head_to_head.png", bbox_inches="tight"); plt.close(fig)

# FIG 4 — scoreboard standard vs optimistic
g0 = ev("g0", algo="greedy", init=0.0)[0][-1]; g1 = ev("g1", algo="greedy", init=1.0)[0][-1]
e0v = ev("e0", algo="epsilon", eps=0.10, init=0.0)[0][-1]; e1v = ev("e1", algo="epsilon", eps=0.10, init=1.0)[0][-1]
s0 = ev("s0", algo="softmax", tau=0.05, init=0.0)[0][-1]; s1 = ev("s1", algo="softmax", tau=0.05, init=1.0)[0][-1]
u0 = ev("u", algo="ucb1", init=1.0)[0][-1]
labels = ["Greedy", r"$\epsilon$-greedy", "Softmax", "UCB1"]; std = [g0, e0v, s0, u0]; opt = [g1, e1v, s1, u0]
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 4.8))
b1 = ax.bar(x - w/2, std, w, label="Standard start ($Q_0=0$)", color="#64748b")
b2 = ax.bar(x + w/2, opt, w, label="Optimistic start ($Q_0=1$)", color="#3b82f6")
ax.set_ylabel("Total regret after 1000 pulls\n(lower is better)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("One knob, total reshuffle: initialization decides the winner"); ax.legend()
for bs in (b1, b2):
    for bar in bs:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4, f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=9.5, color=TXT)
ax.axhline(rnd[0][-1], ls="--", c=C["random"], lw=1.3)
ax.text(3.45, rnd[0][-1] + 4, "random baseline", ha="right", va="bottom", fontsize=9, color=MUT)
ax.set_ylim(0, 440); fig.tight_layout()
fig.savefig(f"{OUT}/bandit_fig4_scoreboard.png", bbox_inches="tight"); plt.close(fig)

print(f"FIG3/4 check: UCB1={u0:.1f} eps1={e1v:.1f} sm1={s1:.1f} | g0={g0:.1f} s0={s0:.1f} e0={e0v:.1f}")
for f in sorted(os.listdir(OUT)):
    if f.startswith("bandit_fig"):
        print("  wrote", f, os.path.getsize(os.path.join(OUT, f)), "bytes")
