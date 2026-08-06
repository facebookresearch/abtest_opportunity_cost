# %% [markdown]
# # Unknown prior: hierarchical Bayes with the Bellman policy
#
# The prior N(mu_0, sigma_0^2) on version lifts is unknown. We place a
# hyperprior over (mu_0, sigma_0^2), update its posterior as versions resolve,
# and for each new version act with the Bellman-optimal policy for the *current*
# estimate (plug-in / certainty equivalent).
#
# The update is exact and prior-free: integrating out a version's lift theta_j,
# its data mean satisfies Xbar_j | (mu_0, sigma_0^2) ~ N(mu_0, sigma_0^2 + delta^2/k_j),
# a noisy observation of mu_0. The grid posterior is multiplied by this
# likelihood after each resolved version.
#
# Compares three policies: **oracle** (Bellman with the true prior), **hier
# plug-in** (learns the prior online), and **fixed wrong** (a confidently-wrong
# prior, never updated), over two starting hyperpriors of increasing difficulty:
# a mild miscalibration and a severe one (worse start + harder, slower-learning
# truth). Produces one self-contained single-panel figure per scenario,
# `unknown_prior_mild.pdf` and `unknown_prior_severe.pdf`, so the two paper
# builds can include them independently.

# %%
from dataclasses import dataclass

import numpy as np
from methods import BellmanCriterion, cached_criterion

# %%
TAU = 1.0
N_A, K_MAX, N_ITER = 301, 120, 300
ROUND = 0.1  # round (mu0, sigma0) before solving/caching the Bellman policy

n_rounds = 5000
n_sims = 60
seed = 42
cost = 0.0  # cost-free (reward = lift); paper figures use c=0


@dataclass(frozen=True)
class Scenario:
    """One unknown-prior experiment: the true prior, the noise, and the start."""

    mu0_true: float
    sigma0_true: float
    delta: float
    init: tuple[float, float]  # starting guess (mu0, sigma0) for the hyperprior
    title: str
    slug: str  # filename suffix: figures/unknown_prior_<slug>.pdf


# %%
def get_bellman(mu0, sigma0, delta, arm_cost):
    """Memoized Bellman solve, keyed by rounded (mu0, sigma0)."""
    mu_r = round(round(mu0 / ROUND) * ROUND, 4)
    sig_r = round(round(max(sigma0, 0.2) / ROUND) * ROUND, 4)
    return cached_criterion(
        BellmanCriterion,
        a0=mu_r,
        sigma0=sig_r,
        delta=delta,
        n_a=N_A,
        k_max=K_MAX,
        n_iter=N_ITER,
        arm_cost=arm_cost,
    )


class OracleBellman:
    label = "oracle (known prior)"

    def __init__(self, mu0_true, sigma0_true, delta, arm_cost):
        self.mu0, self.sigma0, self.delta, self.arm_cost = (
            mu0_true,
            sigma0_true,
            delta,
            arm_cost,
        )

    def policy_for_version(self):
        return (
            get_bellman(self.mu0, self.sigma0, self.delta, self.arm_cost),
            self.mu0,
            self.sigma0,
        )

    def update(self, xbar, k):
        pass


class FixedWrongBellman:
    label = "fixed wrong (no learning)"

    def __init__(self, mu0, sigma0, delta, arm_cost):
        self.mu0, self.sigma0, self.delta, self.arm_cost = mu0, sigma0, delta, arm_cost

    def policy_for_version(self):
        return (
            get_bellman(self.mu0, self.sigma0, self.delta, self.arm_cost),
            self.mu0,
            self.sigma0,
        )

    def update(self, xbar, k):
        pass


class HierPlugInBellman:
    label = "hier plug-in (learns)"

    # Grid laid over *compactified* coordinates so the hyperprior has full
    # support on R x (0, inf) with no bounding box: mu0 = C_MU * arctanh(u) and
    # log(sigma0) = C_LOGSIG * arctanh(v) for (u, v) uniform on (-1, 1). The
    # posterior can therefore converge to any truth regardless of the start,
    # which a fixed box cannot (mass outside it is identically zero).
    N_U, N_V = 141, 101
    C_MU, C_LOGSIG = 4.0, 2.5
    PRIOR_MU_SD, PRIOR_LOGSIG_SD = 5.0, 1.5

    def __init__(self, mu0_init, sigma0_init, delta, arm_cost):
        self.delta, self.arm_cost = delta, arm_cost
        eps = 1e-3
        u = np.linspace(-1.0 + eps, 1.0 - eps, self.N_U)
        v = np.linspace(-1.0 + eps, 1.0 - eps, self.N_V)
        grid_u, grid_v = np.meshgrid(u, v)
        self.MU = self.C_MU * np.arctanh(grid_u)
        self.LOGSIG = self.C_LOGSIG * np.arctanh(grid_v)
        self.SIG = np.exp(self.LOGSIG)
        # Weakly-informative *proper* prior: Gaussian on mu0 and on log(sigma0).
        # The tanh-map log-Jacobian makes grid sums approximate the continuous
        # integral; it is dominated at the edges by the Gaussian tails, so no
        # mass piles up against the numerical grid boundary.
        log_jac = -np.log(1.0 - grid_u**2) - np.log(1.0 - grid_v**2)
        log_prior = (
            -0.5 * ((self.MU - mu0_init) / self.PRIOR_MU_SD) ** 2
            - 0.5 * ((self.LOGSIG - np.log(sigma0_init)) / self.PRIOR_LOGSIG_SD) ** 2
        )
        self.logpost = log_prior + log_jac

    def _mean(self):
        p = np.exp(self.logpost - self.logpost.max())
        p /= p.sum()
        mu0 = float((self.MU * p).sum())
        # Geometric-mean (posterior mean of log-sigma) point estimate: the
        # natural summary in the log parametrization and not inflated by the
        # broad log-normal tail, so it equals the guess exactly before any data.
        log_sigma0 = float((self.LOGSIG * p).sum())
        return mu0, float(np.exp(log_sigma0))

    def policy_for_version(self):
        mu0, sigma0 = self._mean()
        return get_bellman(mu0, sigma0, self.delta, self.arm_cost), mu0, sigma0

    def update(self, xbar, k):
        var = self.SIG**2 + self.delta**2 / k
        self.logpost += (
            -0.5 * np.log(2 * np.pi * var) - 0.5 * (xbar - self.MU) ** 2 / var
        )


# %%
def simulate_run(learner, mu0_true, sigma0_true, delta, n_rounds, arm_cost, rng):
    """One OVT run; returns net-reward rate and reward-over-time."""
    total, t = 0.0, TAU
    v_hist = np.zeros(n_rounds)

    def new_version():
        theta = float(rng.normal(mu0_true, sigma0_true))
        crit, mu0, sigma0 = learner.policy_for_version()
        return theta, crit, mu0, sigma0**2, 0.0, 0

    theta, crit, a_hat, s2, sum_b, cnt = new_version()
    for n in range(n_rounds):
        action = crit.decide(a_hat, np.sqrt(s2))
        if action != "continue":
            if cnt > 0:
                learner.update(sum_b / cnt, cnt)
            if action == "launch":
                total += theta
            total -= arm_cost  # per-renewal sampling cost (launch or discard)
            theta, crit, a_hat, s2, sum_b, cnt = new_version()
        b = float(rng.normal(theta, delta))
        sum_b += b
        cnt += 1
        prec = 1.0 / s2 + 1.0 / delta**2
        a_hat = (a_hat / s2 + b / delta**2) / prec
        s2 = 1.0 / prec
        t += TAU
        v_hist[n] = total / t
    return total / t, v_hist


def run(make_learner, scen, n_rounds, n_sims, arm_cost, seed=42):
    vs, vh = [], []
    for i in range(n_sims):
        rng = np.random.default_rng(seed + i)
        v, vhist = simulate_run(
            make_learner(),
            scen.mu0_true,
            scen.sigma0_true,
            scen.delta,
            n_rounds,
            arm_cost,
            rng,
        )
        vs.append(v)
        vh.append(vhist)
    return np.array(vs), np.array(vh)


def run_scenario(scen, n_rounds, n_sims, arm_cost, seed):
    mu0, sigma0 = scen.init
    makers = {
        "oracle": lambda: OracleBellman(
            scen.mu0_true, scen.sigma0_true, scen.delta, arm_cost
        ),
        "hier plug-in": lambda: HierPlugInBellman(mu0, sigma0, scen.delta, arm_cost),
        "fixed wrong": lambda: FixedWrongBellman(mu0, sigma0, scen.delta, arm_cost),
    }
    return {
        name: run(mk, scen, n_rounds, n_sims, arm_cost, seed)
        for name, mk in makers.items()
    }


# %%
scenarios = [
    Scenario(
        mu0_true=-0.5,
        sigma0_true=1.0,
        delta=2.0,
        init=(0.5, 2.0),
        title=r"Mild start: $(\tilde\mu_0,\tilde\sigma_0)=(0.5,2)$, truth $(-0.5,1)$",
        slug="mild",
    ),
    Scenario(
        mu0_true=-1.0,
        sigma0_true=1.0,
        delta=2.0,
        init=(2.5, 4.0),
        title=r"Severe start: $(\tilde\mu_0,\tilde\sigma_0)=(2.5,4)$, truth $(-1,1)$",
        slug="severe",
    ),
]

results = [run_scenario(scen, n_rounds, n_sims, cost, seed) for scen in scenarios]

order = ["oracle", "hier plug-in", "fixed wrong"]
for scen, res in zip(scenarios, results):
    ov = res["oracle"][0].mean()
    print(f"\n{scen.title}")
    print(f"{'scheme':<14} {'final R':>11} {'%oracle':>8}")
    for name in order:
        vs = res[name][0]
        print(f"{name:<14} {vs.mean():>11.5f} {vs.mean() / ov * 100:>7.1f}%")

# %% [markdown]
# ## Figures: hierarchical-Bayes Bellman vs. true-prior Bellman
#
# One single-panel figure per scenario: average reward over time as a fraction
# of that scenario's oracle. The plug-in policy warms up and converges toward the
# oracle (more slowly the worse the hyperprior); the fixed-wrong prior earns
# negative reward and dives off-scale in the severe scenario.

# %%
import matplotlib.pyplot as plt
from plotting import apply_paper_style, save_figure

apply_paper_style()

styles = {
    "oracle": {"color": "navy", "ls": "-", "lw": 3.5},
    "hier plug-in": {"color": "tab:red", "ls": "--", "lw": 3},
    "fixed wrong": {"color": "tab:olive", "ls": "-.", "lw": 3},
}
labels = {
    "oracle": OracleBellman.label,
    "hier plug-in": HierPlugInBellman.label,
    "fixed wrong": FixedWrongBellman.label,
}

rounds = np.arange(1, n_rounds + 1)
start = 10

for scen, res in zip(scenarios, results):
    ov = res["oracle"][0].mean()  # normalize by this scenario's oracle rate
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in order:
        curve = res[name][1].mean(axis=0) / ov
        ax.plot(rounds[start:], curve[start:], label=labels[name], **styles[name])
    ax.axhline(1.0, color="navy", lw=1, ls=":", alpha=0.5)
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xscale("log")
    # Focus on the convergence band; the fixed-wrong baseline dives far below
    # (to ~-16x oracle in the severe scenario) and is clipped off-scale.
    ax.set_ylim(-4.0, 1.15)
    ax.set_xlabel("Round", fontsize=22)
    ax.set_ylabel("Average reward\n(fraction of oracle)", fontsize=22)
    ax.set_title(scen.title, fontsize=19)
    ax.legend(fontsize=15, loc="lower right")
    ax.grid(True, alpha=0.2, which="both")
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    save_figure(fig, f"unknown_prior_{scen.slug}.pdf")
    plt.show()
