# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %% [markdown]
# # Baselines vs oracle, and the per-run reward distribution
#
# Reproduces the main lift-comparison figure (`baselines_vs_bellman.pdf`), the
# baseline error table (`tab:baseline-errors`), and the reward-distribution
# appendix figure (`reward_distribution.pdf`).
#
# Criteria: the plain z-criterion, its optimistic/conservative deadline variants,
# PPOS, and the reward-optimal Bellman policy. The top row is cost-free (average
# lift, log-log); the bottom row adds a per-version sampling cost c=0.05 (net
# reward, linear).

# %%
import numpy as np
from methods import (
    BellmanCriterion,
    cached_criterion,
    cached_reward_curves,
    GaussSPParams,
    PPoSCriterion,
    simulate,
    ZCriterion,
)
from plotting import apply_paper_style, save_figure, Z_CRIT

# %%
a0_values = [-0.5, -0.1, 0.1]
sigma0 = 1.0
delta = 5.0
costs = [0.0, 0.05]  # top row cost-free, bottom row c=0.05
n_sims = 400
n_rounds = 5000

# Shared horizon so the z-deadline and the PPOS planned analysis coincide.
k_max = 50
sigma_star = 1.0 / np.sqrt(1.0 / sigma0**2 + k_max / delta**2)

print(f"a0={a0_values}, sigma0={sigma0}, delta={delta}, sigma*={sigma_star:.4f}")
print(f"{n_sims} sims x {n_rounds} rounds, costs={costs}")

# %%
# One cost-aware Bellman solve per (cost, prior); baselines are cost-blind.
reward_data = {}
for c in costs:
    reward_data[c] = {}
    for a0 in a0_values:
        bellman = cached_criterion(
            BellmanCriterion,
            a0=a0,
            sigma0=sigma0,
            delta=delta,
            n_a=501,
            k_max=200,
            n_iter=500,
            arm_cost=c,
        )
        params = GaussSPParams(
            a0=a0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=n_rounds, cost=c
        )
        criteria = {
            "z": ZCriterion(z=Z_CRIT),
            "z-optimistic": ZCriterion(z=Z_CRIT, sigma_min=sigma_star),
            "z-conservative": ZCriterion(
                z=Z_CRIT, sigma_min=sigma_star, conservative=True
            ),
            "PPOS": PPoSCriterion(z=Z_CRIT, sigma_star=sigma_star),
            "Bellman": bellman,
        }
        reward_data[c][a0] = cached_reward_curves(
            params, criteria, n_sims=n_sims, seed=42
        )
    print(f"cost c={c} done")

# %%
import matplotlib.pyplot as plt

apply_paper_style()

styles = {
    "z": {"color": "tab:olive", "ls": "--", "lw": 3},
    "z-optimistic": {"color": "tab:orange", "ls": "--", "lw": 3},
    "z-conservative": {"color": "mediumpurple", "ls": "--", "lw": 3},
    "PPOS": {"color": "seagreen", "ls": "-.", "lw": 3},
    "Bellman": {"color": "navy", "ls": "-", "lw": 3.5},
}
crit_labels = {
    "z": rf"$z$-criterion ($z={Z_CRIT}$)",
    "z-optimistic": rf"optimistic $z$ ($k_{{\max}}\!={k_max}$)",
    "z-conservative": rf"conservative $z$ ($k_{{\max}}\!={k_max}$)",
    "PPOS": rf"PPOS ($\sigma^*\!={sigma_star:.2f}$)",
    "Bellman": "Bellman",
}

# %%
start = 10
plot_order = ["z", "z-optimistic", "z-conservative", "PPOS", "Bellman"]
rounds = np.arange(1, n_rounds + 1)

fig, axes = plt.subplots(len(costs), 3, figsize=(18, 5.0 * len(costs)), squeeze=False)
for r, c in enumerate(costs):
    for ci, a0 in enumerate(a0_values):
        ax = axes[r][ci]
        for name in plot_order:
            rc = reward_data[c][a0][name]
            mean_v = rc.mean_curve[start:]
            if rc.std_curve is not None:
                lo = mean_v - rc.std_curve[start:]
                hi = mean_v + rc.std_curve[start:]
                if c == 0:
                    lo = np.maximum(lo, 1e-6)  # log y-axis cannot show <= 0
                ax.fill_between(
                    rounds[start:],
                    lo,
                    hi,
                    color=styles[name]["color"],
                    alpha=0.15,
                    linewidth=0,
                )
            ax.plot(rounds[start:], mean_v, label=crit_labels[name], **styles[name])
        ax.set_xscale("log")
        if c == 0:
            ax.set_yscale("log")  # cost-free lift spans orders of magnitude
        else:
            ax.axhline(0.0, color="0.6", lw=1.0, ls=":")
        ax.set_xlim(start, n_rounds)
        if r == len(costs) - 1:
            ax.set_xlabel("Round", fontsize=22)
        ax.set_title(
            rf"$\mu_0={a0},\ \sigma_0={sigma0:.0f},\ \delta={delta:.0f},\ c={c:g}$",
            fontsize=20,
        )
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=18)
    axes[r][0].set_ylabel("Reward", fontsize=22)

_h, _l = axes[0][0].get_legend_handles_labels()
fig.legend(
    _h,
    _l,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.10),
    ncol=3,
    fontsize=22,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.11, 1, 1))
save_figure(fig, "baselines_vs_bellman.pdf")
plt.show()

# %% [markdown]
# ## Table 1 (`tab:baseline-errors`): error rates and lift, mu_0 = -0.1
#
# For each baseline: launches per sim, false launch rate (FLR = P(theta<0 | launch)),
# worst-case type-I error (P(launch) with theta fixed at 0), and average lift.

# %%
table_a0 = -0.1
n_sims_tab = 300
n_rounds_tab = 2000
seed_tab = 42

tab_params = GaussSPParams(
    a0=table_a0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=n_rounds_tab
)
tab_bellman = cached_criterion(
    BellmanCriterion,
    a0=table_a0,
    sigma0=sigma0,
    delta=delta,
    n_a=501,
    k_max=200,
    n_iter=500,
)
tab_criteria = {
    r"$z$-criterion ($z=1.96$)": ZCriterion(z=Z_CRIT),
    rf"$z$-criterion ($z=1.96$, $k_{{\max}}={k_max}$, optimistic)": ZCriterion(
        z=Z_CRIT, sigma_min=sigma_star
    ),
    rf"$z$-criterion ($z=1.96$, $k_{{\max}}={k_max}$, conservative)": ZCriterion(
        z=Z_CRIT, sigma_min=sigma_star, conservative=True
    ),
    rf"PPOS ($\sigma^* = {sigma_star:.2f}$)": PPoSCriterion(
        z=Z_CRIT, sigma_star=sigma_star
    ),
    "Bellman (unconstrained)": tab_bellman,
}


def prior_rates(crit):
    """(launches/sim, FLR, lift) under theta ~ N(mu_0, sigma_0^2)."""
    n_launch = n_false = 0
    lift_sum = 0.0
    for s in range(n_sims_tab):
        res = simulate(tab_params, crit, rng=np.random.default_rng(seed_tab + s))
        n_launch += res.n_launched
        lift_sum += res.v
        for arm in res.arm_traces:
            if arm.outcome == "launched" and arm.true_value < 0:
                n_false += 1
    flr = n_false / n_launch if n_launch else float("nan")
    return n_launch / n_sims_tab, flr, lift_sum / n_sims_tab


def type_i_at_zero(crit, seed=seed_tab + 9973):
    """P(launch) among resolved versions when theta is fixed at 0 (the null boundary)."""
    delta2 = delta**2
    n_launch = n_resolved = 0
    for s in range(n_sims_tab):
        rng = np.random.default_rng(seed + s)
        a_hat, s2 = table_a0, sigma0**2
        for _ in range(n_rounds_tab):
            act = crit.decide(a_hat, float(np.sqrt(s2)))
            if act != "continue":
                n_resolved += 1
                n_launch += int(act == "launch")
                a_hat, s2 = table_a0, sigma0**2
            b = float(rng.normal(0.0, delta))
            pp, po = 1.0 / s2, 1.0 / delta2
            a_hat, s2 = (pp * a_hat + po * b) / (pp + po), 1.0 / (pp + po)
    return n_launch / n_resolved if n_resolved else float("nan")


tab_rows = []
for label, crit in tab_criteria.items():
    launches, flr, lift = prior_rates(crit)
    tab_rows.append((label, launches, flr, type_i_at_zero(crit), lift))

# %%
print(f"{'Criterion':<54} {'Launches/sim':>12} {'FLR':>7} {'Type-I':>7} {'Lift':>8}")
for label, launches, flr, type_i, lift in tab_rows:
    print(f"{label:<54} {launches:12.1f} {flr:7.3f} {type_i:7.3f} {lift:8.4f}")

print("\n% LaTeX rows for Table 1 (tab:baseline-errors):")
for label, launches, flr, type_i, lift in tab_rows:
    print(
        f"        {label} & {launches:.1f} & {flr:.3f} & {type_i:.3f} & {lift:.4f} \\\\"
    )

# %% [markdown]
# ## Reward distribution (`reward_distribution.pdf`)
#
# The spread of the final per-run reward at mu_0 = -0.1, delta = 5, c = 0. The
# plain z-criterion is heavily right-skewed (a pile near zero), a finite-horizon
# shadow of its vanishing asymptotic rate; the others concentrate around their
# means.

# %%
dist_a0 = -0.1
dist_n_sims = 400
dist_n_rounds = 3000
dist_bellman = cached_criterion(
    BellmanCriterion,
    a0=dist_a0,
    sigma0=sigma0,
    delta=delta,
    n_a=501,
    k_max=200,
    n_iter=500,
)
dist_criteria = {
    "z": ZCriterion(z=Z_CRIT),
    "z-optimistic": ZCriterion(z=Z_CRIT, sigma_min=sigma_star),
    "z-conservative": ZCriterion(z=Z_CRIT, sigma_min=sigma_star, conservative=True),
    "PPOS": PPoSCriterion(z=Z_CRIT, sigma_star=sigma_star),
    "Bellman": dist_bellman,
}
dist_params = GaussSPParams(
    a0=dist_a0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=dist_n_rounds
)
pretty = {
    "z": r"$z$-criterion",
    "z-optimistic": r"optimistic $z$",
    "z-conservative": r"conservative $z$",
    "PPOS": "PPOS",
    "Bellman": "Bellman",
}

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, len(dist_criteria), figsize=(4.2 * len(dist_criteria), 4.0))
print(
    f"\nPer-run final reward over {dist_n_sims} sims "
    f"(mu_0={dist_a0}, delta={delta}, {dist_n_rounds} rounds)"
)
print(f"{'criterion':<15} {'mean':>9} {'median':>9} {'frac<=0':>8}")
for ax, (name, crit) in zip(axes, dist_criteria.items()):
    finals = np.array(
        [
            simulate(dist_params, crit, rng=np.random.default_rng(42 + i)).v
            for i in range(dist_n_sims)
        ]
    )
    mean, median = finals.mean(), np.median(finals)
    print(f"{name:<15} {mean:>9.5f} {median:>9.5f} {float((finals <= 0).mean()):>8.2f}")
    ax.hist(finals, bins=40, color="navy", alpha=0.8)
    ax.axvline(float(mean), color="tab:red", lw=2, label="mean")
    ax.axvline(float(median), color="tab:green", lw=2, ls="--", label="median")
    ax.set_title(pretty[name], fontsize=14)
    ax.set_xlabel("final reward")
    ax.legend(fontsize=10)
fig.tight_layout()
save_figure(fig, "reward_distribution.pdf")
plt.show()
apply_paper_style()  # restore
