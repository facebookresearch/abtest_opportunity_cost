# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %% [markdown]
# # Ablations: PPOS sigma* sweep, and Bayesian vs frequentist criteria
#
# Two appendix ablations, both at delta = 2 with two sampling costs (top row
# c=0, bottom row c=0.05):
#
# 1. **PPOS sigma* sweep** (`ablation_ppos.png`): no choice of the planned-analysis
#    target sigma* lets PPOS approach the Bellman optimum.
# 2. **Bayesian vs frequentist** (`ablation_bayes_vs_freq.png`, `tab:bayes-vs-freq`):
#    the Bayes/frequentist choice is large and prior-dependent for PPOS but
#    negligible for the z-criterion.

# %%
import numpy as np
from methods import (
    BellmanCriterion,
    cached_criterion,
    cached_reward_curves,
    FreqPPoSCriterion,
    FreqZCriterion,
    GaussSPParams,
    PPoSCriterion,
    ZCriterion,
)
from plotting import Z_CRIT

sigma0 = 1.0
delta = 2.0
mu0_values = [-1.0, -0.1, 0.1]
costs = [0.0, 0.05]
start = 10

# ======================================================================
# Ablation 1: PPOS sensitivity to sigma*
# ======================================================================
# %%
sigma_stars = [0.1, 0.2, 0.4, 0.6, 0.8]
ppos_n_sims = 200
ppos_n_rounds = 1000

ppos_curves = {}  # (cost, mu0) -> {name: mean_curve}
for cost in costs:
    for mu0 in mu0_values:
        bellman = cached_criterion(
            BellmanCriterion,
            a0=mu0,
            sigma0=sigma0,
            delta=delta,
            n_a=501,
            k_max=300,
            n_iter=500,
            arm_cost=cost,
        )
        params = GaussSPParams(
            a0=mu0,
            sigma0=sigma0,
            delta=delta,
            tau=1.0,
            n_rounds=ppos_n_rounds,
            cost=cost,
        )
        criteria = {"Bellman": bellman}
        for s in sigma_stars:
            criteria[f"ppos_{s}"] = PPoSCriterion(z=Z_CRIT, sigma_star=s)
        rc = cached_reward_curves(params, criteria, n_sims=ppos_n_sims, seed=42)
        ppos_curves[(cost, mu0)] = {k: v.mean_curve for k, v in rc.items()}
    print(f"ppos ablation cost={cost} done")

# %%
import matplotlib
import matplotlib.pyplot as plt
from plotting import apply_paper_style, save_figure

apply_paper_style()

greens = matplotlib.colormaps["Greens"](np.linspace(0.4, 0.9, len(sigma_stars)))
rounds = np.arange(1, ppos_n_rounds + 1)

fig, axes = plt.subplots(
    len(costs),
    len(mu0_values),
    figsize=(6 * len(mu0_values), 5.0 * len(costs)),
    squeeze=False,
)
for r, cost in enumerate(costs):
    for ci, mu0 in enumerate(mu0_values):
        ax = axes[r][ci]
        cur = ppos_curves[(cost, mu0)]
        for s, color in zip(sigma_stars, greens):
            ax.plot(
                rounds[start:],
                cur[f"ppos_{s}"][start:],
                color=color,
                ls="-.",
                lw=3,
                label=rf"PPOS ($\sigma^*={s}$)",
            )
        ax.plot(
            rounds[start:],
            cur["Bellman"][start:],
            color="navy",
            ls="-",
            lw=3.5,
            label="Bellman",
        )
        ax.set_xscale("log")
        if cost == 0:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color="0.6", lw=1.0, ls=":")
        ax.set_xlim(start, ppos_n_rounds)
        if r == len(costs) - 1:
            ax.set_xlabel("Round", fontsize=22)
        if r == 0:
            ax.set_title(
                rf"$\mu_0={mu0:g},\ \sigma_0={sigma0:g},\ \delta={delta:g}$",
                fontsize=21,
            )
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=18)
    metric = "Average reward" if cost == 0 else r"Average net reward $R$"
    axes[r][0].set_ylabel(metric + "\n" + rf"($c={cost:g}$)", fontsize=22)
_h, _l = axes[0][-1].get_legend_handles_labels()
fig.legend(
    _h,
    _l,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.04),
    ncol=len(sigma_stars) + 1,
    fontsize=16,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_figure(fig, "ablation_ppos.png")
plt.show()

# ======================================================================
# Ablation 2: Bayesian vs frequentist z-criterion and PPOS
# ======================================================================
# %%
sigma_star = 0.23
# Matches the paper (tab:bayes-vs-freq: 200 simulations of 2,500 rounds); the
# plot_bayes_vs_freq_ablation.py CLI defaults (300/3000) were not the paper run.
bf_n_sims = 200
bf_n_rounds = 2500

style = {
    "z-bayes": ("tab:olive", "--", 3, rf"$z$ (Bayes, $z={Z_CRIT}$)"),
    "z-freq": ("tab:olive", ":", 3, rf"$z$ (freq, $z={Z_CRIT}$)"),
    "ppos-bayes": ("seagreen", "--", 3, rf"PPOS (Bayes, $\sigma^*={sigma_star}$)"),
    "ppos-freq": ("seagreen", ":", 3, rf"PPOS (freq, $\sigma^*={sigma_star}$)"),
    "bellman": ("navy", "-", 3.5, "Bellman"),
}

bf_curves = {}  # (cost, mu0) -> {name: mean_curve}
for cost in costs:
    for mu0 in mu0_values:
        bellman = cached_criterion(
            BellmanCriterion,
            a0=mu0,
            sigma0=sigma0,
            delta=delta,
            n_a=501,
            k_max=300,
            n_iter=500,
            arm_cost=cost,
        )
        params = GaussSPParams(
            a0=mu0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=bf_n_rounds, cost=cost
        )
        criteria = {
            "z-bayes": ZCriterion(z=Z_CRIT),
            "z-freq": FreqZCriterion(z=Z_CRIT, a0=mu0, sigma0=sigma0),
            "ppos-bayes": PPoSCriterion(z=Z_CRIT, sigma_star=sigma_star),
            "ppos-freq": FreqPPoSCriterion(
                z=Z_CRIT, sigma_star=sigma_star, a0=mu0, sigma0=sigma0
            ),
            "bellman": bellman,
        }
        rc = cached_reward_curves(params, criteria, n_sims=bf_n_sims, seed=42)
        bf_curves[(cost, mu0)] = {k: v.mean_curve for k, v in rc.items()}
    print(f"bayes-vs-freq cost={cost} done")

# %%
rounds = np.arange(1, bf_n_rounds + 1)
fig, axes = plt.subplots(
    len(costs),
    len(mu0_values),
    figsize=(6 * len(mu0_values), 5.0 * len(costs)),
    squeeze=False,
)
order = ["z-bayes", "z-freq", "ppos-bayes", "ppos-freq", "bellman"]
for r, cost in enumerate(costs):
    for ci, mu0 in enumerate(mu0_values):
        ax = axes[r][ci]
        cur = bf_curves[(cost, mu0)]
        for name in order:
            color, ls, lw, label = style[name]
            ax.plot(
                rounds[start:],
                cur[name][start:],
                color=color,
                ls=ls,
                lw=lw,
                label=label,
            )
        ax.set_xscale("log")
        ax.axhline(
            0.0, color="0.6", lw=1.0, ls=":"
        )  # linear y so negative reward shows
        ax.set_xlim(start, bf_n_rounds)
        if r == len(costs) - 1:
            ax.set_xlabel("Round", fontsize=22)
        if r == 0:
            ax.set_title(
                rf"$\mu_0={mu0:g},\ \sigma_0={sigma0:g},\ \delta={delta:g}$",
                fontsize=21,
            )
        ax.grid(True, alpha=0.2, which="both")
        ax.tick_params(labelsize=18)
    metric = "Average reward" if cost == 0 else r"Average net reward $R$"
    axes[r][0].set_ylabel(metric + "\n" + rf"($c={cost:g}$)", fontsize=22)
_h, _l = axes[0][-1].get_legend_handles_labels()
fig.legend(
    _h,
    _l,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.04),
    ncol=5,
    fontsize=15,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_figure(fig, "ablation_bayes_vs_freq.png")
plt.show()

# %% [markdown]
# ## Table (`tab:bayes-vs-freq`): final reward as a fraction of Bellman (c = 0)

# %%
print("% LaTeX rows for tab:bayes-vs-freq (final reward / Bellman, c=0):")
print("Criterion & " + " & ".join(rf"$\mu_0={m:g}$" for m in mu0_values) + " \\\\")
row_names = [
    ("ppos-freq", "PPOS (frequentist)"),
    ("ppos-bayes", "PPOS (Bayesian)"),
    ("z-freq", "$z$ (frequentist)"),
    ("z-bayes", "$z$ (Bayesian)"),
]
for key, label in row_names:
    fracs = []
    for mu0 in mu0_values:
        cur = bf_curves[(0.0, mu0)]
        fracs.append(cur[key][-1] / cur["bellman"][-1])
    print(f"        {label} & " + " & ".join(f"{f:.2f}" for f in fracs) + " \\\\")
