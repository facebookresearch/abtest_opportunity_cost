# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %% [markdown]
# # Robustness: noise misspecification and horizon truncation
#
# Two appendix checks on the Bellman policy:
#
# 1. **delta misspecification** (`bellman_delta_robustness.pdf`): a policy
#    computed at an assumed noise delta_used is run across true noise levels
#    delta_true, with the noise *estimated from data* during the run (the
#    realistic case), against an oracle that recomputes the policy at each
#    delta_true.
# 2. **horizon truncation** (`kmax_sensitivity.pdf`, `tab:kmax-ablation`): sweep
#    the grid truncation k_max and confirm reward plateaus -- i.e. the deadline
#    is non-binding.

# %%
import numpy as np
from methods import BellmanCriterion, cached_criterion, GaussSPParams, simulate

# ======================================================================
# Part 1: delta misspecification
# ======================================================================
# %%
a0 = -0.1
sigma0 = 1.0
tau = 1.0
delta_used = 2.0  # policy A: assumes the true noise
delta_used2 = 10.0  # policy B: assumes far MORE noise than the truth
n_rounds = 5000
n_sims = 400
delta_true_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])


def mean_lift(criterion, params, n_sims, seed=42):
    vs = [
        simulate(params, criterion, np.random.default_rng(seed + i)).v
        for i in range(n_sims)
    ]
    return float(np.mean(vs)), float(np.std(vs) / np.sqrt(n_sims))


def simulate_estimated_delta(params, criterion, rng, delta_init):
    """Like simulate() but estimates delta from consecutive observation pairs
    (Var(X_i - X_{i-1}) = 2 delta^2) and uses the running estimate for updates."""
    a0, sigma0, delta_true = params.a0, params.sigma0, params.delta
    V, t = 0.0, params.tau
    sum_sq_diff, n_pairs = 0.0, 0
    delta_est_sq = delta_init**2
    a_true = rng.normal(a0, sigma0)
    a_hat, sigma_hat2 = a0, sigma0**2
    prev_obs = None
    for _ in range(params.n_rounds):
        action = criterion.decide(a_hat, np.sqrt(sigma_hat2))
        if action in ("launch", "reject"):
            if action == "launch":
                V += a_true
            a_true = rng.normal(a0, sigma0)
            a_hat, sigma_hat2 = a0, sigma0**2
            prev_obs = None
        b = rng.normal(a_true, delta_true)
        if prev_obs is not None:
            sum_sq_diff += (b - prev_obs) ** 2
            n_pairs += 1
            delta_est_sq = sum_sq_diff / (2 * n_pairs)
        prev_obs = b
        prec_prior = 1.0 / sigma_hat2
        prec_obs = 1.0 / max(delta_est_sq, 1e-6)
        prec_post = prec_prior + prec_obs
        a_hat = (prec_prior * a_hat + prec_obs * b) / prec_post
        sigma_hat2 = 1.0 / prec_post
        t += params.tau
    return V / t if t > 0 else 0.0


def mean_lift_estimated(criterion, params, n_sims, delta_init, seed=42):
    vs = [
        simulate_estimated_delta(
            params, criterion, np.random.default_rng(seed + i), delta_init
        )
        for i in range(n_sims)
    ]
    return float(np.mean(vs)), float(np.std(vs) / np.sqrt(n_sims))


# %%
# A larger delta_used needs a deeper grid: when delta_true < delta_used each real
# observation is more informative than assumed, so sigma_hat reaches small values
# that map to high k.
bellman_misspec = cached_criterion(
    BellmanCriterion, a0=a0, sigma0=sigma0, delta=delta_used, k_max=400, n_iter=500
)
bellman_misspec2 = cached_criterion(
    BellmanCriterion, a0=a0, sigma0=sigma0, delta=delta_used2, k_max=2500, n_iter=500
)

paper_results = []
for dt in delta_true_values:
    params = GaussSPParams(
        a0=a0, sigma0=sigma0, delta=float(dt), tau=tau, n_rounds=n_rounds
    )
    oracle = cached_criterion(
        BellmanCriterion, a0=a0, sigma0=sigma0, delta=float(dt), k_max=200, n_iter=500
    )
    v_bo, se_bo = mean_lift(oracle, params, n_sims)
    v_be, se_be = mean_lift_estimated(
        bellman_misspec, params, n_sims, delta_init=delta_used
    )
    v_be2, se_be2 = mean_lift_estimated(
        bellman_misspec2, params, n_sims, delta_init=delta_used2
    )
    paper_results.append(
        {
            "dt": dt,
            "v_bo": v_bo,
            "se_bo": se_bo,
            "v_be": v_be,
            "se_be": se_be,
            "v_be2": v_be2,
            "se_be2": se_be2,
        }
    )
    print(
        f"delta_true={dt:.1f}: oracle={v_bo:.5f}, used=2 {v_be:.5f}, used=10 {v_be2:.5f}"
    )

# %%
import matplotlib.pyplot as plt
from plotting import apply_paper_style, save_figure

apply_paper_style()

dt = [r["dt"] for r in paper_results]
frac = [r["v_be"] / r["v_bo"] if r["v_bo"] > 0 else np.nan for r in paper_results]
frac2 = [r["v_be2"] / r["v_bo"] if r["v_bo"] > 0 else np.nan for r in paper_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
ax.errorbar(
    dt,
    [r["v_bo"] for r in paper_results],
    yerr=[r["se_bo"] for r in paper_results],
    fmt="s-",
    color="navy",
    lw=3,
    markersize=7,
    capsize=4,
    label="Bellman (oracle)",
)
ax.errorbar(
    dt,
    [r["v_be"] for r in paper_results],
    yerr=[r["se_be"] for r in paper_results],
    fmt="^--",
    color="tab:red",
    lw=3,
    markersize=8,
    capsize=4,
    label=rf"Bellman ($\delta_{{\mathrm{{used}}}}\!=\!{delta_used:.0f}$)",
)
ax.errorbar(
    dt,
    [r["v_be2"] for r in paper_results],
    yerr=[r["se_be2"] for r in paper_results],
    fmt="o--",
    color="tab:green",
    lw=3,
    markersize=8,
    capsize=4,
    label=rf"Bellman ($\delta_{{\mathrm{{used}}}}\!=\!{delta_used2:.0f}$)",
)
ax.axvline(delta_used, color="gray", ls=":", lw=1.5, alpha=0.6)
ax.set_xlabel(r"$\delta_{\mathrm{true}}$", fontsize=22)
ax.set_ylabel("Average lift", fontsize=22)
ax.tick_params(labelsize=18)
ax.legend(fontsize=14, loc="upper right")
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.axhline(1.0, color="navy", ls="-", lw=2, alpha=0.5, label="Oracle")
ax.plot(
    dt,
    frac,
    "^-",
    color="tab:red",
    lw=3,
    markersize=8,
    label=rf"$\delta_{{\mathrm{{used}}}}\!=\!{delta_used:.0f}$",
)
ax.plot(
    dt,
    frac2,
    "o-",
    color="tab:green",
    lw=3,
    markersize=8,
    label=rf"$\delta_{{\mathrm{{used}}}}\!=\!{delta_used2:.0f}$",
)
ax.axvline(delta_used, color="gray", ls=":", lw=1.5, alpha=0.6)
ax.set_xlabel(r"$\delta_{\mathrm{true}}$", fontsize=22)
ax.set_ylabel("Fraction of oracle lift", fontsize=22)
ax.tick_params(labelsize=18)
ax.legend(fontsize=14, loc="lower left")
ax.grid(True, alpha=0.2)
ax.set_ylim(0.55, 1.05)
plt.tight_layout()
save_figure(fig, "bellman_delta_robustness.pdf")
plt.show()

# ======================================================================
# Part 2: horizon truncation k_max
# ======================================================================
# %% [markdown]
# ## Horizon truncation: does the finite k_max deadline bind?
#
# Sweep k_max across priors and measure achieved reward. If reward plateaus as
# k_max grows, the deadline is non-binding and the truncation is harmless.

# %%
# (mu0, sigma0, delta, label)
PARAMS = [
    (-0.5, 1.0, 2.0, r"$\mu_0=-0.5,\ \delta=2$"),
    (-1.0, 1.0, 2.0, r"$\mu_0=-1.0,\ \delta=2$"),
    (-0.5, 1.0, 5.0, r"$\mu_0=-0.5,\ \delta=5$"),
    (-1.0, 1.0, 5.0, r"$\mu_0=-1.0,\ \delta=5$"),
    (0.5, 1.0, 2.0, r"$\mu_0=+0.5,\ \delta=2$ (empty)"),
]
KMAXES = [25, 50, 100, 200, 400, 800]
KN_A, KN_ITER = 301, 250
KN_SIMS, KN_ROUNDS = 40, 4000


def avg_lift(mu0, sigma0, delta, k_max):
    crit = cached_criterion(
        BellmanCriterion,
        a0=mu0,
        sigma0=sigma0,
        delta=delta,
        n_a=KN_A,
        k_max=k_max,
        n_iter=KN_ITER,
    )
    params = GaussSPParams(
        a0=mu0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=KN_ROUNDS
    )
    vs = [
        simulate(params, crit, rng=np.random.default_rng(42 + i)).v
        for i in range(KN_SIMS)
    ]
    return float(np.mean(vs))


lift = {}
for pi, (mu0, sig0, dlt, label) in enumerate(PARAMS):
    for km in KMAXES:
        lift[(pi, km)] = avg_lift(mu0, sig0, dlt, km)
    print(f"{label:<28} " + "  ".join(f"k={km}:{lift[(pi, km)]:.5f}" for km in KMAXES))

# %%
# Table tab:kmax-ablation: each row normalized by its k_max=800 value.
print("\n% LaTeX rows for tab:kmax-ablation (fraction of k_max=800 reward):")
for pi, (_m, _s, _d, label) in enumerate(PARAMS):
    ref = lift[(pi, KMAXES[-1])]
    cells = " & ".join(f"{lift[(pi, km)] / ref:.3f}" for km in KMAXES)
    print(f"        {label} & {cells} \\\\")

# %%
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["navy", "tab:red", "tab:olive", "seagreen", "tab:purple"]
for pi, (_m, _s, _d, label) in enumerate(PARAMS):
    ref = lift[(pi, KMAXES[-1])]
    ys = [lift[(pi, km)] / ref for km in KMAXES]
    ax.plot(KMAXES, ys, "o-", color=colors[pi % len(colors)], lw=3, ms=8, label=label)
ax.axhline(1.0, color="gray", ls=":", lw=2, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"$k_{\max}$", fontsize=22)
ax.set_ylabel(r"reward / reward at $k_{\max}=%d$" % KMAXES[-1], fontsize=22)
ax.set_title("Reward vs. horizon truncation", fontsize=22)
ax.legend(fontsize=15, loc="lower right")
ax.grid(True, alpha=0.2, which="both")
ax.tick_params(labelsize=18)
plt.tight_layout()
save_figure(fig, "kmax_sensitivity.pdf")
plt.show()
