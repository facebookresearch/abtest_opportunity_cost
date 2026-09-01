# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %% [markdown]
# # Bayesian FLR control via the Lagrangian Bellman
#
# The Lagrangian Bellman controls the false launch rate FLR = P(theta < 0 | launch)
# by penalizing the launch reward:
#
#     launch_reward = a_hat - lambda * (Phi(-a_hat/sigma_hat) - alpha).
#
# For each target alpha we find lambda*(alpha) by bisection so the constraint
# binds. We then trace the reward-FLR frontier and compare against the
# conservative z-criterion and PPOS swept over their own FLR knob beta.
#
# Everything is reported at a per-version sampling cost c=0.05 (net reward): the
# cost is charged on every renewal and threaded into the Bellman/Lagrangian
# solves and the lambda* bisection, so lambda*(alpha) is re-bisected at this cost.
#
# Produces `flr_control_c0.05.pdf` and `tab:lagrangian`. lambda* values and
# simulation metrics are cached under cache/ so the first (slow) run is one-time.

# %%
import json
import os

import numpy as np
from matplotlib.lines import Line2D
from methods import (
    BellmanCriterion,
    CACHE_DIR,
    cached_criterion,
    GaussSPParams,
    LagrangianBellmanCriterion,
    PPoSCriterion,
    simulate,
    ZCriterion,
)
from plotting import apply_paper_style, save_figure
from scipy.stats import norm

# %%
a0 = -0.1
sigma0 = 1.0
delta = 5.0
cost = 0.05  # per-version sampling cost; charged on every renewal (net reward)
n_sims = 200
n_rounds = 2000
# Planned-FLR levels traced for the Lagrangian frontier (the paper's beta range).
alpha_values = [0.3, 0.2, 0.1, 0.05, 0.025, 0.01]

params = GaussSPParams(
    a0=a0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=n_rounds, cost=cost
)
print(
    f"a0={a0}, sigma0={sigma0}, delta={delta}, c={cost}, "
    f"{n_sims} sims x {n_rounds} rounds"
)


# %%
def sigma_star_of(k):
    return 1.0 / np.sqrt(1.0 / sigma0**2 + k / delta**2)


def z_of_beta(beta):
    return float(norm.ppf(1.0 - beta))


# --- append-only caches under cache/ (recompute-from-scratch on first run) ---
os.makedirs(CACHE_DIR, exist_ok=True)
LAM_CACHE = os.path.join(CACHE_DIR, "lagrangian_points.jsonl")
SIM_CACHE = os.path.join(CACHE_DIR, "flr_metrics.jsonl")

_lam_cache = {}
if os.path.exists(LAM_CACHE):
    with open(LAM_CACHE) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                _lam_cache[
                    (e["a0"], e["sigma0"], e["delta"], e["alpha"], e.get("cost", 0.0))
                ] = e["lambda_star"]

_sim_cache = {}
if os.path.exists(SIM_CACHE):
    with open(SIM_CACHE) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                _sim_cache[e["key"]] = e["val"]


def _append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def eval_metrics(crit, sim_params, key, n=n_sims, seed=42):
    """Cached (FLR, lift v, SE, launches) for a criterion under sim_params."""
    if key in _sim_cache:
        return _sim_cache[key]
    n_launch = n_false = 0
    v_finals = []
    for i in range(n):
        res = simulate(sim_params, crit, rng=np.random.default_rng(seed + i))
        v_finals.append(res.v)
        for arm in res.arm_traces:
            if arm.outcome == "launched":
                n_launch += 1
                if arm.true_value < 0:
                    n_false += 1
    val = {
        "flr": n_false / n_launch if n_launch else float("nan"),
        "v": float(np.mean(v_finals)),
        "se_v": float(np.std(v_finals) / np.sqrt(n)),
        "n_launched": n_launch,
    }
    _sim_cache[key] = val
    _append(SIM_CACHE, {"key": key, "val": val})
    return val


def base_key(label, extra=""):
    return (
        f"{label}|a0={a0}|s0={sigma0}|d={delta}|c={cost}"
        f"|ns={n_sims}|nr={n_rounds}|seed=42{extra}"
    )


# %% [markdown]
# ## Find lambda*(alpha) by bisection (cached)


# %%
def estimate_flr(crit, n_search, seed=42):
    n_launch = n_false = 0
    for i in range(n_search):
        res = simulate(params, crit, rng=np.random.default_rng(seed + i))
        for arm in res.arm_traces:
            if arm.outcome == "launched":
                n_launch += 1
                if arm.true_value < 0:
                    n_false += 1
    return n_false / n_launch if n_launch else 0.0


n_sims_search = 100
n_bisect = 12
lambda_star = {}
lagrangian = {}
for alpha in alpha_values:
    key = (a0, sigma0, delta, alpha, cost)
    if key in _lam_cache:
        lambda_star[alpha] = _lam_cache[key]
        print(f"alpha={alpha}: cached lambda*={lambda_star[alpha]:.4f}")
    else:
        lo, hi = 0.0, 80.0
        for _ in range(n_bisect):
            lam = (lo + hi) / 2
            crit = cached_criterion(
                LagrangianBellmanCriterion,
                a0=a0,
                sigma0=sigma0,
                delta=delta,
                lam=lam,
                alpha=alpha,
                n_a=301,
                k_max=150,
                n_iter=300,
                arm_cost=cost,
            )
            if estimate_flr(crit, n_sims_search) > alpha:
                lo = lam
            else:
                hi = lam
        lambda_star[alpha] = (lo + hi) / 2
        _lam_cache[key] = lambda_star[alpha]
        _append(
            LAM_CACHE,
            {
                "a0": a0,
                "sigma0": sigma0,
                "delta": delta,
                "alpha": alpha,
                "cost": cost,
                "lambda_star": lambda_star[alpha],
            },
        )
        print(f"alpha={alpha}: found lambda*={lambda_star[alpha]:.4f}")
    lagrangian[alpha] = cached_criterion(
        LagrangianBellmanCriterion,
        a0=a0,
        sigma0=sigma0,
        delta=delta,
        lam=lambda_star[alpha],
        alpha=alpha,
        n_a=501,
        k_max=150,
        n_iter=500,
        arm_cost=cost,
    )

# %% [markdown]
# ## Evaluate the Lagrangian frontier and the swept baselines

# %%
bellman = cached_criterion(
    BellmanCriterion,
    a0=a0,
    sigma0=sigma0,
    delta=delta,
    n_a=501,
    k_max=150,
    n_iter=500,
    arm_cost=cost,
)

lb_vs = []
for alpha in alpha_values:
    m = eval_metrics(lagrangian[alpha], params, base_key(lagrangian[alpha].label))
    lb_vs.append(m["v"])

# Baseline frontiers: sweep beta (the FLR knob) at fixed horizons k_max, over the
# same planned-FLR grid as the Lagrangian so the frontiers span an identical range.
beta_sweep = list(alpha_values)
kmax_lines = [25, 50, 75]
baseline_makers = {
    "z-cons": lambda beta, k: ZCriterion(
        z=z_of_beta(beta), sigma_min=sigma_star_of(k), conservative=True
    ),
    "PPOS": lambda beta, k: PPoSCriterion(
        z=z_of_beta(beta), sigma_star=sigma_star_of(k), launch_q=1.0 - beta
    ),
}
baseline_sweep = {}
for kind, maker in baseline_makers.items():
    for k in kmax_lines:
        pts = []
        for beta in beta_sweep:
            crit = maker(beta, k)
            m = eval_metrics(
                crit, params, base_key(crit.label, extra=f"|{kind}|k={k}|b={beta}")
            )
            pts.append({"beta": beta, "v": m["v"]})
        baseline_sweep[(kind, k)] = pts
    print(f"baseline sweep {kind} done")

# %% [markdown]
# ## Figure `flr_control_c0.05.pdf`: decision boundaries + reward-vs-planned-FLR frontier

# %%
import matplotlib.pyplot as plt

apply_paper_style()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))

# ---- Left: unconstrained vs Lagrangian (alpha=0.05) decision boundaries ----
alpha_plot = 0.05
ura, urs, ula, uls = bellman.extract_boundaries(sigma_min=0.05)
axL.plot(ura, urs, "-", color="navy", lw=3)
axL.plot(ula, uls, "--", color="navy", lw=3)
lra, lrs, lla, lls = lagrangian[alpha_plot].extract_boundaries(sigma_min=0.05)
axL.plot(lra, lrs, "-", color="crimson", lw=3)
axL.plot(lla, lls, "--", color="crimson", lw=3)
axL.plot(a0, sigma0, "k*", ms=16, zorder=6)
axL.axvline(0, color="gray", ls=":", alpha=0.5)
axL.set_xlabel(r"Posterior mean $\hat{\mu}$", fontsize=22)
axL.set_ylabel(r"Posterior std $\hat{\sigma}$", fontsize=22)
axL.set_ylim(0.3, sigma0 * 1.02)
axL.tick_params(labelsize=18)
axL.grid(True, alpha=0.2)
axL.legend(
    handles=[
        Line2D([0], [0], color="navy", lw=3, label="Bellman (unconstrained)"),
        Line2D(
            [0], [0], color="crimson", lw=3, label=rf"Lagrangian ($\beta={alpha_plot}$)"
        ),
        Line2D([0], [0], color="gray", lw=2, ls="-", label="discard boundary"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="launch boundary"),
    ],
    fontsize=15,
    loc="upper right",
    frameon=False,
)

# ---- Right: reward vs planned FLR (alpha for Lagrangian, beta for baselines) ----
axR.plot(
    alpha_values,
    lb_vs,
    "o-",
    color="royalblue",
    lw=3,
    ms=8,
    label="Lagrangian Bellman",
    zorder=5,
)
purples = plt.cm.Purples([0.5, 0.68, 0.88])
greens = plt.cm.Greens([0.5, 0.68, 0.88])
baseline_plot = {
    "z-cons": {"marker": "X", "colors": purples, "label": "cons. $z$"},
    "PPOS": {"marker": "^", "colors": greens, "label": "PPOS"},
}
for kind, st in baseline_plot.items():
    for ci, k in enumerate(kmax_lines):
        pts = baseline_sweep[(kind, k)]
        axR.plot(
            [p["beta"] for p in pts],
            [p["v"] for p in pts],
            ls="--",
            lw=2.3,
            marker=st["marker"],
            ms=8,
            color=st["colors"][ci],
            zorder=4,
            label=rf"{st['label']} ($k_{{\max}}\!=\!{k}$)",
        )
axR.set_xlabel(r"Planned FLR $\beta$", fontsize=22)
axR.set_ylabel(r"Net reward $R$", fontsize=22)
axR.legend(fontsize=11, loc="lower right", ncol=2)
axR.grid(True, alpha=0.2, which="both")
axR.tick_params(labelsize=18)

plt.tight_layout()
save_figure(fig, f"flr_control_c{cost:g}.pdf")
plt.show()

# %% [markdown]
# ## Table `tab:lagrangian`: lambda sweep at fixed alpha=0.05

# %%
tab_lambda_sweep = [1.0, 2.0, 3.0, 5.0, 8.0]
n_sims_tab = 200
print("lambda  FLR     lift_v     launched")
tab_rows = []
for lam in tab_lambda_sweep:
    crit = cached_criterion(
        LagrangianBellmanCriterion,
        a0=a0,
        sigma0=sigma0,
        delta=delta,
        lam=lam,
        alpha=0.05,
        n_a=501,
        k_max=150,
        n_iter=500,
        arm_cost=cost,
    )
    m = eval_metrics(
        crit, params, base_key(crit.label, extra=f"|tab|ns={n_sims_tab}"), n=n_sims_tab
    )
    print(f"{lam:>5.0f}  {m['flr']:.3f}  {m['v']:.4f}  {m['n_launched']:>7d}")
    tab_rows.append((lam, m["flr"], m["v"], m["n_launched"]))

print("\n% LaTeX rows for tab:lagrangian (mu_0=-0.1, alpha=0.05, c=0.05):")
for lam, flr, v, nl in tab_rows:
    print(f"        {lam:.0f} & {flr:.3f} & {v:.4f} & {nl:,} \\\\".replace(",", "{,}"))
