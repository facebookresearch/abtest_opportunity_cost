# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %% [markdown]
# # Type-I error control via the e-process-gated Bellman
#
# The unconstrained Bellman maximizes reward but can launch versions with
# theta <= 0. Gating launch on a truncated-prior Bayes-factor e-process
# E_k > 1/alpha controls the worst-case type-I error P(launch | theta <= 0) at
# alpha for any stopping rule (Ville's inequality).
#
# Produces `tab:flr`: worst-case type-I error, FLR, lift, and average launches
# for the baselines, the unconstrained Bellman, and the e-process gate at several
# alpha levels.

# %%
import numpy as np
from methods import (
    BellmanCriterion,
    cached_criterion,
    ConstrainedBellmanCriterion,
    GaussSPParams,
    PPoSCriterion,
    simulate,
    ZCriterion,
)
from plotting import Z_CRIT

# %%
a0 = -0.1
sigma0 = 1.0
delta = 5.0
n_sims = 300
n_rounds = 2000
alpha_values = [0.5, 0.2, 0.1, 0.05]

k_final = 50
sigma_star = 1.0 / np.sqrt(1.0 / sigma0**2 + k_final / delta**2)
params = GaussSPParams(a0=a0, sigma0=sigma0, delta=delta, tau=1.0, n_rounds=n_rounds)
print(f"a0={a0}, sigma0={sigma0}, delta={delta}, sigma*={sigma_star:.4f}")

# %%
bellman = cached_criterion(
    BellmanCriterion, a0=a0, sigma0=sigma0, delta=delta, n_a=501, k_max=150, n_iter=500
)
constrained = {
    alpha: cached_criterion(
        ConstrainedBellmanCriterion,
        a0=a0,
        sigma0=sigma0,
        delta=delta,
        alpha=alpha,
        n_a=501,
        k_max=150,
        n_iter=500,
    )
    for alpha in alpha_values
}
criteria = {
    "z": ZCriterion(z=Z_CRIT),
    "z-optimistic": ZCriterion(z=Z_CRIT, sigma_min=sigma_star),
    "z-conservative": ZCriterion(z=Z_CRIT, sigma_min=sigma_star, conservative=True),
    "PPOS": PPoSCriterion(z=Z_CRIT, sigma_star=sigma_star),
    "Bellman": bellman,
}
for alpha in alpha_values:
    criteria[f"CB alpha={alpha}"] = constrained[alpha]


# %%
def type_i_at_zero(crit, seed=99991):
    """Worst-case type-I error: launch rate among resolved versions with theta fixed at 0."""
    delta2 = delta**2
    n_launch = n_resolved = 0
    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        a_hat, s2 = a0, sigma0**2
        for _ in range(n_rounds):
            act = crit.decide(a_hat, float(np.sqrt(s2)))
            if act != "continue":
                n_resolved += 1
                n_launch += int(act == "launch")
                a_hat, s2 = a0, sigma0**2
            b = float(rng.normal(0.0, delta))
            pp, po = 1.0 / s2, 1.0 / delta2
            a_hat, s2 = (pp * a_hat + po * b) / (pp + po), 1.0 / (pp + po)
    return n_launch / max(n_resolved, 1)


results = {}
for name, crit in criteria.items():
    n_launch = n_false = 0
    v_finals = []
    for i in range(n_sims):
        res = simulate(params, crit, rng=np.random.default_rng(42 + i))
        v_finals.append(res.v)
        for arm in res.arm_traces:
            if arm.outcome == "launched":
                n_launch += 1
                if arm.true_value < 0:
                    n_false += 1
    results[name] = {
        "flr": n_false / n_launch if n_launch else float("nan"),
        "v": float(np.mean(v_finals)),
        "type_i": type_i_at_zero(crit),
        "avg_launched": n_launch / n_sims,
    }
    r = results[name]
    print(
        f"{name:<16} type-I={r['type_i']:.4f}  FLR={r['flr']:.4f}  "
        f"v={r['v']:.5f}  avg_launched={r['avg_launched']:.1f}"
    )

# %% [markdown]
# ## Table `tab:flr` (Type-I / lift / avg launches per sim)


# %%
def _texrow(label, alpha, r):
    a = "---" if alpha is None else f"{alpha:.2f}"
    return f"        {label} & {a} & {r['type_i']:.3f} & {r['v']:.4f} & {r['avg_launched']:.1f} \\\\"


print("% LaTeX rows for tab:flr:")
print(_texrow(r"$z$-criterion ($z=1.96$)", None, results["z"]))
print(_texrow(r"\quad $k_{\max}=50$, optimistic", None, results["z-optimistic"]))
print(_texrow(r"\quad $k_{\max}=50$, conservative", None, results["z-conservative"]))
print(_texrow(rf"PPOS ($\sigma^* = {sigma_star:.2f}$)", None, results["PPOS"]))
print(_texrow("Bellman (unconstrained)", None, results["Bellman"]))
for alpha in alpha_values:
    print(_texrow("E-process gate", alpha, results[f"CB alpha={alpha}"]))
