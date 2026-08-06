# %% [markdown]
# # Cost ablation: net reward vs. per-version sampling cost
#
# The per-version cost c is paid on every renewal (launch or discard), so net
# reward decomposes exactly as R(c) = L - c*nu, with L the gross launched lift
# per round and nu the renewals per round.
#
# Consequences: (1) **cost-blind policies are linear in c** -- the z-variants and
# PPOS don't change behavior with c, so their whole R(c) line comes from a single
# c=0 run; only the **cost-aware Bellman** re-solves per c. (2) The Bellman widens
# its continue wedge (lowers nu) as c grows, trading throughput for cost-robustness.
#
# Produces `cost_ablation.pdf` and `tab:cost-ablation`.

# %%
import numpy as np
from methods import (
    BellmanCriterion,
    cached_criterion,
    GaussSPParams,
    PPoSCriterion,
    simulate,
    ZCriterion,
)
from plotting import Z_CRIT

# %%
a0_values = [-0.5, -0.1, 0.1]
sigma0 = 1.0
delta = 5.0
tau = 1.0
n_sims = 300
n_rounds = 2000
seed = 42
T = tau * (n_rounds + 1)  # time normalizer matching simulate()

cost_grid = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
k_max_base = 50  # headline baseline deadline (as in the main figure)


def sigma_star_of(k):
    return 1.0 / np.sqrt(1.0 / sigma0**2 + k / delta**2)


sigma_min_base = sigma_star_of(k_max_base)
k_max_grid = [25, 50, 75, 100, 150]  # horizons swept when steelmanning baselines
table_a0 = -0.1  # the printed table uses a single prior, matching the paper

print(f"sigma0={sigma0}, delta={delta}, cost grid: {cost_grid}")


# %%
def lift_and_renewals(a0, crit):
    """Mean (L, nu) at c=0: gross lift/round and renewals/round."""
    p0 = GaussSPParams(
        a0=a0, sigma0=sigma0, delta=delta, tau=tau, n_rounds=n_rounds, cost=0.0
    )
    Ls = np.empty(n_sims)
    nus = np.empty(n_sims)
    for i in range(n_sims):
        res = simulate(p0, crit, rng=np.random.default_rng(seed + i))
        Ls[i] = res.v  # net reward at c=0 == gross lift rate L
        nus[i] = (res.n_launched + res.n_rejected) / T
    return float(Ls.mean()), float(nus.mean())


def blind_reward(L, nu, c):
    return L - c * nu


def breakeven(L, nu):
    return L / nu if nu > 0 else float("inf")


def bellman_k_max(c):
    """Horizon truncation for the cost-aware Bellman, enlarged with cost so the
    forced-renewal deadline stays non-binding across the cost grid."""
    if c <= 0.1:
        return 200
    if c <= 0.2:
        return 500
    return 1000


def bellman_solve(a0, c):
    k_max = bellman_k_max(c)
    return cached_criterion(
        BellmanCriterion,
        a0=a0,
        sigma0=sigma0,
        delta=delta,
        n_a=501,
        k_max=k_max,
        n_iter=max(500, 3 * k_max),
        arm_cost=c,
    )


def bellman_reward_and_nu(a0, c):
    crit = bellman_solve(a0, c)
    p = GaussSPParams(
        a0=a0, sigma0=sigma0, delta=delta, tau=tau, n_rounds=n_rounds, cost=c
    )
    vs = np.empty(n_sims)
    nus = np.empty(n_sims)
    for i in range(n_sims):
        res = simulate(p, crit, rng=np.random.default_rng(seed + i))
        vs[i] = res.v
        nus[i] = (res.n_launched + res.n_rejected) / T
    return float(vs.mean()), float(nus.mean())


# %% [markdown]
# ## Sanity check: R(c) = L - c*nu holds exactly for cost-blind policies

# %%
_chk = ZCriterion(z=Z_CRIT, sigma_min=sigma_min_base, conservative=True)
_L, _nu = lift_and_renewals(table_a0, _chk)
_c = 0.05
_p = GaussSPParams(
    a0=table_a0, sigma0=sigma0, delta=delta, tau=tau, n_rounds=n_rounds, cost=_c
)
_direct = np.mean(
    [simulate(_p, _chk, rng=np.random.default_rng(seed + i)).v for i in range(n_sims)]
)
assert abs(_direct - blind_reward(_L, _nu, _c)) < 1e-9, (
    "cost-blind decomposition broken"
)
print(f"OK: direct R(0.05)={_direct:.6f} == L - c*nu={blind_reward(_L, _nu, _c):.6f}")


# %% [markdown]
# ## (L, nu) for every method and prior; Bellman simulated at each cost


# %%
def headline_baselines():
    return {
        "z (inf-horizon)": ZCriterion(z=Z_CRIT),
        "z (optimistic, k=50)": ZCriterion(z=Z_CRIT, sigma_min=sigma_min_base),
        "z (conservative, k=50)": ZCriterion(
            z=Z_CRIT, sigma_min=sigma_min_base, conservative=True
        ),
        "PPOS (k=50)": PPoSCriterion(z=Z_CRIT, sigma_star=sigma_min_base),
    }


blind = {}  # blind[a0][name] = (L, nu)
bellman = {}  # bellman[a0][c] = (R, nu)
bellman_frozen = {}  # bellman_frozen[a0] = (L0, nu0) from the c=0 policy
for a0 in a0_values:
    blind[a0] = {
        name: lift_and_renewals(a0, crit) for name, crit in headline_baselines().items()
    }
    bellman[a0] = {c: bellman_reward_and_nu(a0, c) for c in cost_grid}
    bellman_frozen[a0] = lift_and_renewals(a0, bellman_solve(a0, 0.0))
    print(
        f"mu0={a0:>5}  Bellman nu: c=0 {bellman[a0][0.0][1]:.4f} -> "
        f"c={cost_grid[-1]:g} {bellman[a0][cost_grid[-1]][1]:.4f}"
    )


# %% [markdown]
# ## Steelman: best cost-blind baseline per cost (sweep k_max, keep the best R)


# %%
def steel_families(k):
    sm = sigma_star_of(k)
    return {
        "z-cons (best k)": ZCriterion(z=Z_CRIT, sigma_min=sm, conservative=True),
        "PPOS (best k)": PPoSCriterion(z=Z_CRIT, sigma_star=sm),
    }


steel_lines = {}  # steel_lines[a0][family] = list of (k, L, nu)
for a0 in a0_values:
    steel_lines[a0] = {"z-cons (best k)": [], "PPOS (best k)": []}
    for k in k_max_grid:
        for fam, crit in steel_families(k).items():
            steel_lines[a0][fam].append((k, *lift_and_renewals(a0, crit)))


def steel_best(a0, fam, c):
    best_R, best_k = -np.inf, None
    for k, L, nu in steel_lines[a0][fam]:
        R = blind_reward(L, nu, c)
        if R > best_R:
            best_R, best_k = R, k
    return best_R, best_k


# %% [markdown]
# ## Primary table (mu_0 = -0.1): net reward R at each cost

# %%
a0 = table_a0
rows = []
for name, (L, nu) in blind[a0].items():
    rows.append((name, nu, [blind_reward(L, nu, c) for c in cost_grid]))
rows.append(
    (
        "Bellman (cost-aware)",
        bellman[a0][0.0][1],
        [bellman[a0][c][0] for c in cost_grid],
    )
)
Lf, nuf = bellman_frozen[a0]
rows.append(
    ("Bellman (frozen c=0)", nuf, [blind_reward(Lf, nuf, c) for c in cost_grid])
)
for fam in ("z-cons (best k)", "PPOS (best k)"):
    rows.append((fam, None, [steel_best(a0, fam, c)[0] for c in cost_grid]))

hdr = ["Method", "nu0"] + [f"c={c:g}" for c in cost_grid]
print("  ".join(f"{h:>12}" if i else f"{h:<22}" for i, h in enumerate(hdr)))
for name, nu, cells in rows:
    nu_s = f"{nu:.3f}" if nu is not None else "  -- "
    print(f"{name:<22}  {nu_s:>12}  " + "  ".join(f"{v:12.4f}" for v in cells))

print(f"\n% LaTeX rows for tab:cost-ablation (mu_0={a0}):")
print("% Method & nu0 & " + " & ".join(f"$c={c:g}$" for c in cost_grid) + " \\\\")
for name, nu, cells in rows:
    nu_s = f"{nu:.3f}" if nu is not None else "--"
    print(f"{name} & {nu_s} & " + " & ".join(f"{v:.4f}" for v in cells) + " \\\\")

# %% [markdown]
# ## Figure: net reward vs. cost across the three priors

# %%
import matplotlib.pyplot as plt
from plotting import apply_paper_style, save_figure

apply_paper_style()

styles = {
    "z (inf-horizon)": {"color": "tab:olive", "ls": "--", "lw": 3},
    "z (optimistic, k=50)": {"color": "tab:orange", "ls": "--", "lw": 3},
    "z (conservative, k=50)": {"color": "mediumpurple", "ls": "--", "lw": 3},
    "PPOS (k=50)": {"color": "seagreen", "ls": "-.", "lw": 3},
}
cc = np.linspace(0.0, cost_grid[-1], 200)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), squeeze=False)
for ci, a0 in enumerate(a0_values):
    ax = axes[0][ci]
    for name, (L, nu) in blind[a0].items():
        ax.plot(cc, L - cc * nu, label=name, **styles[name])
    for fam, col in (
        ("z-cons (best k)", "mediumpurple"),
        ("PPOS (best k)", "seagreen"),
    ):
        env = [steel_best(a0, fam, c)[0] for c in cost_grid]
        ax.plot(cost_grid, env, color=col, ls=":", lw=2.5, marker="s", ms=5, label=fam)
    Lf, nuf = bellman_frozen[a0]
    ax.plot(
        cc, Lf - cc * nuf, color="navy", ls="--", lw=2.5, label="Bellman (frozen c=0)"
    )
    ax.plot(
        cost_grid,
        [bellman[a0][c][0] for c in cost_grid],
        color="navy",
        ls="-",
        lw=3.5,
        marker="o",
        ms=7,
        label="Bellman (cost-aware)",
    )
    ax.axhline(0.0, color="0.6", lw=1.0, ls=":")
    ax.set_xlim(0, cost_grid[-1])
    ax.set_xlabel(r"sampling cost $c$", fontsize=22)
    ax.set_title(
        rf"$\mu_0={a0},\ \sigma_0={sigma0:.0f},\ \delta={delta:.0f}$", fontsize=20
    )
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=18)
    ymax_p = bellman[a0][0.0][0] * 1.15
    ax.set_ylim(-0.35 * ymax_p, ymax_p)
axes[0][0].set_ylabel(r"net reward $R$", fontsize=22)
_h, _l = axes[0][0].get_legend_handles_labels()
fig.legend(
    _h,
    _l,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.06),
    ncol=4,
    fontsize=18,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save_figure(fig, "cost_ablation.pdf")
plt.show()
