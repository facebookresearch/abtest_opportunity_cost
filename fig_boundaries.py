# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %% [markdown]
# # Decision boundaries: z-criterion vs PPOS vs Bellman
#
# Decision boundaries in posterior (mu_hat, sigma_hat) space for two negative
# priors (mu_0 = -0.5 and mu_0 = -0.1; sigma_0 and delta fixed). Each criterion
# has two boundaries: **solid = discard**, **dashed = launch**; the wedge
# between them is CONTINUE. Both priors are negative because at delta = 5 the
# Bellman policy only has a continue region for mu_0 < 0 (for mu_0 >= 0 it
# launches immediately, with no wedge to plot).
#
# Produces `boundaries.pdf` (paper Fig. `fig:boundaries`).

# %%
import numpy as np
from matplotlib.lines import Line2D
from methods import AdaptiveBellmanCriterion, cached_criterion
from plotting import apply_paper_style, save_figure
from scipy.stats import norm

# %%
a0_values = [-0.5, -0.1]
sigma0, delta = 1.0, 5.0
cost = 0.05  # per-version sampling cost for the cost-aware Bellman overlay
z_crit = 1.96  # z-criterion threshold (two-sided 0.025)
z_sig = norm.ppf(0.95)  # PPOS significance threshold (one-sided 0.05)

# Shared maximum sample size: the z-deadline forces a decision and PPOS plans
# its final analysis at the same k_max horizon (a fair comparison).
k_max = 50
sigma_min_z = 1.0 / np.sqrt(1.0 / sigma0**2 + k_max / delta**2)  # z-deadline cutoff
sigma_star = sigma_min_z  # PPOS planned-final posterior std (same horizon)

# Grid over posterior mean (a_hat) and posterior std (sigma_hat).
A, S = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(0.02, sigma0, 500))

print(f"a0_values={a0_values}, sigma0={sigma0}, delta={delta}, z={z_crit}")
print(f"k_max={k_max}, sigma_min_z=sigma*={sigma_star:.4f}")


# %%
def sigma_star_g_analytic(a0_val, sigma0_val):
    """Analytic g-criterion sigma* -- an over-estimate used only to size the grid."""
    z0 = a0_val / sigma0_val
    h = float(norm.pdf(z0) + z0 * norm.cdf(z0))
    return sigma0_val * (h * np.sqrt(2 * np.pi)) ** (1.0 / 3.0)


def proper_kmax(a0_val, sigma0_val, delta_val):
    """Heuristic k_max so the Bellman grid reaches well below sigma*."""
    delta_fast = min(delta_val, 3.0)
    s_star_g = sigma_star_g_analytic(a0_val, sigma0_val)
    target_fast = s_star_g * 0.4
    m0 = 1.0 / sigma0_val**2
    km_fast = max(int((1.0 / target_fast**2 - m0) / (1.0 / delta_fast**2)) + 10, 100)
    bc_fast = cached_criterion(
        AdaptiveBellmanCriterion,
        a0=a0_val,
        sigma0=sigma0_val,
        delta=delta_fast,
        n_z=101,
        k_max=km_fast,
        n_iter=100,
    )
    _ra, rs, _la, ls = bc_fast.extract_boundaries()
    s_star_fast = min(rs.min(), ls.min()) if len(rs) > 0 else s_star_g * 0.3
    sigma_target = s_star_fast * 0.7
    return max(int((1.0 / sigma_target**2 - m0) / (1.0 / delta_val**2) * 1.1) + 10, 100)


# %%
# Bayesian PPOS grid is a0-independent: P(significant at the planned final).
valid = S > sigma_star
PPOS = np.where(
    valid,
    norm.cdf(
        (A - z_sig * sigma_star) / np.sqrt(np.maximum(S**2 - sigma_star**2, 1e-30))
    ),
    np.nan,
)

# Precision terms shared across priors (depend only on S, sigma0, sigma*, delta).
prec_prior = 1.0 / sigma0**2
prec_post = 1.0 / S**2
prec_data = prec_post - prec_prior
prec_data_final = 1.0 / sigma_star**2 - prec_prior
prec_future = np.maximum(prec_data_final - prec_data, 1e-12)
min_prec_data = 1.0 / delta**2

# Per-prior: Bellman boundaries (these depend on a0). The Bellman policy only
# has a continue wedge for a0 < 0; the problem is NOT mirror-symmetric (the
# launch reward max(a, 0) is one-sided), so each prior is solved directly.
panel = {}
for a0 in a0_values:
    assert a0 < 0, "Bellman has no continue wedge for a0 >= 0 at this delta"
    bellman_kmax = proper_kmax(a0, sigma0, delta)
    bellman = cached_criterion(
        AdaptiveBellmanCriterion,
        a0=a0,
        sigma0=sigma0,
        delta=delta,
        n_z=501,
        k_max=bellman_kmax,
        n_iter=500,
    )
    b_dis_a, b_dis_s, b_lau_a, b_lau_s = bellman.extract_boundaries()
    sigma_star_B = min(b_dis_s.min(), b_lau_s.min())

    # Cost-aware Bellman: per-renewal cost widens the wedge, so grid deeper.
    bellman_cost = cached_criterion(
        AdaptiveBellmanCriterion,
        a0=a0,
        sigma0=sigma0,
        delta=delta,
        n_z=501,
        k_max=int(bellman_kmax * 1.6),
        n_iter=500,
        arm_cost=cost,
    )
    bc_dis_a, bc_dis_s, bc_lau_a, bc_lau_s = bellman_cost.extract_boundaries()

    panel[a0] = {
        "b_dis_a": b_dis_a,
        "b_dis_s": b_dis_s,
        "b_lau_a": b_lau_a,
        "b_lau_s": b_lau_s,
        "bc_dis_a": bc_dis_a,
        "bc_dis_s": bc_dis_s,
        "bc_lau_a": bc_lau_a,
        "bc_lau_s": bc_lau_s,
        "sigma_star_B": sigma_star_B,
    }
    print(f"a0={a0:+.1f}: bellman_kmax={bellman_kmax}, sigma*_B={sigma_star_B:.4f}")

# %%
import matplotlib.pyplot as plt

apply_paper_style()

sigmas = np.linspace(0.0, sigma0, 200)

# z-deadline (k_max): the z-criterion lines clipped at sigma_min, then a
# horizontal segment to a=0 at the deadline (force launch if a>0, else discard).
_diag_s = np.linspace(sigma0, sigma_min_z, 100)
zdl_dis_a = np.append(-z_crit * _diag_s, 0.0)
zdl_dis_s = np.append(_diag_s, sigma_min_z)
zdl_lau_a = np.append(z_crit * _diag_s, 0.0)
zdl_lau_s = np.append(_diag_s, sigma_min_z)


def label_regions(ax):
    ax.text(-1.6, 0.12, "DISCARD", fontsize=18, color="gray", ha="center")
    ax.text(1.6, 0.12, "LAUNCH", fontsize=18, color="gray", ha="center")
    ax.text(0.0, 0.85, "CONTINUE", fontsize=18, color="gray", ha="center")
    ax.axvline(0, color="gray", lw=0.5, ls=":")


def draw_bayesian(ax, a0, xlim=(-2.2, 2.2)):
    d = panel[a0]
    ax.plot(-z_crit * sigmas, sigmas, color="tab:olive", lw=3, ls="-")
    ax.plot(z_crit * sigmas, sigmas, color="tab:olive", lw=3, ls="--")
    ax.plot(zdl_dis_a, zdl_dis_s, color="tab:orange", lw=3, ls="-")
    ax.plot(zdl_lau_a, zdl_lau_s, color="tab:orange", lw=3, ls="--")
    ax.contour(
        A, S, PPOS, levels=[0.05], colors="seagreen", linewidths=3, linestyles="-"
    )
    ax.contour(
        A, S, PPOS, levels=[0.95], colors="seagreen", linewidths=3, linestyles="--"
    )
    ax.plot(d["b_dis_a"], d["b_dis_s"], color="navy", lw=3.5, ls="-")
    ax.plot(d["b_lau_a"], d["b_lau_s"], color="navy", lw=3.5, ls="--")
    ax.plot(d["bc_dis_a"], d["bc_dis_s"], color="crimson", lw=3.5, ls="-")
    ax.plot(d["bc_lau_a"], d["bc_lau_s"], color="crimson", lw=3.5, ls="--")
    ax.plot(a0, sigma0, "k*", ms=14, zorder=5)
    label_regions(ax)
    ax.set_xlabel(r"Posterior mean $\hat{\mu}$", fontsize=22)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, sigma0 * 1.02)
    ax.tick_params(labelsize=18)
    ax.set_title(
        rf"$\mu_0={a0},\ \sigma_0={sigma0:.0f},\ \delta={delta:.0f}$", fontsize=22
    )


# %%
# Two plot panels + a third narrow panel holding the legend.
fig, axes = plt.subplots(
    1, 3, figsize=(20, 7), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.55]}
)
xlims = [(-2.2, 2.2), (-2.2, 2.2)]
for ax, a0, xlim in zip(axes[:2], a0_values, xlims):
    draw_bayesian(ax, a0, xlim=xlim)
axes[0].set_ylabel(r"Posterior std $\hat\sigma$", fontsize=22)

axes[2].axis("off")
axes[2].legend(
    handles=[
        Line2D(
            [0], [0], color="tab:olive", lw=3, label=rf"$z$-criterion ($z={z_crit}$)"
        ),
        Line2D(
            [0],
            [0],
            color="tab:orange",
            lw=3,
            label=rf"$z$-criterion ($z={z_crit}$, $k_{{\max}}={k_max}$)",
        ),
        Line2D(
            [0],
            [0],
            color="seagreen",
            lw=3,
            label=rf"PPOS ($\sigma^*={sigma_star:.2f}$)",
        ),
        Line2D([0], [0], color="navy", lw=3.5, label=r"Bellman ($c=0$)"),
        Line2D([0], [0], color="crimson", lw=3.5, label=rf"Bellman ($c={cost:g}$)"),
        Line2D([0], [0], color="gray", lw=2, ls="-", label="discard boundary"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="launch boundary"),
    ],
    loc="center left",
    fontsize=20,
    frameon=False,
)
plt.tight_layout()
save_figure(fig, "boundaries.pdf")
plt.show()
