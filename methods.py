"""Core algorithms for the Online Version Testing (OVT) paper.

Self-contained reference implementation of every decision rule and simulation
used to produce the figures and tables in the paper. Depends only on numpy and
scipy; each figure notebook imports what it needs from here.

Contents
--------
Model / simulation
    GaussSPParams        parameters of the Gaussian stream problem
    simulate             one long simulation of a policy over a version stream
    SimResult, ArmTrace  per-run results and per-version trajectories

Decision rules (all implement ``decide(a_hat, sigma_hat) -> action``)
    ZCriterion, FreqZCriterion          z-score thresholds (Bayes / frequentist)
    PPoSCriterion, FreqPPoSCriterion    predictive probability of success
    BellmanCriterion                    reward-optimal policy (value iteration)
    ConstrainedBellmanCriterion         type-I control via an e-process gate
    LagrangianBellmanCriterion          FLR control via a Lagrangian penalty
    AdaptiveBellmanCriterion            Bellman on z=a/sigma coords (smooth
                                        boundary extraction for the boundary plot)

Helpers
    cached_criterion       build + pickle-cache a (slow) Bellman-family policy
    cached_reward_curves   run + JSONL-cache mean reward curves for a set of rules
    eprocess_truncated_bf  truncated-prior Bayes-factor e-process for H0: theta<=0

All caches live under ``cache/`` next to this file; delete it to recompute.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast, Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

Action = Literal["launch", "reject", "continue"]

# All pickle / JSONL caches live here (created on first use). Safe to delete.
CACHE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


# ==========================================================================
# Caching helpers
# ==========================================================================
def cached_criterion(
    cls: type, cache_dir: str | None = None, **kwargs: object
) -> "Criterion":
    """Construct a Criterion, caching the (expensive) result to disk.

    Bellman-family policies solve a value-iteration problem in ``__init__``;
    this memoizes the constructed object by (class, kwargs) so repeated runs
    (and reruns across notebooks) are instant. Always build Bellman-family
    criteria through this rather than calling the class directly.
    """
    if cache_dir is None:
        cache_dir = os.path.join(CACHE_DIR, "criteria")
    os.makedirs(cache_dir, exist_ok=True)

    key = f"{cls.__name__}|" + "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    # sha256 used only as a filename hash for the cache key, not for security.
    fname = hashlib.sha256(key.encode()).hexdigest()[:16] + ".pkl"
    cache_path = os.path.join(cache_dir, fname)

    # The cache holds only objects this module wrote to a local directory, so
    # unpickling is safe here.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    obj = cls(**kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump(obj, f)
    return obj


def eprocess_truncated_bf(
    a_hat: np.ndarray | float,
    sigma_hat: float,
    a0: float,
    sigma0: float,
    a_alt: float | None = None,
    sigma_alt: float | None = None,
) -> np.ndarray | float:
    """Truncated-prior Bayes-factor e-process for H_0: theta <= 0.

    Uses N(a_alt, sigma_alt^2) truncated to theta > 0 as the alternative,
    defaulting to the prior (a0, sigma0). Because it is a valid e-process,
    gating a launch on E_k >= 1/alpha controls the type-I error at alpha for
    any (data-dependent) stopping rule, by Ville's inequality.
    """
    if a_alt is None:
        a_alt = a0
    if sigma_alt is None:
        sigma_alt = sigma0
    a_hat = np.asarray(a_hat)
    m_d = 1.0 / sigma_hat**2 - 1.0 / sigma0**2
    m_p = m_d + 1.0 / sigma_alt**2
    sigma_p = 1.0 / np.sqrt(m_p)
    mu_p = (a_hat / sigma_hat**2 - a0 / sigma0**2 + a_alt / sigma_alt**2) / m_p
    log_ep = (
        np.log(sigma_p / sigma_alt)
        + mu_p**2 / (2 * sigma_p**2)
        - a_alt**2 / (2 * sigma_alt**2)
        + norm.logcdf(mu_p / sigma_p)
        - norm.logcdf(a_alt / sigma_alt)
    )
    return np.exp(log_ep)


# ==========================================================================
# Problem definition
# ==========================================================================
@dataclass
class GaussSPParams:
    """Parameters for the Gaussian stream problem GaussSP(a0, sigma0^2, delta^2, tau).

    A stream of versions with true lifts theta_i ~ N(a0, sigma0^2) is observed
    through per-round noise of std ``delta``; each round advances the clock by
    ``tau``. ``cost`` is a fixed sampling cost charged on every renewal (launch
    or discard).
    """

    a0: float  # prior mean of version lifts
    sigma0: float  # prior std of version lifts
    delta: float  # measurement noise std per round
    tau: float  # time per round
    n_rounds: int  # total number of rounds to simulate
    cost: float = 0.0  # per-version sampling cost (subtracted each renewal)


class Criterion(ABC):
    """A decision rule mapping the posterior state (a_hat, sigma_hat) to an action."""

    @abstractmethod
    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        """Return 'launch', 'reject', or 'continue' for the current posterior."""
        ...

    @property
    def label(self) -> str:
        return self.__class__.__name__


# ==========================================================================
# Myopic baselines
# ==========================================================================
class ZCriterion(Criterion):
    """z-criterion: threshold on the posterior z-score a_hat / sigma_hat.

    reject if  a_hat / sigma_hat < -z ;  launch if  a_hat / sigma_hat > z.

    Optional ``sigma_min`` sets a deadline: once sigma_hat <= sigma_min a
    decision is forced. With ``conservative=False`` the deadline launches on the
    sign of a_hat (optimistic); with ``conservative=True`` it launches only if
    still significant. Convert a max round count via
    sigma_min = 1/sqrt(1/sigma0^2 + k_max/delta^2).
    """

    def __init__(
        self,
        z: float = 1.96,
        sigma_min: float | None = None,
        conservative: bool = False,
    ) -> None:
        if conservative and sigma_min is None:
            raise ValueError("conservative=True requires sigma_min")
        self.z = z
        self.sigma_min = sigma_min
        self.conservative = conservative

    def _deadline_action(self, a_hat: float, sigma_hat: float) -> Action:
        if self.conservative:
            zscore = a_hat / sigma_hat if sigma_hat > 0 else float("inf")
            return "launch" if zscore > self.z else "reject"
        return "launch" if a_hat > 0 else "reject"

    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        if sigma_hat <= 0:
            return "launch" if a_hat > 0 else "reject"
        if self.sigma_min is not None and sigma_hat <= self.sigma_min:
            return self._deadline_action(a_hat, sigma_hat)
        zscore = a_hat / sigma_hat
        if zscore < -self.z:
            return "reject"
        if zscore > self.z:
            return "launch"
        return "continue"

    @property
    def label(self) -> str:
        if self.sigma_min is not None:
            return f"z-criterion (z={self.z}, sigma_min={self.sigma_min:.3f})"
        return f"z-criterion (z={self.z})"


class FreqZCriterion(Criterion):
    """Frequentist z-criterion: threshold on the sample z-statistic (ignores the prior).

    Recovers the sample z-score from the posterior state:
        z = (a_hat/sigma_hat^2 - a0/sigma0^2) / sqrt(1/sigma_hat^2 - 1/sigma0^2).
    Optional ``sigma_min`` sets a deadline (see ZCriterion).
    """

    def __init__(
        self, z: float, a0: float, sigma0: float, sigma_min: float | None = None
    ) -> None:
        self.z = z
        self.a0 = a0
        self.sigma0 = sigma0
        self.sigma_min = sigma_min

    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        if self.sigma_min is not None and sigma_hat <= self.sigma_min:
            return "launch" if a_hat > 0 else "reject"
        prec_post = 1.0 / sigma_hat**2
        prec_prior = 1.0 / self.sigma0**2
        prec_data = prec_post - prec_prior
        if prec_data < 1e-12:
            return "continue"
        numerator = a_hat * prec_post - self.a0 * prec_prior
        zscore = numerator / np.sqrt(prec_data)
        if zscore < -self.z:
            return "reject"
        if zscore > self.z:
            return "launch"
        return "continue"

    @property
    def label(self) -> str:
        if self.sigma_min is not None:
            return f"freq-z-criterion (z={self.z}, sigma_min={self.sigma_min:.3f})"
        return f"freq-z-criterion (z={self.z})"


class PPoSCriterion(Criterion):
    """Predictive Probability of Success.

    At each round computes the probability that the final analysis (at
    ``sigma_star``) would be significant, launching when it exceeds ``launch_q``
    and rejecting when it falls below ``1 - launch_q``. Once sigma_hat has
    reached sigma_star it makes a hard z-test decision.

        sigma_star = 1 / sqrt(1/sigma0^2 + k_final/delta^2).
    """

    def __init__(self, z: float, sigma_star: float, launch_q: float = 0.95) -> None:
        self.z = z
        self.sigma_star = sigma_star
        self.launch_q = launch_q

    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        if sigma_hat <= self.sigma_star:
            if a_hat / sigma_hat > self.z:
                return "launch"
            return "reject"
        spread = np.sqrt(sigma_hat**2 - self.sigma_star**2)
        ppos = float(norm.cdf((a_hat - self.z * self.sigma_star) / spread))
        if ppos > self.launch_q:
            return "launch"
        if ppos < 1.0 - self.launch_q:
            return "reject"
        return "continue"

    @property
    def label(self) -> str:
        return f"PPoS (z={self.z}, sigma*={self.sigma_star:.2f})"


class FreqPPoSCriterion(Criterion):
    """Frequentist Predictive Probability of Success.

    Like PPoSCriterion but built on the sample mean and data precision (the
    prior is stripped out), using the frequentist z-statistic at the planned
    final analysis.
    """

    def __init__(self, z: float, sigma_star: float, a0: float, sigma0: float) -> None:
        self.z = z
        self.a0 = a0
        self.sigma0 = sigma0
        prec_prior = 1.0 / sigma0**2
        self.prec_prior = prec_prior
        self.prec_data_final = 1.0 / sigma_star**2 - prec_prior

    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        prec_post = 1.0 / sigma_hat**2
        prec_data = prec_post - self.prec_prior
        if prec_data < 1e-12:
            return "continue"

        x_bar = (a_hat * prec_post - self.a0 * self.prec_prior) / prec_data

        if prec_data >= self.prec_data_final:
            z_stat = x_bar * np.sqrt(prec_data)
            if z_stat > self.z:
                return "launch"
            return "reject"

        prec_future = self.prec_data_final - prec_data
        z_final_mean = x_bar * np.sqrt(self.prec_data_final)
        z_final_std = np.sqrt(prec_future / self.prec_data_final)

        ppos = float(norm.cdf((z_final_mean - self.z) / z_final_std))
        if ppos > 0.95:
            return "launch"
        if ppos < 0.05:
            return "reject"
        return "continue"

    @property
    def label(self) -> str:
        return f"Freq PPoS (z={self.z})"


# ==========================================================================
# Bellman family: value iteration on the (a_hat, k) grid
# ==========================================================================
class _BellmanDP(Criterion):
    """Shared value-iteration scaffolding for the (a_hat, k)-grid Bellman policies.

    Solves the average-reward Bellman equation on a discretized posterior-mean
    grid, one row per number of observations k (equivalently per sigma_hat).
    Subclasses supply the per-stop reward and the launch/reject split.
    """

    a_grid: np.ndarray
    V: np.ndarray
    _stop: np.ndarray

    def _setup(
        self, a0: float, sigma0: float, delta: float, n_a: int, k_max: int, n_iter: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Store parameters and build the (a-grid, transition, weights) arrays."""
        self.a0, self.sigma0, self.delta = a0, sigma0, delta
        self.n_a, self.k_max, self.n_iter = n_a, k_max, n_iter

        m0 = 1.0 / sigma0**2
        m_d = 1.0 / delta**2
        self.m0, self.m_d = m0, m_d
        prec = cast(np.ndarray, m0 + np.arange(k_max + 2) * m_d)

        half = max(6.0 * sigma0, 6.0)
        a_grid = np.linspace(-half, half, n_a)
        da = a_grid[1] - a_grid[0]
        self.a_grid = a_grid

        dvar = cast(np.ndarray, 1.0 / prec[:-1] - 1.0 / prec[1:])
        sg = np.sqrt(dvar) / da

        w_stop = cast(np.ndarray, norm.pdf(a_grid, loc=a0, scale=np.sqrt(dvar[0])))
        w_stop /= w_stop.sum()

        stop_rewards = np.array(
            [
                np.asarray(self._stop_reward(a_grid, 1.0 / np.sqrt(m0 + k * m_d)))
                for k in range(k_max + 1)
            ]
        )
        return a_grid, sg, w_stop, stop_rewards

    def _value_iteration(
        self, w_stop: np.ndarray, stop_rewards: np.ndarray, sg: np.ndarray, ref_idx: int
    ) -> np.ndarray:
        """Relative value iteration; returns the gauge-fixed value grid V."""
        k_max, n_a = self.k_max, self.n_a
        V = np.zeros((k_max + 1, n_a))
        for _ in range(self.n_iter):
            Vn = np.zeros_like(V)
            e_stop = w_stop @ V[min(1, k_max)]
            for k in range(k_max + 1):
                stop_val = stop_rewards[k] + e_stop
                if k < k_max:
                    cont_val = gaussian_filter1d(V[k + 1], sigma=sg[k], mode="nearest")
                    Vn[k] = np.maximum(stop_val, cont_val)
                else:
                    Vn[k] = stop_val
            V = Vn - Vn[0, ref_idx]
        return V

    def _stopping_policy(
        self,
        V: np.ndarray,
        w_stop: np.ndarray,
        stop_rewards: np.ndarray,
        sg: np.ndarray,
    ) -> np.ndarray:
        """Boolean stop/continue grid from the converged values."""
        k_max, n_a = self.k_max, self.n_a
        e_stop = w_stop @ V[min(1, k_max)]
        stop = np.zeros((k_max + 1, n_a), dtype=bool)
        for k in range(k_max + 1):
            stop_val = stop_rewards[k] + e_stop
            if k < k_max:
                cont_val = gaussian_filter1d(V[k + 1], sigma=sg[k], mode="nearest")
                stop[k] = stop_val >= cont_val
            else:
                stop[k] = True
        return stop

    def _solve(
        self, a0: float, sigma0: float, delta: float, n_a: int, k_max: int, n_iter: int
    ) -> None:
        a_grid, sg, w_stop, stop_rewards = self._setup(
            a0, sigma0, delta, n_a, k_max, n_iter
        )
        ref_idx = int(np.argmin(np.abs(a_grid - a0)))
        self.V = self._value_iteration(w_stop, stop_rewards, sg, ref_idx)
        self._stop = self._stopping_policy(self.V, w_stop, stop_rewards, sg)

    def _stop_reward(self, a_grid: np.ndarray, sigma_k: float) -> np.ndarray:
        raise NotImplementedError

    def _stop_action(self, a_hat: float, sigma_hat: float) -> Action:
        raise NotImplementedError

    def _is_launch_boundary(self, a_bnd: float, sigma_k: float) -> bool:
        raise NotImplementedError

    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        k = int(round((1.0 / max(sigma_hat, 1e-12) ** 2 - self.m0) / self.m_d))
        k = int(np.clip(k, 0, self.k_max))
        i = int(np.searchsorted(self.a_grid, a_hat).clip(0, len(self.a_grid) - 1))
        if self._stop[k, i]:
            return self._stop_action(a_hat, sigma_hat)
        return "continue"

    def extract_boundaries(
        self, sigma_min: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract (reject_a, reject_sigma, launch_a, launch_sigma) sorted by sigma."""
        ra, rs, la, ls = [], [], [], []
        for k in range(self.k_max + 1):
            sk = 1.0 / np.sqrt(self.m0 + k * self.m_d)
            if sk > self.sigma0 or sk < sigma_min:
                continue
            for i in range(len(self.a_grid) - 1):
                if self._stop[k][i] and not self._stop[k][i + 1]:
                    ra.append(0.5 * (self.a_grid[i] + self.a_grid[i + 1]))
                    rs.append(sk)
                elif not self._stop[k][i] and self._stop[k][i + 1]:
                    a_bnd = 0.5 * (self.a_grid[i] + self.a_grid[i + 1])
                    if self._is_launch_boundary(a_bnd, sk):
                        la.append(a_bnd)
                        ls.append(sk)
                    else:
                        ra.append(a_bnd)
                        rs.append(sk)
        out = [np.array(x) for x in [ra, rs, la, ls]]
        order_r = np.argsort(out[1]) if len(out[1]) > 0 else np.array([], dtype=int)
        order_l = np.argsort(out[3]) if len(out[3]) > 0 else np.array([], dtype=int)
        return out[0][order_r], out[1][order_r], out[2][order_l], out[3][order_l]


class BellmanCriterion(_BellmanDP):
    """Reward-optimal policy via value iteration on the (a_hat, k) grid.

    Maximizes long-run average reward: launch reward max(a_hat, 0), minus an
    optional per-renewal ``arm_cost``. This is the oracle the baselines are
    compared against.
    """

    def __init__(
        self,
        a0: float,
        sigma0: float,
        delta: float,
        n_a: int = 301,
        k_max: int = 100,
        n_iter: int = 500,
        arm_cost: float = 0.0,
    ) -> None:
        self.arm_cost = arm_cost
        self._solve(a0, sigma0, delta, n_a, k_max, n_iter)

    def _stop_reward(self, a_grid: np.ndarray, sigma_k: float) -> np.ndarray:
        return np.maximum(a_grid, 0.0) - self.arm_cost

    def _stop_action(self, a_hat: float, sigma_hat: float) -> Action:
        return "launch" if a_hat > 0 else "reject"

    def _is_launch_boundary(self, a_bnd: float, sigma_k: float) -> bool:
        return True

    @property
    def label(self) -> str:
        if self.arm_cost > 0:
            return f"Bellman (optimal, c={self.arm_cost})"
        return "Bellman (optimal)"


class ConstrainedBellmanCriterion(_BellmanDP):
    """Bellman with type-I control via a truncated-prior Bayes-factor e-process.

    Gates launch on E_k(a_hat, sigma_hat) > 1/alpha, where E_k is a valid
    e-process for H_0: theta <= 0. By Ville's inequality the type-I error is at
    most alpha for any stopping rule.
    """

    def __init__(
        self,
        a0: float,
        sigma0: float,
        delta: float,
        alpha: float = 0.05,
        a_alt: float | None = None,
        sigma_alt: float | None = None,
        n_a: int = 301,
        k_max: int = 100,
        n_iter: int = 500,
        arm_cost: float = 0.0,
    ) -> None:
        self.alpha = alpha
        self.arm_cost = arm_cost
        self.a_alt = a_alt if a_alt is not None else a0
        self.sigma_alt = sigma_alt if sigma_alt is not None else sigma0
        self._solve(a0, sigma0, delta, n_a, k_max, n_iter)

    def _eprocess(
        self, a_hat: np.ndarray | float, sigma_hat: float
    ) -> np.ndarray | float:
        return eprocess_truncated_bf(
            a_hat, sigma_hat, self.a0, self.sigma0, self.a_alt, self.sigma_alt
        )

    def _stop_reward(self, a_grid: np.ndarray, sigma_k: float) -> np.ndarray:
        launch_ok = self._eprocess(a_grid, sigma_k) > 1.0 / self.alpha
        return np.where(launch_ok, np.maximum(a_grid, 0.0), 0.0) - self.arm_cost

    def _stop_action(self, a_hat: float, sigma_hat: float) -> Action:
        if a_hat > 0 and self._eprocess(a_hat, sigma_hat) > 1.0 / self.alpha:
            return "launch"
        return "reject"

    def _is_launch_boundary(self, a_bnd: float, sigma_k: float) -> bool:
        return bool(self._eprocess(a_bnd, sigma_k) > 1.0 / self.alpha)

    @property
    def label(self) -> str:
        if self.sigma_alt != self.sigma0 or self.a_alt != self.a0:
            return (
                f"Constrained Bellman (alpha={self.alpha}, sigma_alt={self.sigma_alt})"
            )
        return f"Constrained Bellman (alpha={self.alpha})"


def lagrangian_launch_reward(
    a_hat: np.ndarray | float,
    sigma_hat: float,
    lam: float,
    alpha: float,
) -> np.ndarray | float:
    """Launch reward under the Lagrangian for FLR control: a_hat - lam*(Phi(-a_hat/sigma_hat) - alpha)."""
    posterior_error = norm.cdf(-np.asarray(a_hat) / sigma_hat)
    return np.asarray(a_hat) - lam * (posterior_error - alpha)


class LagrangianBellmanCriterion(_BellmanDP):
    """Bellman with false-launch-rate (FLR) control via a Lagrangian penalty.

    penalize_frr=False (default): penalize false launches only,
        launch_reward = a_hat - lam*(Phi(-a_hat/sigma_hat) - alpha), reject_reward = 0.
    penalize_frr=True: penalize both false launches and false rejects,
        launch_reward = a_hat - lam*Phi(-a_hat/sigma_hat),
        reject_reward = -lam*Phi(a_hat/sigma_hat)   (alpha is ignored).

    The smallest lam meeting a target FLR is found by bisection in the FLR notebook.
    """

    def __init__(
        self,
        a0: float,
        sigma0: float,
        delta: float,
        lam: float,
        alpha: float = 0.05,
        penalize_frr: bool = False,
        n_a: int = 301,
        k_max: int = 100,
        n_iter: int = 500,
        arm_cost: float = 0.0,
    ) -> None:
        self.lam, self.alpha = lam, alpha
        self.penalize_frr = penalize_frr
        self.arm_cost = arm_cost
        self._solve(a0, sigma0, delta, n_a, k_max, n_iter)

    def _reward(self, a: np.ndarray | float, sigma: float) -> np.ndarray | float:
        a = np.asarray(a)
        if self.penalize_frr:
            launch_r = a - self.lam * norm.cdf(-a / sigma)
            reject_r = -self.lam * norm.cdf(a / sigma)
            return np.maximum(launch_r, reject_r) - self.arm_cost
        lr = lagrangian_launch_reward(a, sigma, self.lam, self.alpha)
        return np.maximum(lr, 0.0) - self.arm_cost

    def _should_launch(self, a_hat: float, sigma_hat: float) -> bool:
        if self.penalize_frr:
            p_fl = float(norm.cdf(-a_hat / sigma_hat))
            p_fr = float(norm.cdf(a_hat / sigma_hat))
            return a_hat - self.lam * p_fl > -self.lam * p_fr
        return (
            float(lagrangian_launch_reward(a_hat, sigma_hat, self.lam, self.alpha)) > 0
        )

    def _stop_reward(self, a_grid: np.ndarray, sigma_k: float) -> np.ndarray:
        return np.asarray(self._reward(a_grid, sigma_k))

    def _stop_action(self, a_hat: float, sigma_hat: float) -> Action:
        if self._should_launch(a_hat, max(sigma_hat, 1e-12)):
            return "launch"
        return "reject"

    def _is_launch_boundary(self, a_bnd: float, sigma_k: float) -> bool:
        return self._should_launch(a_bnd, sigma_k)

    @property
    def label(self) -> str:
        suffix = " +FRR" if self.penalize_frr else ""
        return f"Lagrangian Bellman (lam={self.lam}{suffix})"


# ==========================================================================
# Bellman on z = a_hat / sigma coordinates (smooth boundary extraction)
# ==========================================================================
class AdaptiveBellmanCriterion(Criterion):
    """Bellman solver in z = a_hat/sigma coordinates for sigma-adaptive resolution.

    Physical grid spacing da(sigma) = sigma*dz is finest near sigma* where the
    continue wedge is narrowest, and boundaries are extracted by linear
    interpolation -- giving the smooth curves used in the decision-boundary
    figure. ``lam``/``penalize_frr``/``alpha`` mirror LagrangianBellmanCriterion;
    the defaults (lam=0) give the plain reward-optimal policy.
    """

    def __init__(
        self,
        a0: float,
        sigma0: float,
        delta: float,
        n_z: int = 301,
        z_half: float = 6.0,
        k_max: int = 100,
        n_iter: int = 500,
        lam: float = 0.0,
        penalize_frr: bool = True,
        alpha: float = 0.05,
        arm_cost: float = 0.0,
        tol: float = 1e-7,
    ) -> None:
        self.a0, self.sigma0, self.delta = a0, sigma0, delta
        self.n_z, self.k_max, self.n_iter = n_z, k_max, n_iter
        self.lam, self.penalize_frr = lam, penalize_frr
        self.arm_cost = arm_cost

        m0 = 1.0 / sigma0**2
        m_d = 1.0 / delta**2
        self.m0, self.m_d = m0, m_d

        prec = cast(np.ndarray, m0 + np.arange(k_max + 2) * m_d)
        sigma_all = cast(np.ndarray, 1.0 / np.sqrt(prec))
        self.sigma_arr = sigma_all[: k_max + 1]

        z_grid = np.linspace(-z_half, z_half, n_z)
        dz = z_grid[1] - z_grid[0]
        self.z_grid, self.dz = z_grid, dz

        # Stop reward in z-coords (a = sigma*z). The penalties depend on z only.
        if penalize_frr:
            self._launch_pen = lam * cast(np.ndarray, norm.cdf(-z_grid))
            self._reject_rew = -lam * cast(np.ndarray, norm.cdf(z_grid))
        else:  # penalize false launches only
            self._launch_pen = lam * (cast(np.ndarray, norm.cdf(-z_grid)) - alpha)
            self._reject_rew = np.zeros_like(z_grid)

        dvar = cast(np.ndarray, 1.0 / prec[:k_max] - 1.0 / prec[1 : k_max + 1])
        r = self.sigma_arr[:k_max] / sigma_all[1 : k_max + 1]
        sg = np.sqrt(dvar) / (sigma_all[1 : k_max + 1] * dz)
        self.min_sg = float(sg.min())

        k_stop = min(1, k_max)
        s_stop = self.sigma_arr[k_stop]
        w_stop = cast(
            np.ndarray,
            norm.pdf(z_grid, loc=a0 / s_stop, scale=np.sqrt(dvar[0]) / s_stop),
        )
        w_stop = w_stop / w_stop.sum()

        ref_idx = np.argmin(np.abs(z_grid - a0 / sigma0))

        W = np.zeros((k_max + 1, n_z))
        for _ in range(n_iter):
            Wn = np.zeros_like(W)
            e_stop = w_stop @ W[k_stop]
            for k in range(k_max + 1):
                stop_val = (
                    np.maximum(
                        self.sigma_arr[k] * z_grid - self._launch_pen, self._reject_rew
                    )
                    + e_stop
                    - arm_cost
                )
                if k < k_max:
                    smoothed = gaussian_filter1d(W[k + 1], sigma=sg[k], mode="nearest")
                    cont_val = np.interp(z_grid * r[k], z_grid, smoothed)
                    Wn[k] = np.maximum(stop_val, cont_val)
                else:
                    Wn[k] = stop_val
            Wn -= Wn[0, ref_idx]
            # Early stop at genuine convergence; boundary is identical to full n_iter.
            converged = bool(np.max(np.abs(Wn - W)) <= tol * (1.0 + np.max(np.abs(Wn))))
            W = Wn
            if converged:
                break

        self.W = W

        e_stop = float(w_stop @ W[k_stop])
        self._stop = np.zeros((k_max + 1, n_z), dtype=bool)
        self._diff = np.zeros((k_max + 1, n_z))
        for k in range(k_max + 1):
            sv = (
                np.maximum(
                    self.sigma_arr[k] * z_grid - self._launch_pen, self._reject_rew
                )
                + e_stop
                - arm_cost
            )
            if k < k_max:
                sm = gaussian_filter1d(W[k + 1], sigma=sg[k], mode="nearest")
                cv = np.interp(z_grid * r[k], z_grid, sm)
                self._diff[k] = sv - cv
                self._stop[k] = sv >= cv
            else:
                self._diff[k] = np.inf
                self._stop[k] = True

        # First k where the continue region vanishes (the wedge tip, sigma*).
        self.k_intersection: int | None = None
        self.sigma_star: float | None = None
        if not self._stop[0].all():
            lo, hi = 0, k_max
            while lo < hi:
                mid = (lo + hi) // 2
                if self._stop[mid].all():
                    hi = mid
                else:
                    lo = mid + 1
            if lo < k_max:
                self.k_intersection = lo
                self.sigma_star = float(self.sigma_arr[lo])

    def decide(self, a_hat: float, sigma_hat: float) -> Action:
        k = int(round((1.0 / max(sigma_hat, 1e-12) ** 2 - self.m0) / self.m_d))
        k = int(np.clip(k, 0, self.k_max))
        z = a_hat / max(sigma_hat, 1e-12)
        i = int(np.searchsorted(self.z_grid, z).clip(0, len(self.z_grid) - 1))
        if self._stop[k, i]:
            return "launch" if a_hat > 0 else "reject"
        return "continue"

    def extract_boundaries(
        self, sigma_min: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract boundaries with linear interpolation, sorted by sigma."""
        ra, rs, la, ls = [], [], [], []
        for k in range(self.k_max + 1):
            sk = self.sigma_arr[k]
            if sk > self.sigma0 or sk < sigma_min:
                continue
            for i in range(len(self.z_grid) - 1):
                if self._stop[k][i] != self._stop[k][i + 1]:
                    d0, d1 = self._diff[k][i], self._diff[k][i + 1]
                    denom = d0 - d1
                    frac = np.clip(d0 / denom, 0, 1) if abs(denom) > 1e-15 else 0.5
                    a_bnd = (self.z_grid[i] + frac * self.dz) * sk
                    if self._stop[k][i] and not self._stop[k][i + 1]:
                        ra.append(a_bnd)
                        rs.append(sk)
                    else:
                        la.append(a_bnd)
                        ls.append(sk)
        out = [np.array(x) for x in [ra, rs, la, ls]]
        order_r = np.argsort(out[1]) if len(out[1]) > 0 else np.array([], dtype=int)
        order_l = np.argsort(out[3]) if len(out[3]) > 0 else np.array([], dtype=int)
        return out[0][order_r], out[1][order_r], out[2][order_l], out[3][order_l]

    @property
    def label(self) -> str:
        return "Adaptive Bellman (optimal)"


# ==========================================================================
# Simulation
# ==========================================================================
@dataclass
class ArmTrace:
    """Records the exploration trajectory of a single version."""

    true_value: float
    a_hat_history: list[float] = field(default_factory=list)
    sigma_hat_history: list[float] = field(default_factory=list)
    outcome: str = "exploring"


@dataclass
class SimResult:
    """Collected results from one simulation run."""

    criterion_label: str
    V: float  # total lift shipped (sum of launched true values)
    v: float  # net reward rate = (V - cost * renewals) / total time
    n_launched: int
    n_rejected: int
    n_arms_seen: int
    v_over_time: np.ndarray
    arm_traces: list[ArmTrace]


def simulate(
    params: GaussSPParams,
    criterion: Criterion,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Run one long simulation of ``criterion`` over a stream of versions.

    Each version has true lift theta ~ N(a0, sigma0^2), observed via Gaussian
    updates until the policy launches (banking theta) or discards it, after
    which a fresh version begins. Returns the reward rate over time and the
    per-version outcome traces (used to compute FLR / type-I error).
    """
    if rng is None:
        rng = np.random.default_rng()

    a0, sigma0 = params.a0, params.sigma0
    delta2 = params.delta**2

    V = 0.0
    cost = params.cost
    t = params.tau
    n_launched = 0
    n_rejected = 0
    arm_idx = 0

    a_true = rng.normal(a0, sigma0)
    a_hat = a0
    sigma_hat2 = sigma0**2

    arm_traces = []
    current_trace = ArmTrace(
        true_value=a_true,
        a_hat_history=[a_hat],
        sigma_hat_history=[np.sqrt(sigma_hat2)],
    )
    v_over_time = np.zeros(params.n_rounds)

    for n in range(params.n_rounds):
        sigma_hat = np.sqrt(sigma_hat2)
        action = criterion.decide(a_hat, sigma_hat)

        if action == "launch":
            V += a_true
            current_trace.outcome = "launched"
            n_launched += 1
            arm_traces.append(current_trace)
            arm_idx += 1
            a_true = rng.normal(a0, sigma0)
            a_hat = a0
            sigma_hat2 = sigma0**2
            current_trace = ArmTrace(
                true_value=a_true,
                a_hat_history=[a_hat],
                sigma_hat_history=[np.sqrt(sigma_hat2)],
            )

        elif action == "reject":
            current_trace.outcome = "rejected"
            n_rejected += 1
            arm_traces.append(current_trace)
            arm_idx += 1
            a_true = rng.normal(a0, sigma0)
            a_hat = a0
            sigma_hat2 = sigma0**2
            current_trace = ArmTrace(
                true_value=a_true,
                a_hat_history=[a_hat],
                sigma_hat_history=[np.sqrt(sigma_hat2)],
            )

        b = rng.normal(a_true, params.delta)
        precision_prior = 1.0 / sigma_hat2
        precision_obs = 1.0 / delta2
        precision_post = precision_prior + precision_obs
        a_hat = (precision_prior * a_hat + precision_obs * b) / precision_post
        sigma_hat2 = 1.0 / precision_post

        current_trace.a_hat_history.append(a_hat)
        current_trace.sigma_hat_history.append(np.sqrt(sigma_hat2))

        t += params.tau
        stops = n_launched + n_rejected
        v_over_time[n] = (V - cost * stops) / t

    arm_traces.append(current_trace)

    stops = n_launched + n_rejected
    return SimResult(
        criterion_label=criterion.label,
        V=V,
        v=(V - cost * stops) / t,
        n_launched=n_launched,
        n_rejected=n_rejected,
        n_arms_seen=arm_idx + 1,
        v_over_time=v_over_time,
        arm_traces=arm_traces,
    )


# ==========================================================================
# Cached reward curves (JSONL) for the lift-over-time figures
# ==========================================================================
@dataclass
class RewardCurves:
    """Summary of n_sims simulation runs for one criterion.

    Holds the mean reward curve over rounds and the final-round mean/SE -- not
    the full per-sim arrays (too large to cache).
    """

    mean_curve: np.ndarray  # mean of v_over_time across sims, length n_rounds
    final_mean: float  # mean final-round reward across sims
    final_se: float  # standard error of the final-round reward
    n_sims: int
    std_curve: np.ndarray | None = None  # std of v_over_time across sims


def _criterion_cache_config(criterion: Criterion) -> dict[str, object]:
    """JSON-serializable scalar config identifying a criterion for caching.

    Captures the class name plus every scalar attribute. Non-scalar attributes
    (numpy grids, value tables) are deterministic functions of those scalars,
    so dropping them is safe.
    """
    cfg: dict[str, object] = {"class": type(criterion).__name__}
    for k, v in sorted(vars(criterion).items()):
        if isinstance(v, (bool, int, float, str)) or v is None:
            cfg[k] = v
    return cfg


def cached_reward_curves(
    params: GaussSPParams,
    criteria: dict[str, Criterion],
    n_sims: int,
    seed: int = 42,
    cache_path: str | None = None,
) -> dict[str, RewardCurves]:
    """Compute (or load) mean reward curves + final stats per criterion.

    For each (params, criterion, n_sims, seed) this runs n_sims simulations with
    seeds seed+i and stores the mean reward curve and final-round mean/SE to an
    append-only JSONL cache. Subsequent calls with the same parameters load from
    the cache. Returns {name: RewardCurves} in the order of ``criteria``.
    """
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "reward_curves.jsonl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    base_key = {
        "a0": params.a0,
        "sigma0": params.sigma0,
        "delta": params.delta,
        "tau": params.tau,
        "n_rounds": params.n_rounds,
        "cost": params.cost,
        "n_sims": n_sims,
        "seed": seed,
    }

    cache: dict[str, dict[str, object]] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["key_hash"]] = entry

    out: dict[str, RewardCurves] = {}
    new_entries: list[dict[str, object]] = []
    for name, crit in criteria.items():
        key = {**base_key, "criterion": _criterion_cache_config(crit)}
        key_hash = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

        entry = cache.get(key_hash)
        if entry is None:
            v_curves = np.empty((n_sims, params.n_rounds))
            for i in range(n_sims):
                rng = np.random.default_rng(seed + i)
                v_curves[i] = simulate(params, crit, rng=rng).v_over_time
            final = v_curves[:, -1]
            entry = {
                "key_hash": key_hash,
                "name": name,
                **key,
                "final_mean": float(final.mean()),
                "final_se": float(final.std() / np.sqrt(n_sims)),
                "mean_curve": v_curves.mean(axis=0).tolist(),
                "std_curve": v_curves.std(axis=0).tolist(),
            }
            cache[key_hash] = entry
            new_entries.append(entry)

        std_raw = entry.get("std_curve")
        out[name] = RewardCurves(
            mean_curve=np.asarray(entry["mean_curve"], dtype=float),
            final_mean=float(cast(float, entry["final_mean"])),
            final_se=float(cast(float, entry["final_se"])),
            n_sims=int(cast(int, entry["n_sims"])),
            std_curve=np.asarray(std_raw, dtype=float) if std_raw is not None else None,
        )

    if new_entries:
        with open(cache_path, "a") as f:
            for entry in new_entries:
                f.write(json.dumps(entry) + "\n")

    return out
