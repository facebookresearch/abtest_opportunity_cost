Code for the paper _A/B Testing with Opportunity Costs: Trading Off Reward and Error Control_. 

# Reproducing the figures

Self-contained code for every figure and table in the paper. It depends only on
`numpy`, `scipy`, and `matplotlib`.

## Layout

```
code/
  methods.py      all algorithms: decision rules, the Bellman solvers, and the
                  simulation harness
  plotting.py     shared matplotlib style and colors
  requirements.txt
  cache/          created on first run; memoized Bellman solves and simulation
                  results so reruns are instant. Safe to delete to recompute.
  <one notebook per figure group>
```

Each notebook is a plain-text notebook using `# %%` cell markers. Open it
directly in Jupyter or VS Code, or convert to `.ipynb` with
`jupytext --to notebook fig_*.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run any notebook top to bottom; it writes its figure(s) to `../figures/` and
prints any accompanying LaTeX table rows. Everything is recomputed from scratch
on the first run and cached under `cache/` afterwards — the first run of the
simulation-heavy notebooks takes a few minutes; later runs are instant.

## Figure and table map

| Notebook | Figures | Tables |
|----------|---------|--------|
| `fig_baselines.py`     | `baselines_vs_bellman.pdf`, `reward_distribution.pdf` | `tab:baseline-errors` |
| `fig_boundaries.py`    | `boundaries.pdf` | — |
| `fig_flr_control.py`   | `flr_control_c0.05.pdf` | `tab:lagrangian` |
| `fig_type1_control.py` | — | `tab:flr` |
| `fig_unknown_prior.py` | `unknown_prior_mild.pdf`, `unknown_prior_severe.pdf` | — |
| `fig_cost_ablation.py` | `cost_ablation.pdf` | `tab:cost-ablation` |
| `fig_ablations.py`     | `ablation_ppos.png`, `ablation_bayes_vs_freq.png` | `tab:bayes-vs-freq` |
| `fig_robustness.py`    | `bellman_delta_robustness.pdf`, `kmax_sensitivity.pdf` | `tab:kmax-ablation` |

## The methods

`methods.py` implements:

- **Baselines** — `ZCriterion` / `FreqZCriterion` (posterior / sample z-score),
  `PPoSCriterion` / `FreqPPoSCriterion` (predictive probability of success).
- **Bellman** (`BellmanCriterion`) — the reward-optimal policy via value
  iteration; the oracle the baselines are measured against.
- **Error control** — `ConstrainedBellmanCriterion` (type-I control via an
  e-process launch gate) and `LagrangianBellmanCriterion` (false-launch-rate
  control via a Lagrangian penalty).
- **`AdaptiveBellmanCriterion`** — the same optimal policy solved in
  `z = a/sigma` coordinates, used only to extract smooth decision boundaries for
  `fig_boundaries.py`.
- **`simulate`** — runs a policy over a long stream of versions and returns the
  reward rate over time plus per-version outcomes (for false-rate estimates).

Build Bellman-family policies through `cached_criterion(Cls, ...)` (never the
class directly) so the value-iteration solve is memoized.

## License

abtest_opportunity_cost is MIT licensed, as found in the [LICENSE](LICENSE)
file.
