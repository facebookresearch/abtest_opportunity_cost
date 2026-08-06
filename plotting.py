"""Shared plotting style and colors for the paper figures.

Every figure notebook calls ``apply_paper_style()`` first, then uses the color
and size constants below so the figures are visually consistent.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Directory the notebooks write final figures to (../figures relative to code/).
FIGURE_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
)

PAPER_STYLE: dict[str, object] = {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 22,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}

# Consistent colors across figures.
COLOR_ORACLE: str = "navy"  # Bellman / oracle / reference curves
COLOR_Z: str = "tab:olive"  # z-criterion
COLOR_PPOS: str = "seagreen"  # PPOS
COLOR_CONSTRAINED: str = "crimson"  # e-process / constrained Bellman
COLOR_LAGRANGIAN: str = "royalblue"  # Lagrangian Bellman
COLOR_COST: str = "tab:red"  # cost-aware / misspecified overlays

# Line/marker sizing conventions.
LW_MAIN: float = 3.5  # oracle / primary curves
LW: float = 3.0  # comparison curves
MS: int = 8  # marker size
CAPSIZE: int = 4  # error-bar caps
TICK_SIZE: int = 18
LABEL_SIZE: int = 22
LEGEND_SIZE: int = 16

Z_CRIT: float = 1.96  # two-sided 0.025 z-threshold used throughout


def apply_paper_style() -> None:
    """Set the global matplotlib rcParams to the paper style."""
    plt.rcParams.update(PAPER_STYLE)


def save_figure(fig: Figure, name: str, out_dir: str | None = None) -> str:
    """Save ``fig`` to the paper figures directory as a tight, 300-dpi file.

    ``name`` should include the extension (``.pdf`` for vector figures,
    ``.png`` for the ablation panels). Returns the path written.
    """
    out_dir = FIGURE_DIR if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Saved {path}")
    return path
