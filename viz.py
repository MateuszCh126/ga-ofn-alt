"""
pyofn.viz
=========
Wizualizacja OFN przy użyciu matplotlib.

Funkcje:
    plot(ofn, ...)            – jeden OFN
    plot_many(ofns, ...)      – wiele OFN na jednym wykresie
    plot_arithmetic(a, b, op) – pokazuje A, B i wynik operacji
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .core import OFN

_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706",
    "#7c3aed", "#0891b2", "#be185d", "#374151",
]


def _require_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib jest wymagany do wizualizacji: pip install matplotlib")


def plot(
    ofn: OFN,
    label: str = "OFN",
    color: str = "#2563eb",
    ax=None,
    show_membership: bool = True,
    show_direction: bool = True,
    title: Optional[str] = None,
    figsize: tuple = (7, 4),
) -> "plt.Axes":
    """
    Rysuje pojedynczą OFN jako wykres funkcji przynależności
    z zaznaczonym kierunkiem ramion.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    _require_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f8fafc")

    y = ofn.y
    xu = ofn.up
    xd = ofn.down

    # --- funkcja przynależności jako wypełnienie ---
    # zbieramy kontur: ramię wznoszące (y: 0→1) + opadające (y: 1→0)
    x_contour = np.concatenate([xu, xd[::-1]])
    y_contour = np.concatenate([y, y[::-1]])

    ax.fill(x_contour, y_contour, alpha=0.15, color=color)
    ax.plot(x_contour, y_contour, color=color, linewidth=1.5, alpha=0.4, linestyle="--")

    # --- ramię wznoszące ---
    ax.plot(xu, y, color=color, linewidth=2.5, label=f"{label} ↑up")

    # --- ramię opadające (inny styl) ---
    ax.plot(xd, y, color=color, linewidth=2.5, linestyle=":", label=f"{label} ↓down")

    if show_direction and ofn.direction != 0:
        arrow_x = (xu[-1] + xd[-1]) / 2
        dx = 0.3 * ofn.direction * (max(xu.max(), xd.max()) - min(xu.min(), xd.min()))
        ax.annotate(
            "",
            xy=(arrow_x + dx, 1.02),
            xytext=(arrow_x, 1.02),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("μ(x)", fontsize=11)
    ax.set_xlabel("x", fontsize=11)
    ax.set_title(title or label, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_many(
    ofns: Sequence[OFN],
    labels: Optional[Sequence[str]] = None,
    title: str = "Zbiory OFN",
    figsize: tuple = (9, 5),
    ax=None,
) -> "plt.Axes":
    """
    Rysuje wiele OFN na jednym wykresie.
    """
    _require_mpl()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f8fafc")

    if labels is None:
        labels = [f"OFN {i+1}" for i in range(len(ofns))]

    patches = []
    for i, (ofn, lbl) in enumerate(zip(ofns, labels)):
        c = _COLORS[i % len(_COLORS)]
        y = ofn.y
        xu, xd = ofn.up, ofn.down

        x_contour = np.concatenate([xu, xd[::-1]])
        y_contour = np.concatenate([y, y[::-1]])

        ax.fill(x_contour, y_contour, alpha=0.12, color=c)
        ax.plot(xu, y, color=c, linewidth=2.2)
        ax.plot(xd, y, color=c, linewidth=2.2, linestyle=":")

        patches.append(mpatches.Patch(color=c, label=lbl))

    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("μ(x)", fontsize=11)
    ax.set_xlabel("x", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(handles=patches, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    return ax


def plot_arithmetic(
    a: OFN,
    b: OFN,
    result: OFN,
    op_symbol: str = "+",
    label_a: str = "A",
    label_b: str = "B",
    figsize: tuple = (14, 4),
) -> "plt.Figure":
    """
    Trójpanelowy wykres: A | B | A ○ B
    Przydatny do demonstracji operacji arytmetycznych.
    """
    _require_mpl()

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="#f8fafc")
    fig.suptitle(f"OFN: {label_a}  {op_symbol}  {label_b}", fontsize=14, fontweight="bold")

    plot(a, label=label_a, color=_COLORS[0], ax=axes[0], title=label_a)
    plot(b, label=label_b, color=_COLORS[1], ax=axes[1], title=label_b)
    plot(
        result,
        label=f"{label_a}{op_symbol}{label_b}",
        color=_COLORS[2],
        ax=axes[2],
        title=f"Wynik: {label_a} {op_symbol} {label_b}",
    )

    plt.tight_layout()
    return fig


def plot_direction_demo(
    value: float = 5.0,
    spread: float = 2.0,
    n: int = 512,
    figsize: tuple = (10, 4),
) -> "plt.Figure":
    """
    Demonstruje różnicę między OFN → i ← dla tego samego kształtu.
    """
    _require_mpl()
    from .shapes import triangular, triangular_left

    a = triangular(value - spread, value, value + spread, n)
    b = triangular_left(value - spread, value, value + spread, n)

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="#f8fafc")
    fig.suptitle("Kierunek OFN: ta sama forma, różne skierowanie", fontsize=13)

    plot(a, label="OFN →", color=_COLORS[0], ax=axes[0], title="Skierowanie: →")
    plot(b, label="OFN ←", color=_COLORS[1], ax=axes[1], title="Skierowanie: ←")

    plt.tight_layout()
    return fig
