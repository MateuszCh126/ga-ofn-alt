"""
pyofn - Ordered Fuzzy Numbers (skierowane liczby rozmyte).

Biblioteka implementuje arytmetykę OFN wg koncepcji Kosińskiego.

Moduły:
    core    - klasa OFN (arytmetyka, defuzzyfikacja, odległości)
    shapes  - konstruktory kształtów (triangular, trapezoidal, gaussian, ...)
    viz     - wizualizacja (matplotlib)
"""

from .core import OFN
from .shapes import (
    triangular,
    triangular_left,
    trapezoidal,
    gaussian,
    singleton,
    linear_ofn,
    about,
)
from .viz import plot, plot_many, plot_arithmetic, plot_direction_demo

__version__ = "0.1.0"

__all__ = [
    "OFN",
    # shapes
    "triangular",
    "triangular_left",
    "trapezoidal",
    "gaussian",
    "singleton",
    "linear_ofn",
    "about",
    # viz
    "plot",
    "plot_many",
    "plot_arithmetic",
    "plot_direction_demo",
]
