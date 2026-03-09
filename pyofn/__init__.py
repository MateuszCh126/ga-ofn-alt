"""
pyofn — Ordered Fuzzy Numbers (Skierowane Liczby Rozmyte)
=========================================================

Biblioteka implementuje arytmetykę OFN wg koncepcji Kosińskiego.

Szybki start
------------
>>> from pyofn import OFN, triangular, about, plot
>>> A = triangular(1, 3, 5)        # OFN trójkątna, kierunek →
>>> B = about(4, spread=1.5)       # "około 4"
>>> C = A + B
>>> print(C)
>>> plot(C, title="A + B")

Moduły
------
core    – klasa OFN (arytmetyka, defuzzyfikacja, dystanse)
shapes  – konstruktory kształtów (triangular, trapezoidal, gaussian, …)
viz     – wizualizacja (matplotlib)
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
__author__ = "pyofn contributors"

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
