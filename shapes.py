"""
pyofn.shapes
============
Konstruktory typowych kształtów OFN.

Każda funkcja zwraca OFN z sensownym kierunkiem domyślnym.
"""

from __future__ import annotations

import numpy as np
from .core import OFN


def triangular(a: float, b: float, c: float, n: int = 512) -> OFN:
    """
    Trójkątny OFN skierowany w prawo: wierzchołek w b.

        a ≤ b ≤ c
        x_up  : ramię wznoszące  a → b   (y: 0→1)
        x_down: ramię opadające  b → c   (y: 1→0, ale param. przez y: 0→1)

    kierunek: → (prawo)
    """
    if not (a <= b <= c):
        raise ValueError(f"Wymagane a≤b≤c, podano a={a}, b={b}, c={c}")
    y = np.linspace(0.0, 1.0, n)
    x_up = a + y * (b - a)
    x_down = b + (1.0 - y) * (c - b)
    return OFN(x_up, x_down, n)


def triangular_left(a: float, b: float, c: float, n: int = 512) -> OFN:
    """
    Trójkątny OFN skierowany w lewo (←): wierzchołek w b.

    W OFN kierunek ← oznacza, że x_up jest MALEJĄCE (od c do b),
    a x_down kontynuuje malejąco (od b do a).
    Interpretacja: OFN "biegnie" z prawej strony w lewo.

        x_up  : c → b   (y: 0→1, wartości x maleją)
        x_down: b → a   (y: 0→1, wartości x maleją)
    """
    if not (a <= b <= c):
        raise ValueError(f"Wymagane a≤b≤c, podano a={a}, b={b}, c={c}")
    y = np.linspace(0.0, 1.0, n)
    x_up   = c - y * (c - b)   # malejące: c..b  → direction = -1 (←)
    x_down = b - y * (b - a)   # malejące: b..a
    return OFN(x_up, x_down, n)


def trapezoidal(a: float, b: float, c: float, d: float, n: int = 512) -> OFN:
    """
    Trapezowy OFN skierowany w prawo: plateau b..c.

        a ≤ b ≤ c ≤ d
        x_up  : a → b   (y: 0→1)
        x_down: c → d   (y: 1→0, przez y: 0→1 mapujemy c..d)

    kierunek: →
    """
    if not (a <= b <= c <= d):
        raise ValueError(f"Wymagane a≤b≤c≤d, podano {a},{b},{c},{d}")
    y = np.linspace(0.0, 1.0, n)
    x_up = a + y * (b - a)
    x_down = c + (1.0 - y) * (d - c)
    return OFN(x_up, x_down, n)


def gaussian(mean: float, sigma: float, n: int = 512, support_sigma: float = 3.0) -> OFN:
    """
    Gaussowski OFN (symetryczny, skierowany w prawo).

    Ramiona parametryzowane odwrotną funkcją Gaussa:
        x_up  : mean - sigma*sqrt(-2*ln(y))  dla lewej strony
        x_down: mean + sigma*sqrt(-2*ln(y))  dla prawej strony

    Przy y→0 wartości dążą do ±∞, więc obcinamy do support_sigma*sigma.
    """
    if sigma <= 0:
        raise ValueError("sigma musi być > 0")
    y = np.linspace(0.0, 1.0, n)
    # Unikamy log(0) przy y=0
    y_safe = np.clip(y, 1e-10, 1.0)
    half_width = sigma * np.sqrt(-2.0 * np.log(y_safe))
    half_width = np.clip(half_width, 0, support_sigma * sigma)
    x_up = mean - half_width          # lewa strona: maleje dla y→0
    x_down = mean + half_width[::-1]  # prawa strona: rośnie dla y→0, rev = opadające
    return OFN(x_up, x_down, n)


def singleton(value: float, n: int = 512) -> OFN:
    """
    Singleton (crisp number jako OFN) — oba ramiona stałe = value.
    Kierunek: 0 (neutralny).
    """
    ones = np.full(n, float(value))
    return OFN(ones.copy(), ones.copy(), n)


def linear_ofn(
    x_up_start: float,
    x_up_end: float,
    x_down_start: float,
    x_down_end: float,
    n: int = 512,
) -> OFN:
    """
    Ogólna liniowa OFN — oba ramiona liniowe (trapezoid / trójkąt /
    dowolna kombinacja z ostrymi ramionami).

    Parametry (wszystkie dla y ∈ [0..1]):
        x_up_start   = x_up(y=0)
        x_up_end     = x_up(y=1)
        x_down_start = x_down(y=0)
        x_down_end   = x_down(y=1)
    """
    y = np.linspace(0.0, 1.0, n)
    x_up = x_up_start + y * (x_up_end - x_up_start)
    x_dn = x_down_start + y * (x_down_end - x_down_start)
    return OFN(x_up, x_dn, n)


def about(value: float, spread: float = 1.0, n: int = 512) -> OFN:
    """
    Symetryczny trójkąt "około value" ± spread.
    Skrót do triangular(value-spread, value, value+spread).
    """
    return triangular(value - spread, value, value + spread, n)
