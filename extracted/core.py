"""
pyofn.core
==========
Ordered Fuzzy Numbers (OFN) — rdzeń biblioteki.

Reprezentacja:
    OFN = (x_up, x_down) gdzie obie funkcje: [0,1] -> R
    - x_up  : ramię wznoszące  (f: y -> x)
    - x_down: ramię opadające  (f: y -> x)
    - kierunek: x_up(0) < x_up(1) oznacza kierunek "w prawo" (+)
               x_up(0) > x_up(1) oznacza kierunek "w lewo"  (-)

Dyskretyzacja: obie funkcje przechowywane jako wektory numpy
długości N próbkowane równomiernie dla y ∈ [0, 1].
Wszystkie operacje arytmetyczne są wektoryzowane.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Union


_DEFAULT_N = 512  # liczba punktów dyskretyzacji — kompromis szybkość/dokładność


class OFN:
    """
    Skierowana Liczba Rozmyta (Ordered Fuzzy Number).

    Parameters
    ----------
    x_up : array-like lub callable
        Ramię wznoszące. Jeśli callable: f(y) -> float/array dla y ∈ [0,1].
    x_down : array-like lub callable
        Ramię opadające. Jeśli callable: f(y) -> float/array dla y ∈ [0,1].
    n : int
        Liczba punktów dyskretyzacji (domyślnie 512).
    """

    __slots__ = ("_up", "_dn", "_y")

    def __init__(
        self,
        x_up: Union[np.ndarray, Callable],
        x_down: Union[np.ndarray, Callable],
        n: int = _DEFAULT_N,
    ):
        self._y = np.linspace(0.0, 1.0, n)

        if callable(x_up):
            self._up = np.asarray(x_up(self._y), dtype=np.float64)
        else:
            up = np.asarray(x_up, dtype=np.float64)
            if up.shape[0] != n:
                self._up = np.interp(self._y, np.linspace(0, 1, len(up)), up)
            else:
                self._up = up.copy()

        if callable(x_down):
            self._dn = np.asarray(x_down(self._y), dtype=np.float64)
        else:
            dn = np.asarray(x_down, dtype=np.float64)
            if dn.shape[0] != n:
                self._dn = np.interp(self._y, np.linspace(0, 1, len(dn)), dn)
            else:
                self._dn = dn.copy()

    # ------------------------------------------------------------------
    # Właściwości geometryczne
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return len(self._y)

    @property
    def up(self) -> np.ndarray:
        """Ramię wznoszące (wartości x dla y ∈ [0,1])."""
        return self._up

    @property
    def down(self) -> np.ndarray:
        """Ramię opadające (wartości x dla y ∈ [0,1])."""
        return self._dn

    @property
    def y(self) -> np.ndarray:
        """Oś y — wspólna dla obu ramion."""
        return self._y

    @property
    def direction(self) -> int:
        """
        +1 jeśli OFN skierowana w prawo (x_up rośnie),
        -1 jeśli w lewo,
         0 jeśli singleton.
        """
        delta = float(self._up[-1] - self._up[0])
        if delta > 1e-12:
            return 1
        elif delta < -1e-12:
            return -1
        return 0

    @property
    def core(self) -> tuple[float, float]:
        """
        Jądro OFN: (x_up(1), x_down(1)) — punkt/zakres z przynależnością = 1.
        Dla trójkąta/trapeze oba punkty mogą być równe lub tworzyć przedział.
        """
        return float(self._up[-1]), float(self._dn[-1])

    @property
    def support(self) -> tuple[float, float]:
        """Nośnik (support): przedział [min, max] wartości x z μ > 0."""
        all_x = np.concatenate([self._up, self._dn])
        return float(np.min(all_x)), float(np.max(all_x))

    def membership(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """
        Oblicz funkcję przynależności μ(x) na podstawie obu ramion.
        Metoda: odwrotna interpolacja — dla każdego x szukamy y.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        result = np.zeros_like(x)

        # Ramię wznoszące — monotoniczne, można odwrócić przez interp
        # sortujemy by x_up było rosnące dla interp
        idx_up = np.argsort(self._up)
        mu_up = np.interp(x, self._up[idx_up], self._y[idx_up], left=0.0, right=0.0)

        idx_dn = np.argsort(self._dn)
        mu_dn = np.interp(x, self._dn[idx_dn], self._y[idx_dn], left=0.0, right=0.0)

        result = np.maximum(mu_up, mu_dn)
        return result

    # ------------------------------------------------------------------
    # Operacje arytmetyczne — OFN ○ OFN
    # ------------------------------------------------------------------

    def _check_compat(self, other: OFN) -> None:
        if not isinstance(other, OFN):
            raise TypeError(f"Oczekiwano OFN, otrzymano {type(other)}")
        if self.n != other.n:
            raise ValueError(
                f"Niezgodna dyskretyzacja: {self.n} vs {other.n}. "
                "Użyj OFN.resample() lub stwórz z tym samym n."
            )

    def __add__(self, other: Union[OFN, float, int]) -> OFN:
        if isinstance(other, (int, float)):
            # Przesunięcie o skalar
            return OFN(self._up + other, self._dn + other, self.n)
        self._check_compat(other)
        return OFN(self._up + other._up, self._dn + other._dn, self.n)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union[OFN, float, int]) -> OFN:
        if isinstance(other, (int, float)):
            return OFN(self._up - other, self._dn - other, self.n)
        self._check_compat(other)
        # Odejmowanie = dodanie odwróconej (negacja kierunku)
        return OFN(self._up - other._dn, self._dn - other._up, self.n)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other: Union[OFN, float, int]) -> OFN:
        if isinstance(other, (int, float)):
            if other >= 0:
                return OFN(self._up * other, self._dn * other, self.n)
            else:
                # Ujemny skalar odwraca kierunek
                return OFN(self._dn * other, self._up * other, self.n)
        self._check_compat(other)
        return OFN(self._up * other._up, self._dn * other._dn, self.n)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union[OFN, float, int]) -> OFN:
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError
            return self.__mul__(1.0 / other)
        self._check_compat(other)
        # Dzielenie — ramiona parami; uwaga na zera!
        with np.errstate(divide="raise", invalid="raise"):
            try:
                up = self._up / other._up
                dn = self._dn / other._dn
            except FloatingPointError:
                raise ZeroDivisionError("Mianownik OFN zawiera zero w ramionach.")
        return OFN(up, dn, self.n)

    def __neg__(self) -> OFN:
        """Negacja — odwraca kierunek."""
        return OFN(-self._dn, -self._up, self.n)

    def __abs__(self) -> OFN:
        return OFN(np.abs(self._up), np.abs(self._dn), self.n)

    def __repr__(self) -> str:
        c = self.core
        s = self.support
        d = {1: "→", -1: "←", 0: "•"}[self.direction]
        return (
            f"OFN(core=({c[0]:.3g}, {c[1]:.3g}), "
            f"support=[{s[0]:.3g}, {s[1]:.3g}], "
            f"dir={d}, n={self.n})"
        )

    # ------------------------------------------------------------------
    # Narzędzia
    # ------------------------------------------------------------------

    def resample(self, n: int) -> OFN:
        """Zmień rozdzielczość dyskretyzacji."""
        new_y = np.linspace(0, 1, n)
        up = np.interp(new_y, self._y, self._up)
        dn = np.interp(new_y, self._y, self._dn)
        return OFN(up, dn, n)

    def reverse(self) -> OFN:
        """Odwróć skierowanie (zamień ramiona)."""
        return OFN(self._dn.copy(), self._up.copy(), self.n)

    def defuzzify_cog(self) -> float:
        """
        Wyostrzanie metodą środka ciężkości (Center of Gravity).
        Całkuje po całym kształcie funkcji przynależności.
        """
        # Zbieramy próbki x z obu ramion i ich przynależności y
        xs = np.concatenate([self._up, self._dn])
        ys = np.concatenate([self._y, self._y])
        # Sortujemy po x
        idx = np.argsort(xs)
        xs_s, ys_s = xs[idx], ys[idx]
        num = np.trapezoid(ys_s * xs_s, xs_s)
        den = np.trapezoid(ys_s, xs_s)
        if abs(den) < 1e-15:
            return float(np.mean(xs))
        return float(num / den)

    def defuzzify_mean_core(self) -> float:
        """Wyostrzanie: średnia wartości jądra."""
        return float(np.mean(self.core))

    def distance_hamming(self, other: OFN) -> float:
        """Odległość Hamminga między dwoma OFN (całkowanie po y)."""
        self._check_compat(other)
        d_up = np.trapezoid(np.abs(self._up - other._up), self._y)
        d_dn = np.trapezoid(np.abs(self._dn - other._dn), self._y)
        return float(0.5 * (d_up + d_dn))

    def to_dict(self) -> dict:
        """Serializacja do słownika (np. do JSON)."""
        return {
            "up": self._up.tolist(),
            "down": self._dn.tolist(),
            "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OFN:
        """Deserializacja ze słownika."""
        return cls(np.array(d["up"]), np.array(d["down"]), d["n"])
