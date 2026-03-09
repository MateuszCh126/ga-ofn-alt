"""
ga_core.py
==========
Algorytm genetyczny dla Trapezoidalnych Skierowanych Liczb Rozmytych (OFN).

Reprezentacja genu:
  Każdy gen = trapezoid OFN → (a, b, c, d)  gdzie a ≤ b ≤ c ≤ d
  Chromosom  = tablica (n_genes, 4)
  Populacja  = tablica (pop_size, n_genes, 4)

Integracja z pyofn:
  - Target tworzony przez pyofn.trapezoidal(a,b,c,d)
  - Parametry targetu wyciągane z ramion pyofn.OFN
  - Najlepszy osobnik eksportowany do listy pyofn.OFN do rysowania

Fitness:
  Analityczna odległość Hamminga trapezoidalnych OFN — wektoryzowana na
  całej populacji jedną operacją numpy (zero pętli Python po osobnikach).

  Dla trapez OFN (a,b,c,d):
    x_up(y)   = a + y*(b-a)       — ramię wznoszące,  liniowe w y
    x_down(y) = d - y*(d-c)       — ramię opadające,  liniowe w y
  (zgodnie z pyofn.shapes.trapezoidal)

  Różnica między osobnikiem i targetem:
    Δ_up(y)   = (aP-aT) + y*[(bP-aP)-(bT-aT)]  = β_up + y*α_up
    Δ_down(y) = (dP-dT) - y*[(dP-cP)-(dT-cT)]  = β_dn + y*α_dn
  Oba liniowe → ∫₀¹|αy+β|dy analitycznie (pole trapezu/trójkąta).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List

# Importujemy pyofn — używamy go do tworzenia OFN i eksportu wyników
from pyofn import trapezoidal, OFN


# ---------------------------------------------------------------------------
# Matematyka: ∫₀¹ |αy + β| dy — analitycznie
# ---------------------------------------------------------------------------

def _integral_abs_linear(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Analityczna całka ∫₀¹ |αy + β| dy dla tablic α, β (dowolny kształt).

    Geometria: funkcja αy+β jest liniowa → wykres to trapez lub trójkąt.
    Wartości na krańcach przedziału:
        v0 = β        (przy y=0)
        v1 = α + β    (przy y=1)

    Przypadki:
      1) v0, v1 tego samego znaku → pole trapezu = (|v0|+|v1|)/2
      2) Zmiana znaku (y₀ ∈ (0,1)) → dwa trójkąty, y₀ = v0/(v0-v1)
    """
    v0 = beta
    v1 = alpha + beta

    same_sign = (v0 * v1) >= 0
    # przypadek 1: trapez
    no_cross = 0.5 * np.abs(v0 + v1)

    # przypadek 2: dwa trójkąty
    eps = 1e-14
    safe = np.where(np.abs(v0 - v1) < eps, eps, v0 - v1)
    y0   = np.clip(v0 / safe, 0.0, 1.0)
    cross = 0.5 * (np.abs(v0) * y0 + np.abs(v1) * (1.0 - y0))

    return np.where(same_sign, no_cross, cross)


# ---------------------------------------------------------------------------
# Fitness: wektoryzowana analityczna odległość Hamminga (trapez OFN)
# ---------------------------------------------------------------------------

def fitness_batch(population: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Oblicza fitness całej populacji jedną operacją numpy.

    Parameters
    ----------
    population : (P, G, 4)  — P osobników, G genów, params [a,b,c,d]
    target     : (G, 4)     — cel: parametry trapezu dla każdego genu

    Returns
    -------
    fitness : (P,)  — średnia odległość Hamminga po genach (↓ = lepiej)
    """
    t  = target[np.newaxis]                              # (1, G, 4)
    aP, bP, cP, dP = (population[..., i] for i in range(4))
    aT, bT, cT, dT = (t[..., i] for i in range(4))

    # Ramię wznoszące: x_up(y) = a + y*(b-a)
    beta_up  = aP - aT                                   # (P, G)
    alpha_up = (bP - aP) - (bT - aT)                    # (P, G)

    # Ramię opadające: x_down(y) = d - y*(d-c)
    beta_dn  = dP - dT                                   # (P, G)
    alpha_dn = -(dP - cP) + (dT - cT)                   # (P, G)

    d_up = _integral_abs_linear(alpha_up, beta_up)       # (P, G)
    d_dn = _integral_abs_linear(alpha_dn, beta_dn)       # (P, G)

    return 0.5 * (d_up + d_dn).mean(axis=1)             # (P,)


# ---------------------------------------------------------------------------
# Konwersja: pyofn.OFN ↔ parametry trapezu (a,b,c,d)
# ---------------------------------------------------------------------------

def ofn_to_params(ofn: OFN) -> np.ndarray:
    """
    Wyciąga parametry trapezu (a,b,c,d) z obiektu pyofn.OFN.
    Zakłada liniowe ramiona (trapezoidal / triangular).

      a = x_up(y=0),  b = x_up(y=1)
      c = x_down(y=1), d = x_down(y=0)
    """
    a = float(ofn.up[0])
    b = float(ofn.up[-1])
    c = float(ofn.down[-1])
    d = float(ofn.down[0])
    return np.array(sorted([a, b, c, d]), dtype=np.float64)


def params_to_ofn(params: np.ndarray, n: int = 512) -> OFN:
    """
    Tworzy pyofn.OFN (trapezoidal) z tablicy (4,) = [a,b,c,d].
    Korzysta z pyofn.trapezoidal — oficjalna fabryka biblioteki.
    """
    a, b, c, d = params
    return trapezoidal(float(a), float(b), float(c), float(d), n=n)


def individual_to_ofns(individual: np.ndarray, n: int = 256) -> List[OFN]:
    """
    Konwertuje chromosom (n_genes, 4) → listę n_genes obiektów pyofn.OFN.
    Używane do wizualizacji wyników w app.py przez pyofn.viz.
    """
    return [params_to_ofn(row, n) for row in individual]


# ---------------------------------------------------------------------------
# Operatory genetyczne — wektoryzowane
# ---------------------------------------------------------------------------

def tournament_select(fitness: np.ndarray, n_select: int,
                      k: int, rng: np.random.Generator) -> np.ndarray:
    """Wektoryzowana selekcja turniejowa. Zwraca indeksy wygranych (n_select,)."""
    P = len(fitness)
    cand = rng.integers(0, P, size=(n_select, k))        # (n_select, k)
    fit_c = fitness[cand]                                  # (n_select, k)
    return cand[np.arange(n_select), fit_c.argmin(axis=1)]


def crossover_uniform(population: np.ndarray, rate: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Krzyżowanie jednorodne na poziomie genów (cały trójkąt/trapez jako jednostka)."""
    P, G, D = population.shape
    offspring = population.copy()
    if P < 2: return offspring

    idx  = rng.permutation(P)
    half = P // 2
    p1, p2 = idx[:half], idx[half:2*half]

    do_cross = (rng.random(half) < rate)[:, None, None]   # (half,1,1)
    gene_mask = (rng.random((half, G)) < 0.5)[:, :, None] # (half,G,1)
    mask = gene_mask & do_cross

    g1, g2 = population[p1], population[p2]
    offspring[p1] = np.where(mask, g2, g1)
    offspring[p2] = np.where(mask, g1, g2)
    return offspring


def mutate_gaussian(population: np.ndarray, rate: float, sigma: float,
                    bounds: tuple[float, float],
                    rng: np.random.Generator) -> np.ndarray:
    """
    Gaussowska mutacja. Sortowanie wzdłuż osi parametrów gwarantuje a≤b≤c≤d.
    """
    P, G, D = population.shape
    mask    = rng.random((P, G, D)) < rate
    noise   = rng.normal(0.0, sigma, (P, G, D))
    mutated = np.where(mask, population + noise, population)
    mutated = np.clip(mutated, bounds[0], bounds[1])
    mutated.sort(axis=-1)   # wymusza a ≤ b ≤ c ≤ d — bez alokacji
    return mutated


# ---------------------------------------------------------------------------
# Konfiguracja i statystyki
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    n_genes: int           = 10
    pop_size: int          = 100
    n_generations: int     = 200
    mutation_rate: float   = 0.05
    mutation_sigma: float  = 0.3
    crossover_rate: float  = 0.8
    tournament_k: int      = 3
    elitism_n: int         = 5
    gene_bounds: tuple     = (0.0, 10.0)
    # Target OFN — trapezoid (a,b,c,d), a≤b≤c≤d
    target_a: float        = 2.0
    target_b: float        = 4.0
    target_c: float        = 6.0
    target_d: float        = 8.0
    seed: Optional[int]    = None


@dataclass
class GAStats:
    generation: int             = 0
    best_fitness: float         = float("inf")
    mean_fitness: float         = float("inf")
    best_individual: np.ndarray = field(default_factory=lambda: np.array([]))
    history_best: list          = field(default_factory=list)
    history_mean: list          = field(default_factory=list)
    elapsed_ms: float           = 0.0
    fitness_snapshots: list     = field(default_factory=list)
    fitness_global_max: float   = 1.0


# ---------------------------------------------------------------------------
# Główna klasa GA
# ---------------------------------------------------------------------------

class OFNGeneticAlgorithm:
    """
    Algorytm genetyczny dla trapezoidalnych OFN.

    Korzysta z pyofn do:
      - budowania targetu (pyofn.trapezoidal)
      - eksportu wyników (individual_to_ofns → lista pyofn.OFN)

    Wszystkie obliczenia fitness/selekcja/mutacja — wektoryzowane numpy.
    """

    def __init__(self, config: GAConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._build_target()
        self.population:   Optional[np.ndarray] = None
        self.fitness_vals: Optional[np.ndarray] = None
        self.stats = GAStats()
        self._running = False

    def _build_target(self):
        """
        Buduje tensor targetu (G, 4) korzystając z pyofn.trapezoidal.
        Parametry są wyciągane z ramion OFN — zapewnia spójność z biblioteką.
        """
        cfg = self.cfg
        # Tworzymy pyofn.OFN żeby uzyskać oficjalną reprezentację
        self.target_ofn = trapezoidal(
            cfg.target_a, cfg.target_b, cfg.target_c, cfg.target_d
        )
        params = ofn_to_params(self.target_ofn)             # (4,) posortowane
        self.target = np.tile(params, (cfg.n_genes, 1))     # (G, 4)

    def update_target(self, a: float, b: float, c: float, d: float):
        """Hot-swap celu bez restartowania populacji."""
        v = sorted([a, b, c, d])
        self.cfg.target_a, self.cfg.target_b = v[0], v[1]
        self.cfg.target_c, self.cfg.target_d = v[2], v[3]
        self._build_target()

    def initialize(self):
        lo, hi = self.cfg.gene_bounds
        P, G   = self.cfg.pop_size, self.cfg.n_genes
        raw    = self.rng.uniform(lo, hi, size=(P, G, 4))
        raw.sort(axis=-1)
        self.population   = raw
        self.fitness_vals = fitness_batch(self.population, self.target)
        self.stats        = GAStats()
        self._update_stats(0, 0.0)

    def step(self) -> GAStats:
        import time
        t0  = time.perf_counter()
        cfg = self.cfg
        pop = self.population
        fit = self.fitness_vals

        # Elityzm
        elite_idx = np.argpartition(fit, cfg.elitism_n)[:cfg.elitism_n]
        elite     = pop[elite_idx].copy()

        # Selekcja → krzyżowanie → mutacja
        parent_idx = tournament_select(fit, cfg.pop_size, cfg.tournament_k, self.rng)
        offspring  = crossover_uniform(pop[parent_idx], cfg.crossover_rate, self.rng)
        offspring  = mutate_gaussian(offspring, cfg.mutation_rate,
                                     cfg.mutation_sigma, cfg.gene_bounds, self.rng)

        # Wstawiamy elity
        offspring[-cfg.elitism_n:] = elite

        # Ewaluacja
        new_fit = fitness_batch(offspring, self.target)

        self.population   = offspring
        self.fitness_vals = new_fit
        elapsed = (time.perf_counter() - t0) * 1000
        self.stats.generation += 1
        self._update_stats(self.stats.generation, elapsed)
        return self.stats

    def _update_stats(self, gen: int, elapsed_ms: float):
        fit    = self.fitness_vals
        best_i = int(np.argmin(fit))
        self.stats.generation      = gen
        self.stats.best_fitness    = float(fit[best_i])
        self.stats.mean_fitness    = float(fit.mean())
        self.stats.best_individual = self.population[best_i].copy()
        self.stats.history_best.append(self.stats.best_fitness)
        self.stats.history_mean.append(self.stats.mean_fitness)
        self.stats.elapsed_ms      = elapsed_ms
        self.stats.fitness_snapshots.append(np.sort(fit).copy())
        cur_max = float(fit.max())
        if cur_max > self.stats.fitness_global_max:
            self.stats.fitness_global_max = cur_max

    def run(self, callback: Optional[Callable[[GAStats], bool]] = None) -> GAStats:
        self.initialize()
        self._running = True
        for _ in range(1, self.cfg.n_generations + 1):
            if not self._running: break
            stats = self.step()
            if callback and callback(stats) is False: break
        self._running = False
        return self.stats

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Eksport do pyofn
    # ------------------------------------------------------------------

    def best_as_ofns(self, n: int = 256) -> List[OFN]:
        """
        Zwraca najlepszy chromosom jako listę pyofn.OFN (trapezoidal).
        Używane przez app.py do wizualizacji przez pyofn.viz.
        """
        if self.stats.best_individual.size == 0:
            return []
        return individual_to_ofns(self.stats.best_individual, n=n)

    def best_as_params(self) -> np.ndarray:
        """Zwraca parametry (n_genes, 4) najlepszego osobnika."""
        if self.stats.best_individual.size == 0:
            return np.zeros((self.cfg.n_genes, 4))
        return self.stats.best_individual

    def get_fitness_matrix(self) -> np.ndarray:
        """Macierz (n_gens, pop_size) — wiersze=generacje, posortowane rosnąco."""
        snaps = self.stats.fitness_snapshots
        return np.stack(snaps, axis=0) if snaps else np.zeros((1, 1))

    def population_diversity(self) -> float:
        if self.population is None: return 0.0
        return float(self.population.std(axis=0).mean())
