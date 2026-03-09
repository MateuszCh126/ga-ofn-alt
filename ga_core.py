"""
ga_core.py
==========
Algorytm genetyczny dla Skierowanych Liczb Rozmytych (OFN).

OPTYMALIZACJE:
  - Populacja jako tensor NumPy (P, G, 3) — zero pętli Python po osobnikach
  - Analityczna odległość Hamminga (bez dyskretyzacji) — czysta algebra
  - Krzyżowanie i mutacja w pełni wektoryzowane (np.where, broadcasting)
  - Selekcja turniejowa wektoryzowana
  - Elityzm bez kopiowania przez fancy indexing
  - Szybki RNG: np.random.Generator (PCG64)

Reprezentacja genu:
  Każdy gen = trójkąt OFN → (a, b, c)  gdzie a ≤ b ≤ c
  Chromsom = tablica (n_genes, 3)
  Populacja = tablica (pop_size, n_genes, 3)

Cel (target):
  Jedna OFN trójkątna (ta, tb, tc) — każdy gen porównywany do niej.
  Fitness = średnia analityczna odległość Hamminga po wszystkich genach.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Matematyka: analityczna całka ∫₀¹ |αy + β| dy
# ---------------------------------------------------------------------------

def _integral_abs_linear(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Oblicza ∫₀¹ |αy + β| dy analitycznie dla tablic α, β dowolnego kształtu.

    Dla ramienia liniowego OFN f(y) = start + y*(end - start):
      różnica między dwoma ramionami: δ(y) = (s1-s2) + y*((e1-s2)-(e2-s2))
    mamy α = e1-e2, β = s1-s2 (lub odwrotnie zaleznie od parametryzacji).

    Wzór:
      F(y) = αy²/2 + βy  (antypochodna |αy+β| przed rozważeniem znaku)
      Pierwiastek: y₀ = -β/α  (jeśli α≠0)
      - Jeśli y₀ ∉ (0,1): całka = |F(1)| = |α/2 + β|
      - Jeśli y₀ ∈ (0,1): całka = |F(y₀)| + |F(1) - F(y₀)|
    """
    # Antypochodna wartości bezwzględnej: ∫|αy+β|dy = sign(αy+β)*(αy²/2 + βy) + C
    # Szybciej: policzymy geometrycznie przez pole trapezu/trójkąta
    # Wartości na końcach: v0 = β, v1 = α + β
    v0 = beta                   # wartość αy+β przy y=0
    v1 = alpha + beta           # wartość αy+β przy y=1

    # Przypadek bez zmiany znaku (oba końce tego samego znaku lub zero)
    same_sign = v0 * v1 >= 0
    no_cross = 0.5 * np.abs(v0 + v1)   # pole trapezu = (|v0|+|v1|)/2

    # Przypadek ze zmianą znaku: y₀ = -β/α = v0/(v0-v1) ∈ (0,1)
    # Pole = (|v0|*y₀ + |v1|*(1-y₀)) / 2  — dwa trójkąty
    eps = 1e-14
    safe_denom = np.where(np.abs(v0 - v1) < eps, eps, v0 - v1)
    y0 = v0 / safe_denom                # y₀ ∈ [0,1] gdy zmiana znaku
    y0_clipped = np.clip(y0, 0.0, 1.0)
    cross = 0.5 * (np.abs(v0) * y0_clipped + np.abs(v1) * (1.0 - y0_clipped))

    return np.where(same_sign, no_cross, cross)


# ---------------------------------------------------------------------------
# Fitness: wektoryzowana analityczna odległość Hamminga
# ---------------------------------------------------------------------------

def fitness_batch(population: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Oblicza fitness całej populacji jedną operacją numpy.

    Parameters
    ----------
    population : (P, G, 3)  — P osobników, G genów, params [a,b,c]
    target     : (G, 3)     — cel: parametry trójkąta dla każdego genu

    Returns
    -------
    fitness : (P,)  — średnia odległość Hamminga (mniejsza = lepszy osobnik)

    Dla trójkąta OFN (a,b,c):
      ramię wznoszące:  x_up(y)   = a + y*(b-a)  = a*(1-y) + b*y
      ramię opadające:  x_down(y) = c - y*(c-b)  = c*(1-y) + b*y  [bo b+( 1-y)*(c-b)]

    Różnica ramion między osobnikiem A i targetem T:
      Δ_up(y)   = [a_A + y*(b_A - a_A)] - [a_T + y*(b_T - a_T)]
                = (a_A - a_T) + y*[(b_A - a_A) - (b_T - a_T)]
                ≡ β_up  + y * α_up

      Δ_down(y) = [c_A - y*(c_A - b_A)] - [c_T - y*(c_T - b_T)]
                = (c_A - c_T) - y*[(c_A - b_A) - (c_T - b_T)]
                ≡ β_dn  + y * α_dn
    """
    # Broadcast: population (P,G,3), target → (1,G,3)
    t = target[np.newaxis]                             # (1, G, 3)

    a_p, b_p, c_p = population[..., 0], population[..., 1], population[..., 2]
    a_t, b_t, c_t = t[..., 0], t[..., 1], t[..., 2]

    # Ramię wznoszące
    beta_up  = a_p - a_t                              # (P, G)
    alpha_up = (b_p - a_p) - (b_t - a_t)             # (P, G)

    # Ramię opadające (x_dn parametryzujemy y: 0→1 jako x_dn(y)=c-y*(c-b))
    beta_dn  = c_p - c_t                              # (P, G)
    alpha_dn = -(c_p - b_p) + (c_t - b_t)            # (P, G)

    d_up = _integral_abs_linear(alpha_up, beta_up)    # (P, G)
    d_dn = _integral_abs_linear(alpha_dn, beta_dn)    # (P, G)

    hamming = 0.5 * (d_up + d_dn)                    # (P, G) — per gen
    return hamming.mean(axis=1)                       # (P,)


# ---------------------------------------------------------------------------
# Operatory genetyczne — wszystkie wektoryzowane
# ---------------------------------------------------------------------------

def tournament_select(
    fitness: np.ndarray,
    n_select: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Wektoryzowana selekcja turniejowa.
    Dla każdego miejsca losuje k kandydatów i wybiera tego z najmniejszym fitness.

    Returns: indeksy wybranych osobników (n_select,)
    """
    P = len(fitness)
    # Losujemy k kandydatów dla każdego z n_select miejsc
    candidates = rng.integers(0, P, size=(n_select, k))   # (n_select, k)
    fit_cand   = fitness[candidates]                        # (n_select, k)
    winners    = candidates[np.arange(n_select), fit_cand.argmin(axis=1)]
    return winners                                          # (n_select,)


def crossover_uniform(
    population: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Wektoryzowane krzyżowanie jednorodne (uniform crossover) na poziomie GENÓW.

    Dla każdej pary (p1, p2) i każdego genu losuje, czy gen idzie do potomka.
    Wynikowe potomstwo ma tę samą liczbę osobników co populacja wejściowa.
    """
    P, G, D = population.shape
    offspring = population.copy()

    if P < 2:
        return offspring

    # Tworzymy pary (shuffle → pierwsze P//2 paruje z drugimi P//2)
    idx   = rng.permutation(P)
    half  = P // 2
    p1_i  = idx[:half]                       # (half,)
    p2_i  = idx[half: 2 * half]              # (half,)

    # Maska krzyżowania per par: czy w ogóle krzyżujemy?
    do_cross = rng.random(half) < crossover_rate     # (half,)

    # Maska per gen per para: który rodzic dostarcza gen potomkowi p1?
    gene_mask = rng.random((half, G)) < 0.5          # (half, G)
    gene_mask3 = gene_mask[:, :, np.newaxis]         # (half, G, 1)  broadcast na 3 params

    # Zerujemy maskę dla par które nie krzyżują
    do_cross3 = do_cross[:, np.newaxis, np.newaxis]  # (half, 1, 1)
    effective_mask = gene_mask3 & do_cross3          # (half, G, 1)

    p1_genes = population[p1_i]                      # (half, G, 3)
    p2_genes = population[p2_i]                      # (half, G, 3)

    offspring[p1_i] = np.where(effective_mask, p2_genes, p1_genes)
    offspring[p2_i] = np.where(effective_mask, p1_genes, p2_genes)

    return offspring


def mutate_gaussian(
    population: np.ndarray,
    mutation_rate: float,
    sigma: float,
    bounds: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Gaussowska mutacja z zachowaniem ograniczenia a ≤ b ≤ c.

    Po dodaniu szumu wartości są sortowane wzdłuż ostatniej osi
    (co automatycznie przywraca porządek a ≤ b ≤ c).
    """
    P, G, D = population.shape
    mask  = rng.random((P, G, D)) < mutation_rate   # (P, G, 3)
    noise = rng.normal(0.0, sigma, (P, G, D))        # (P, G, 3)

    mutated = np.where(mask, population + noise, population)
    mutated = np.clip(mutated, bounds[0], bounds[1])
    mutated.sort(axis=-1)   # wymusza a ≤ b ≤ c  (bez nowych alokacji)
    return mutated


# ---------------------------------------------------------------------------
# Klasa GA
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    n_genes: int          = 10
    pop_size: int         = 100
    n_generations: int    = 200
    mutation_rate: float  = 0.05
    mutation_sigma: float = 0.3
    crossover_rate: float = 0.8
    tournament_k: int     = 3
    elitism_n: int        = 5
    gene_bounds: tuple    = (0.0, 10.0)
    # Target OFN — trójkąt (a, b, c) — replicated across all genes
    target_a: float       = 3.0
    target_b: float       = 5.0
    target_c: float       = 7.0
    seed: Optional[int]   = None


@dataclass
class GAStats:
    generation: int        = 0
    best_fitness: float    = float("inf")
    mean_fitness: float    = float("inf")
    best_individual: np.ndarray = field(default_factory=lambda: np.array([]))
    history_best: list     = field(default_factory=list)
    history_mean: list     = field(default_factory=list)
    elapsed_ms: float      = 0.0
    # Posortowany snapshot fitness całej populacji na każdą generację
    # Lista ndarray (pop_size,) — używana do heatmapy ewolucji
    fitness_snapshots: list = field(default_factory=list)
    fitness_global_max: float = 1.0   # do normalizacji koloru


class OFNGeneticAlgorithm:
    """
    Algorytm genetyczny dla OFN — w pełni wektoryzowany.

    Wszystkie obliczenia na tensorach numpy.
    Zero pętli Python po osobnikach w krytycznej ścieżce.
    """

    def __init__(self, config: GAConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._build_target()
        self.population: Optional[np.ndarray] = None
        self.fitness_vals: Optional[np.ndarray] = None
        self.stats = GAStats()
        self._running = False

    def _build_target(self):
        """Buduje tensor targetu (G, 3) z konfiguracji."""
        cfg = self.cfg
        params = np.array([cfg.target_a, cfg.target_b, cfg.target_c], dtype=np.float64)
        params.sort()   # upewniamy się a ≤ b ≤ c
        self.target = np.tile(params, (cfg.n_genes, 1))  # (G, 3)

    def update_target(self, a: float, b: float, c: float):
        """Aktualizuje cel bez restartowania populacji."""
        self.cfg.target_a, self.cfg.target_b, self.cfg.target_c = a, b, c
        self._build_target()

    def initialize(self):
        """Inicjalizuje losową populację w granicach gene_bounds."""
        cfg = self.cfg
        lo, hi = cfg.gene_bounds
        P, G = cfg.pop_size, cfg.n_genes

        # Losujemy trzy wartości i sortujemy → gwarantowane a ≤ b ≤ c
        raw = self.rng.uniform(lo, hi, size=(P, G, 3))
        raw.sort(axis=-1)
        self.population = raw
        self.fitness_vals = fitness_batch(self.population, self.target)
        self.stats = GAStats()
        self._update_stats(0, 0.0)

    def step(self) -> GAStats:
        """Wykonuje jedną generację GA. Zwraca zaktualizowane statystyki."""
        import time
        t0 = time.perf_counter()

        cfg = self.cfg
        pop = self.population
        fit = self.fitness_vals

        # 1. Elityzm — zachowaj najlepszych
        elite_idx = np.argpartition(fit, cfg.elitism_n)[:cfg.elitism_n]
        elite     = pop[elite_idx].copy()

        # 2. Selekcja rodziców (cała populacja jako rodzice — selekcja przez turniej)
        parent_idx = tournament_select(fit, cfg.pop_size, cfg.tournament_k, self.rng)
        parents    = pop[parent_idx]

        # 3. Krzyżowanie
        offspring = crossover_uniform(parents, cfg.crossover_rate, self.rng)

        # 4. Mutacja
        offspring = mutate_gaussian(
            offspring, cfg.mutation_rate, cfg.mutation_sigma, cfg.gene_bounds, self.rng
        )

        # 5. Wstawiamy elity (zastępujemy ostatnich elitism_n osobników)
        offspring[-cfg.elitism_n:] = elite

        # 6. Ewaluacja
        new_fit = fitness_batch(offspring, self.target)

        self.population   = offspring
        self.fitness_vals = new_fit

        elapsed = (time.perf_counter() - t0) * 1000
        self.stats.generation += 1
        self._update_stats(self.stats.generation, elapsed)
        return self.stats

    def _update_stats(self, gen: int, elapsed_ms: float):
        fit = self.fitness_vals
        best_i = int(np.argmin(fit))
        self.stats.generation    = gen
        self.stats.best_fitness  = float(fit[best_i])
        self.stats.mean_fitness  = float(fit.mean())
        self.stats.best_individual = self.population[best_i].copy()
        self.stats.history_best.append(self.stats.best_fitness)
        self.stats.history_mean.append(self.stats.mean_fitness)
        self.stats.elapsed_ms    = elapsed_ms
        # Snapshot posortowanych wartości fitness dla heatmapy
        self.stats.fitness_snapshots.append(np.sort(fit).copy())
        cur_max = float(fit.max())
        if cur_max > self.stats.fitness_global_max:
            self.stats.fitness_global_max = cur_max

    def run(
        self,
        callback: Optional[Callable[[GAStats], bool]] = None,
    ) -> GAStats:
        """
        Uruchamia pełną ewolucję.

        Parameters
        ----------
        callback : f(stats) -> bool | None
            Wywoływana po każdej generacji.
            Zwróć False aby przerwać.
        """
        self.initialize()
        self._running = True
        for gen in range(1, self.cfg.n_generations + 1):
            if not self._running:
                break
            stats = self.step()
            if callback is not None:
                if callback(stats) is False:
                    break
        self._running = False
        return self.stats

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------

    def best_as_ofn_params(self) -> np.ndarray:
        """Zwraca parametry najlepszego osobnika: (n_genes, 3)."""
        if self.stats.best_individual.size == 0:
            return np.zeros((self.cfg.n_genes, 3))
        return self.stats.best_individual

    def get_fitness_matrix(self) -> np.ndarray:
        """
        Zwraca macierz fitness (n_gens, pop_size) z całej historii.
        Wiersze = generacje, kolumny = osobnicy posortowani rosnąco (najlepszy [0]).
        Używana do heatmapy ewolucji.
        """
        snaps = self.stats.fitness_snapshots
        if not snaps:
            return np.zeros((1, 1))
        return np.stack(snaps, axis=0)   # (n_gens, pop_size)

    def population_diversity(self) -> float:
        """Różnorodność populacji: średnie odchylenie standardowe parametrów genów."""
        if self.population is None:
            return 0.0
        return float(self.population.std(axis=0).mean())
