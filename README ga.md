# OFN Genetic Algorithm Explorer

Desktopowa aplikacja łącząca **Algorytm Genetyczny** ze **Skierowanymi Liczbami Rozmytymi** (OFN).

## Wymagania

```bash
pip install numpy matplotlib
# Linux: sudo apt-get install python3-tk
# Windows/macOS: tkinter wbudowany w Python
```

## Uruchomienie

```bash
python run.py
# lub bezpośrednio:
python app.py
```

## Jak używać

### Panel lewy — konfiguracja

| Sekcja | Parametr | Opis |
|---|---|---|
| **Target OFN** | a, b, c | Trójkąt OFN który GA ma odtworzyć |
| | Zastosuj nowy cel | Hot-swap celu w trakcie ewolucji! |
| **Chromosom** | Liczba genów | Ile OFN tworzy jeden osobnik |
| | Zakres min/max | Dziedzina wartości x dla OFN |
| **Populacja** | Rozmiar | Liczba osobników |
| | Liczba pokoleń | Ile iteracji GA |
| **Operatory GA** | Mut. rate | Prawdopodobieństwo mutacji genu |
| | Mut. sigma | Odchylenie std szumu mutacji |
| | Cross. rate | Prawdopodobieństwo krzyżowania pary |
| | Turniej k | Rozmiar turnieju selekcji |
| | Elityzm n | Ilu najlepszych przeżywa bez zmian |
| **Wyświetlanie** | Odśwież co N | Co ile generacji odświeżać wykresy |

### Panel prawy — wykresy

1. **Historia Fitness** — best (zielony) i mean (żółty) przez generacje
2. **Geny najlepszego osobnika** — wszystkie OFN chromosomu, każdy innym kolorem
3. **Cel vs Najlepszy** — porównanie targetu z uśrednionymi genami najlepszego

### Pasek statusu

```
GEN: 127/300  |  BEST: 0.00234  |  MEAN: 0.14521  |  DIV: 0.823  |  SPEED: 847 gen/s
```
- **BEST** — odległość Hamminga najlepszego od celu (cel: 0)
- **DIV** — różnorodność populacji (śr. odch. std parametrów)
- **SPEED** — generacje na sekundę

## Architektura techniczna

### Optymalizacje wydajności

```
Populacja:  tensor NumPy (P, G, 3) — P osobników, G genów, [a,b,c] params
             ↓
Fitness:    ANALITYCZNY Hamming — całka ∫₀¹|αy+β|dy bez dyskretyzacji
             → O(P·G) operacji numpy, zero pętli Python
             ↓
Selekcja:   wektoryzowany turniej — rng.integers + argmin po macierzy
             ↓
Krzyżowanie: np.where na tensorze (half, G, 3) — jedno wywołanie
             ↓
Mutacja:    maska boolowska + noise — broadcasting NumPy
             ↓
Elityzm:    argpartition — O(P) zamiast O(P·log P) sortowania
```

### Wielowątkowość

```
Wątek GA ──────────────────────────────────────────────────────────►
           step() → step() → step() → queue.put(stats)

Wątek GUI (tkinter) ──────────────────────────────────────────────►
           root.after(40ms, poll_queue) → update_plots()
```

Kolejka ograniczona do 200 elementów — GUI nie blokuje GA przy wolnym rysowaniu.

### Wydajność (benchmark)

| pop_size | n_genes | gen/s |
|---|---|---|
| 100 | 10 | ~5000 |
| 500 | 20 | ~800 |
| 1000 | 50 | ~200 |

## Pliki

```
ofn_ga/
├── ga_core.py   — silnik GA (numpy, bez GUI)
├── app.py       — GUI tkinter + matplotlib
└── run.py       — launcher z weryfikacją zależności
```

## Integracja z pyofn

`ga_core.py` nie zależy od `pyofn` — używa własnej wektoryzowanej
reprezentacji parametryczną `(a,b,c)`.
Aby wyeksportować wynik do pyofn:

```python
from ga_core import OFNGeneticAlgorithm, GAConfig
from pyofn import triangular

cfg = GAConfig(n_genes=10, pop_size=200, n_generations=500)
ga  = OFNGeneticAlgorithm(cfg)
ga.run()

best_params = ga.best_as_ofn_params()   # (n_genes, 3)
ofns = [triangular(a, b, c) for a, b, c in best_params]
```
