# pyofn — Ordered Fuzzy Numbers for Python

Biblioteka implementuje **Skierowane Liczby Rozmyte** (OFN, ang. *Ordered Fuzzy Numbers*)
zgodnie z teorią W. Kosińskiego. Wszystkie operacje arytmetyczne są wektoryzowane przez NumPy.

## Instalacja

```bash
pip install numpy matplotlib   # zależności
pip install -e .               # instalacja lokalna (development)
```

## Szybki start

```python
from pyofn import triangular, about, plot, plot_arithmetic

# Tworzenie OFN
A = triangular(1, 3, 5)       # trójkąt (a=1, szczyt=3, c=5), kierunek →
B = about(4, spread=1.5)      # "około 4", symetryczny ±1.5

# Arytmetyka
C = A + B
D = A - B
E = A * 2 + 1                 # skalowanie i przesunięcie

# Info
print(A)
# OFN(core=(3, 3), support=[1, 5], dir=→, n=512)

# Defuzzyfikacja
print(C.defuzzify_cog())      # środek ciężkości

# Wizualizacja
plot(C, title="A + B")
plot_arithmetic(A, B, C, "+")
```

## Kluczowa własność: kompensacja nośnika

```python
from pyofn import triangular, triangular_left, plot_many

A_right = triangular(1, 3, 5)       # kierunek →
A_left  = triangular_left(1, 3, 5)  # kierunek ←, ten sam kształt

# Klasyczne rozmyte — nośnik rośnie:
expand = A_right + A_right
print(expand.support)    # (2.0, 10.0)  — szerszy!

# OFN z kompensacją — nośnik maleje:
compact = A_right + A_left
print(compact.support)   # (4.0, 8.0)  — węższy!
```

## API

### Klasa `OFN`

| Metoda / właściwość | Opis |
|---|---|
| `ofn.up`, `ofn.down` | Ramiona wznoszące/opadające (ndarray) |
| `ofn.direction` | `+1` (→), `-1` (←), `0` (singleton) |
| `ofn.core` | Jądro — punkt z μ=1 |
| `ofn.support` | Nośnik `(min_x, max_x)` |
| `ofn.membership(x)` | Funkcja przynależności μ(x) |
| `ofn.defuzzify_cog()` | Wyostrzanie — środek ciężkości |
| `ofn.defuzzify_mean_core()` | Wyostrzanie — średnia jądra |
| `ofn.distance_hamming(other)` | Odległość Hamminga |
| `ofn.resample(n)` | Zmień rozdzielczość dyskretyzacji |
| `ofn.reverse()` | Odwróć skierowanie |
| `ofn.to_dict()` / `OFN.from_dict()` | Serializacja JSON |

### Operatory

```python
C = A + B    # dodawanie
C = A - B    # odejmowanie (przez negację B)
C = A * B    # mnożenie
C = A / B    # dzielenie
C = -A       # negacja (odwraca kierunek)
C = A * 2.5  # mnożenie przez skalar
C = A + 3    # przesunięcie o stałą
```

### Konstruktory kształtów

```python
from pyofn import (
    triangular,       # triangular(a, b, c)      trójkąt →
    triangular_left,  # triangular_left(a, b, c) trójkąt ←
    trapezoidal,      # trapezoidal(a, b, c, d)  trapez →
    gaussian,         # gaussian(mean, sigma)     Gauss →
    singleton,        # singleton(value)          liczba ostra jako OFN
    linear_ofn,       # pełna kontrola nad ramionami
    about,            # about(value, spread)      "około value"
)
```

### Wizualizacja (wymaga matplotlib)

```python
from pyofn import plot, plot_many, plot_arithmetic, plot_direction_demo

plot(ofn, title="Moja OFN")
plot_many([A, B, C], labels=["A", "B", "C"])
plot_arithmetic(A, B, A + B, "+")
plot_direction_demo(value=5, spread=2)
```

## Parametr `n` — rozdzielczość dyskretyzacji

Domyślnie `n=512`. Większe `n` = wyższa precyzja, wolniejsze obliczenia.

```python
A = triangular(1, 3, 5, n=1024)  # wyższa precyzja
A = triangular(1, 3, 5, n=128)   # szybciej, mniej dokładnie
```

## Struktura projektu

```
pyofn/
├── pyofn/
│   ├── __init__.py   # publiczne API
│   ├── core.py       # klasa OFN, arytmetyka
│   ├── shapes.py     # konstruktory kształtów
│   └── viz.py        # wizualizacja (matplotlib)
├── demo_ofn.py        # przykłady użycia
└── setup.py
```

## Teoria

OFN (Kosiński, 2003) definiuje liczbę rozmytą jako parę funkcji ciągłych:

```
A = (x_up, x_down),  gdzie x_up, x_down: [0,1] → R
```

- `x_up` — ramię wznoszące (parametryzuje lewą stronę)
- `x_down` — ramię opadające (parametryzuje prawą stronę)
- **Kierunek** kodowany jest w monotoniczności `x_up`: rosnące = →, malejące = ←

Arytmetyka działa niezależnie na odpowiadających ramionach dla każdego `y ∈ [0,1]`:

```
(A + B)_up(y)   = A_up(y) + B_up(y)
(A + B)_down(y) = A_down(y) + B_down(y)
```

**Kluczowa zaleta vs klasyczne zbiory rozmyte:**
dodawanie OFN o przeciwnych kierunkach (→ + ←) *zmniejsza* nośnik wyniku,
eliminując problem "puchnącego nośnika".
