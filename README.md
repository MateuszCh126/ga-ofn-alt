# OFN GA Explorer

Projekt składa się z dwóch części:

- `pyofn/` - biblioteka skierowanych liczb rozmytych (OFN, Ordered Fuzzy Numbers)
  wg teorii W. Kosińskiego, z arytmetyką wektoryzowaną w NumPy,
- `app.py` i `ga_core.py` - aplikacja desktopowa (tkinter + matplotlib), w której
  algorytm genetyczny dopasowuje populację trapezoidalnych OFN do zadanego celu.

## Wymagania

- Python 3.9 lub nowszy
- numpy, matplotlib
- tkinter (w Windows i macOS wbudowany w Pythona, w Linuksie pakiet `python3-tk`)

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install numpy matplotlib
pip install -e .
```

## Uruchomienie

Aplikacja GUI:

```bash
python run.py
```

Demo biblioteki (zapisuje wykresy PNG do katalogu `demo_output/`):

```bash
python demo_ofn.py
```

## Aplikacja

W lewym panelu ustawia się parametry chromosomu (liczba genów, zakres wartości),
populacji (rozmiar, liczba pokoleń), operatorów GA (mutacja, krzyżowanie, turniej,
elityzm) oraz cel, czyli trapez OFN o parametrach a, b, c, d. Przycisk
"Zastosuj nowy cel" podmienia cel w trakcie działania algorytmu.

Po prawej stronie rysowana jest historia fitness (najlepszy i średni osobnik),
geny najlepszego osobnika, porównanie celu z najlepszym genem oraz mapa ewolucji
całej populacji, którą można przeglądać generacja po generacji.

Fitness to odległość Hamminga między trapezem osobnika a celem, liczona
analitycznie dla całej populacji naraz, bez pętli po osobnikach.

## Biblioteka pyofn

```python
from pyofn import triangular, triangular_left, about, plot

A = triangular(1, 3, 5)        # trójkąt, kierunek w prawo
B = about(4, spread=1.5)       # "około 4"
C = A + B
print(C)                       # OFN(core=(7, 7), support=[3.5, 10.5], dir=1, n=512)
print(C.defuzzify_cog())
plot(C, title="A + B")
```

Dodanie liczb o przeciwnych kierunkach zawęża nośnik wyniku zamiast go rozszerzać:

```python
A_r = triangular(1, 3, 5)
A_l = triangular_left(1, 3, 5)
print((A_r + A_r).support)     # (2.0, 10.0)
print((A_r + A_l).support)     # (4.0, 8.0)
```

Konstruktory: `triangular`, `triangular_left`, `trapezoidal`, `gaussian`,
`singleton`, `linear_ofn`, `about`. Operatory: `+`, `-`, `*`, `/`, negacja,
działania ze skalarem.

Najważniejsze metody i właściwości klasy `OFN`: `up`, `down`, `direction`, `core`,
`support`, `membership(x)`, `defuzzify_cog()`, `defuzzify_mean_core()`,
`distance_hamming(other)`, `resample(n)`, `reverse()`, `to_dict()`, `OFN.from_dict()`.

Parametr `n` (domyślnie 512) określa liczbę punktów dyskretyzacji ramion.

## Pliki

```
app.py         GUI aplikacji
ga_core.py     silnik algorytmu genetycznego (numpy, bez GUI)
run.py         launcher sprawdzający zależności
demo_ofn.py    przykłady użycia biblioteki
pyofn/         biblioteka OFN (core.py, shapes.py, viz.py)
setup.py       instalacja pakietu pyofn
```
