"""
Demonstracja biblioteki pyofn.
Uruchomienie: python demo_ofn.py
Wykresy zapisywane są jako PNG w katalogu demo_output/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyofn import (
    OFN, triangular, triangular_left, trapezoidal, gaussian, about,
    plot, plot_many, plot_arithmetic, plot_direction_demo,
)

OUTPUT_DIR = "demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {path}")


# 1. Podstawowe kształty
section("1. Podstawowe kształty OFN")

A = triangular(1, 3, 5)           # trójkąt, kierunek w prawo
B = triangular_left(1, 3, 5)      # ten sam kształt, kierunek w lewo
T = trapezoidal(1, 2, 4, 5)
G = gaussian(mean=5, sigma=1.2)
S7 = about(7, spread=2)           # "około 7"

for name, ofn in [("triangular", A), ("triangular_left", B),
                  ("trapezoidal", T), ("gaussian", G), ("about_7", S7)]:
    print(f"  {name:16s} {ofn}")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#f8fafc")
plot_many([A, B, T, G, S7],
          labels=["Trójkąt (w prawo)", "Trójkąt (w lewo)", "Trapez", "Gauss", "Około 7"],
          title="Podstawowe kształty OFN", ax=ax)
save(fig, "01_shapes.png")


# 2. Arytmetyka
section("2. Arytmetyka OFN")

A = triangular(1, 3, 5)
B = triangular(2, 4, 6)
print(f"  A       = {A}")
print(f"  B       = {B}")
print(f"  A + B   = {A + B}")
print(f"  A - B   = {A - B}")
print(f"  A * B   = {A * B}")
print(f"  A*2 + 1 = {A * 2 + 1}")

save(plot_arithmetic(A, B, A + B, "+", "A", "B"), "02_addition.png")
save(plot_arithmetic(A, B, A - B, "-", "A", "B"), "03_subtraction.png")


# 3. Kompensacja nośnika
section("3. Kompensacja nośnika")

A_right = triangular(1, 3, 5)
A_left = triangular_left(1, 3, 5)

# dwie liczby o tym samym kierunku: nośnik rośnie
expand = A_right + A_right
print(f"  prawo + prawo: support = {expand.support}")

# liczby o przeciwnych kierunkach: nośnik się zawęża
compensate = A_right + A_left
print(f"  prawo + lewo:  support = {compensate.support}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#f8fafc")
fig.suptitle("Kompensacja nośnika w OFN", fontsize=13, fontweight="bold")
plot(A_right, label="A (w prawo)", color="#2563eb", ax=axes[0], title="A skierowana w prawo")
plot(A_left, label="A (w lewo)", color="#dc2626", ax=axes[1], title="A skierowana w lewo")
plot(compensate, label="suma", color="#16a34a", ax=axes[2], title="Suma (kompensacja)")
plt.tight_layout()
save(fig, "04_compensation.png")


# 4. Rozszerzanie nośnika przy wielokrotnym dodawaniu
section("4. Rozszerzanie nośnika przy kolejnych dodawaniach")

base = triangular(0, 2, 4)
running = base
supports = [base.support]
for _ in range(5):
    running = running + base
    supports.append(running.support)

for i, (lo, hi) in enumerate(supports):
    print(f"  krok {i}: [{lo:.2f}, {hi:.2f}]  szerokość = {hi - lo:.2f}")


# 5. Defuzzyfikacja
section("5. Defuzzyfikacja")

ofn = triangular(2, 5, 9)
print(f"  OFN          = {ofn}")
print(f"  COG          = {ofn.defuzzify_cog():.4f}")
print(f"  mean of core = {ofn.defuzzify_mean_core():.4f}")


# 6. Odległość Hamminga
section("6. Odległość Hamminga")

X = triangular(1, 3, 5)
Y = triangular(2, 4, 6)
Z = triangular(1, 3, 5)
print(f"  d(X, Y) = {X.distance_hamming(Y):.4f}")
print(f"  d(X, Z) = {X.distance_hamming(Z):.4f}  (powinno być 0)")


# 7. Wykres kierunków
section("7. Wykres kierunków")

save(plot_direction_demo(value=5, spread=2), "05_directions.png")


# 8. Serializacja
section("8. Serializacja")

original = gaussian(mean=7, sigma=1.5)
restored = OFN.from_dict(original.to_dict())
print(f"  błąd po zapisie i odczycie: {original.distance_hamming(restored):.2e}")

print(f"\nGotowe. Wykresy w katalogu {OUTPUT_DIR}/")
