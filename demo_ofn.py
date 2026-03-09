"""
demo_ofn.py
===========
Kompletna demonstracja biblioteki pyofn.
Uruchom: python demo_ofn.py
Zapisuje wykresy jako PNG.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyofn import (
    OFN, triangular, triangular_left, trapezoidal,
    gaussian, singleton, about,
    plot, plot_many, plot_arithmetic, plot_direction_demo,
)

OUTPUT_DIR = "demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Tworzenie podstawowych kształtów
# ============================================================
print("=" * 60)
print("1. Podstawowe kształty OFN")
print("=" * 60)

A = triangular(1, 3, 5)           # trójkąt symetryczny →
B = triangular_left(1, 3, 5)      # ten sam kształt, kierunek ←
T = trapezoidal(1, 2, 4, 5)       # trapez →
G = gaussian(mean=5, sigma=1.2)   # Gauss →
S7 = about(7, spread=2)           # "około 7"

for name, ofn in [("triangular_R", A), ("triangular_L", B),
                  ("trapezoidal", T), ("gaussian", G), ("about_7", S7)]:
    print(f"  {name:15s}: {ofn}")

# ============================================================
# 2. Wykres wszystkich kształtów
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#f8fafc")
ax = plot_many(
    [A, B, T, G, S7],
    labels=["Trójkąt →", "Trójkąt ←", "Trapez →", "Gauss →", "Około 7"],
    title="Podstawowe kształty OFN",
    ax=ax,
)
fig.savefig(f"{OUTPUT_DIR}/01_shapes.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Zapisano: {OUTPUT_DIR}/01_shapes.png")


# ============================================================
# 3. Arytmetyka OFN — kluczowa własność: zachowanie kierunku
# ============================================================
print("\n" + "=" * 60)
print("2. Arytmetyka OFN")
print("=" * 60)

A = triangular(1, 3, 5)           # kierunek →
B = triangular(2, 4, 6)           # kierunek →
C_add = A + B
C_sub = A - B
C_mul = A * B
C_scalar = A * 2 + 1              # skalowanie i przesunięcie

print(f"  A         = {A}")
print(f"  B         = {B}")
print(f"  A + B     = {C_add}")
print(f"  A - B     = {C_sub}")
print(f"  A * B     = {C_mul}")
print(f"  A*2 + 1   = {C_scalar}")

fig = plot_arithmetic(A, B, C_add, "+", "A", "B")
fig.savefig(f"{OUTPUT_DIR}/02_addition.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {OUTPUT_DIR}/02_addition.png")

fig = plot_arithmetic(A, B, C_sub, "−", "A", "B")
fig.savefig(f"{OUTPUT_DIR}/03_subtraction.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {OUTPUT_DIR}/03_subtraction.png")


# ============================================================
# 4. Problem "puchnącego nośnika" vs kompensacja OFN
# ============================================================
print("\n" + "=" * 60)
print("3. Kompensacja nośnika: OFN → + OFN ←")
print("=" * 60)

A_right = triangular(1, 3, 5)      # kierunek →, support = [1, 5]
A_left  = triangular_left(1, 3, 5) # kierunek ←, support = [1, 5]

# Klasyczne mnożenie: A → + A → — nośnik rośnie
expand = A_right + A_right
print(f"  A→ + A→  support = {expand.support}")  # powiększony

# Kompensacja: A → + A ← — nośnik się zmniejsza!
compensate = A_right + A_left
print(f"  A→ + A← support = {compensate.support}")  # skompensowany

fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#f8fafc")
fig.suptitle("Kompensacja nośnika w OFN", fontsize=13, fontweight="bold")
plot(A_right,    label="A →",    color="#2563eb", ax=axes[0], title="A (kierunek →)")
plot(A_left,     label="A ←",    color="#dc2626", ax=axes[1], title="A (kierunek ←)")
plot(compensate, label="A→ + A←",color="#16a34a", ax=axes[2], title="A→ + A← (kompensacja)")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_compensation.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {OUTPUT_DIR}/04_compensation.png")


# ============================================================
# 5. Porównanie OFN z klasycznym dodawaniem rozmytym
#    (Pokazuje: klasyczne dodawanie zawsze rozszerza nośnik)
# ============================================================
print("\n" + "=" * 60)
print("4. Demonstracja 'puchnącego nośnika' (klasyczne fuzzy)")
print("=" * 60)

base = triangular(0, 2, 4)
supports = [base.support]
running = base
for i in range(5):
    running = running + base   # seria dodawań
    supports.append(running.support)

print("  Nośniki po kolejnych dodaniach A→ + A→ + ...:")
for i, (lo, hi) in enumerate(supports):
    print(f"    krok {i}: [{lo:.2f}, {hi:.2f}]  szerokość = {hi-lo:.2f}")


# ============================================================
# 6. Defuzzyfikacja
# ============================================================
print("\n" + "=" * 60)
print("5. Defuzzyfikacja")
print("=" * 60)

ofn = triangular(2, 5, 9)
print(f"  OFN           = {ofn}")
print(f"  COG           = {ofn.defuzzify_cog():.4f}")
print(f"  Mean of core  = {ofn.defuzzify_mean_core():.4f}")


# ============================================================
# 7. Odległość Hamminga
# ============================================================
print("\n" + "=" * 60)
print("6. Odległość Hamminga")
print("=" * 60)

X = triangular(1, 3, 5)
Y = triangular(2, 4, 6)
Z = triangular(1, 3, 5)  # identyczny z X
print(f"  d(X, Y) = {X.distance_hamming(Y):.4f}")
print(f"  d(X, Z) = {X.distance_hamming(Z):.4f}  (powinno być 0)")


# ============================================================
# 8. Wykres kierunków
# ============================================================
fig = plot_direction_demo(value=5, spread=2)
fig.savefig(f"{OUTPUT_DIR}/05_directions.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  Zapisano: {OUTPUT_DIR}/05_directions.png")


# ============================================================
# 9. Serializacja
# ============================================================
print("\n" + "=" * 60)
print("7. Serializacja / deserializacja")
print("=" * 60)

original = gaussian(mean=7, sigma=1.5)
d = original.to_dict()
restored = OFN.from_dict(d)
err = original.distance_hamming(restored)
print(f"  Błąd po round-trip serializacji: {err:.2e}")

print("\n✓ Demo zakończone. Wykresy w katalogu:", OUTPUT_DIR)
