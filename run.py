"""
run.py — launcher aplikacji OFN GA Explorer
Sprawdza zależności i uruchamia GUI.
"""
import sys

def check_deps():
    missing = []
    try:
        import numpy
        v = tuple(int(x) for x in numpy.__version__.split(".")[:2])
        if v < (1, 24):
            print(f"[WARN] NumPy {numpy.__version__} — zalecany ≥1.24")
    except ImportError:
        missing.append("numpy")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import tkinter
    except ImportError:
        missing.append("tkinter (python3-tk)")

    if missing:
        print(f"[ERROR] Brakujące zależności: {', '.join(missing)}")
        print("Zainstaluj: pip install numpy matplotlib")
        print("Na Linux: sudo apt-get install python3-tk")
        sys.exit(1)

if __name__ == "__main__":
    check_deps()
    from app import main
    main()
