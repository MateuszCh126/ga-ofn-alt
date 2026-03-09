"""
app.py — OFN Genetic Algorithm Explorer
Wymagania: pip install numpy matplotlib  +  python3-tk
Uruchomienie: python run.py
"""
from __future__ import annotations
import queue, threading, time, tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from ga_core import OFNGeneticAlgorithm, GAConfig, GAStats

# ── paleta ──────────────────────────────────────────────────────────────────
DARK   = "#0d1117"; PANEL  = "#161b22"; BORDER = "#30363d"
TEXT   = "#e6edf3"; MUTED  = "#8b949e"; ACCENT = "#58a6ff"
GREEN  = "#3fb950"; RED    = "#f85149"; YELLOW = "#d29922"; PURPLE = "#bc8cff"

FONT_MONO = ("Consolas", 9)
FONT_UI   = ("Segoe UI", 9)
FONT_H    = ("Segoe UI", 10, "bold")

MPL_STYLE = {
    "figure.facecolor": DARK, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": BORDER, "text.color": TEXT, "lines.linewidth": 1.8,
}

# Zielony (blisko 0) → żółty → ciemny czerwony (daleko od celu)
EVO_CMAP = LinearSegmentedColormap.from_list("ofn_evo", [
    (0.00, "#3fb950"), (0.25, "#56d364"), (0.50, "#d29922"),
    (0.75, "#f85149"), (1.00, "#7d0000"),
])

# ── pomocnicze widgety ───────────────────────────────────────────────────────

def _lbl(parent, text, font=FONT_UI, color=TEXT, **kw):
    return tk.Label(parent, text=text, bg=PANEL, fg=color, font=font, **kw)

def _sep(parent):
    return tk.Frame(parent, bg=BORDER, height=1)

class LabeledSlider(tk.Frame):
    def __init__(self, parent, label, from_, to, default, fmt="{:.0f}", cb=None, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self._fmt = fmt; self._cb = cb
        row = tk.Frame(self, bg=PANEL); row.pack(fill="x")
        tk.Label(row, text=label, bg=PANEL, fg=TEXT, font=FONT_UI, anchor="w").pack(side="left")
        self._v = tk.Label(row, text=fmt.format(default), bg=PANEL, fg=ACCENT,
                           font=FONT_MONO, width=7, anchor="e")
        self._v.pack(side="right")
        self.var = tk.DoubleVar(value=default)
        ttk.Scale(self, from_=from_, to=to, variable=self.var,
                  orient="horizontal", command=self._chg).pack(fill="x", padx=2)
        r2 = tk.Frame(self, bg=PANEL); r2.pack(fill="x")
        tk.Label(r2, text=str(from_), bg=PANEL, fg=MUTED, font=("Consolas",7)).pack(side="left")
        tk.Label(r2, text=str(to),   bg=PANEL, fg=MUTED, font=("Consolas",7)).pack(side="right")

    def _chg(self, *_):
        self._v.config(text=self._fmt.format(self.var.get()))
        if self._cb: self._cb(self.var.get())

    def get(self): return self.var.get()
    def set(self, v):
        self.var.set(v); self._v.config(text=self._fmt.format(v))

class StatusBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=DARK, height=24); self._items = {}

    def add(self, key, text="", color=TEXT):
        lbl = tk.Label(self, text=text, bg=DARK, fg=color, font=FONT_MONO, padx=10, pady=2)
        lbl.pack(side="left")
        tk.Frame(self, bg=BORDER, width=1).pack(side="left", fill="y", pady=2)
        self._items[key] = lbl

    def upd(self, key, text, color=TEXT):
        if key in self._items: self._items[key].config(text=text, fg=color)

# ── główna aplikacja ─────────────────────────────────────────────────────────

class OFNGAApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()
        self._setup_style()
        self._build_ui()
        self._init_mpl_main()
        self._init_mpl_evo()

        self.ga: Optional[OFNGeneticAlgorithm] = None
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[GAStats] = queue.Queue(maxsize=300)
        self._running = False
        self._last_stats: Optional[GAStats] = None
        self._n_gen_total = 0
        self._refresh_every = 5

        # stan ewolucji
        self._evo_matrix: Optional[np.ndarray] = None
        self._evo_fit_max: float = 1.0
        self._evo_cursor: int = 0

        self._poll()

    # ── okno ──────────────────────────────────────────────────────────────

    def _setup_window(self):
        self.root.title("OFN Genetic Algorithm Explorer")
        self.root.configure(bg=DARK)
        W, H = 1400, 930
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.root.resizable(True, True)
        self.root.minsize(1000, 700)

    def _setup_style(self):
        s = ttk.Style(self.root); s.theme_use("clam")
        s.configure("TScale", background=PANEL, troughcolor=DARK, sliderlength=14, sliderrelief="flat")
        s.map("TScale", background=[("active", PANEL)])

    # ── UI skeleton ────────────────────────────────────────────────────────

    def _build_ui(self):
        # nagłówek
        hdr = tk.Frame(self.root, bg=DARK, pady=6); hdr.pack(fill="x")
        tk.Label(hdr, text="⬡  OFN Genetic Algorithm Explorer",
                 bg=DARK, fg=ACCENT, font=("Segoe UI",13,"bold")).pack(side="left", padx=16)
        self._gen_lbl = tk.Label(hdr, text="GEN  0 / 0", bg=DARK, fg=MUTED, font=FONT_MONO)
        self._gen_lbl.pack(side="right", padx=16)
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        main = tk.Frame(self.root, bg=DARK); main.pack(fill="both", expand=True)

        # lewy panel
        self._left = tk.Frame(main, bg=PANEL, width=265)
        self._left.pack(side="left", fill="y", padx=(6,0), pady=6)
        self._left.pack_propagate(False)
        self._build_left(self._left)

        # prawa strona: górna (wykresy GA) + dolna (ewolucja)
        right = tk.Frame(main, bg=DARK)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self._top_frame = tk.Frame(right, bg=DARK)
        self._top_frame.pack(fill="both", expand=True)

        tk.Frame(right, bg=BORDER, height=2).pack(fill="x", pady=(3,0))

        self._evo_outer = tk.Frame(right, bg=PANEL, height=295)
        self._evo_outer.pack(fill="x")
        self._evo_outer.pack_propagate(False)
        self._build_evo_panel(self._evo_outer)

        # status
        self._sb = StatusBar(self.root); self._sb.pack(fill="x", side="bottom")
        self._sb.add("gen",   "GEN: —",       MUTED)
        self._sb.add("best",  "BEST: —",       GREEN)
        self._sb.add("mean",  "MEAN: —",       YELLOW)
        self._sb.add("div",   "DIV: —",        PURPLE)
        self._sb.add("speed", "SPEED: —",      MUTED)
        self._sb.add("state", "⏹ Zatrzymany", RED)

    # ── lewy panel ────────────────────────────────────────────────────────

    def _build_left(self, parent):
        cv = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        inner = tk.Frame(cv, bg=PANEL)
        cv.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")
        cv.configure(yscrollcommand=sb.set)
        _id = cv.create_window((0,0), window=inner, anchor="nw")

        def _c(e):
            cv.configure(scrollregion=cv.bbox("all"))
            cv.itemconfig(_id, width=cv.winfo_width())
        inner.bind("<Configure>", _c); cv.bind("<Configure>", _c)
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta/120), "units"))

        P = dict(padx=8, pady=3, fill="x")
        def H(t): return _lbl(inner, t, font=FONT_H, color=ACCENT)
        def S(): _sep(inner).pack(fill="x", padx=6, pady=3)

        H("▶  TARGET OFN").pack(**P); S()
        _lbl(inner, "Trójkąt OFN (a ≤ b ≤ c)", color=MUTED).pack(anchor="w", padx=8)
        self.s_ta = LabeledSlider(inner, "a  (lewy)",   0, 10, 3.0, fmt="{:.2f}"); self.s_ta.pack(**P)
        self.s_tb = LabeledSlider(inner, "b  (szczyt)", 0, 10, 5.0, fmt="{:.2f}"); self.s_tb.pack(**P)
        self.s_tc = LabeledSlider(inner, "c  (prawy)",  0, 10, 7.0, fmt="{:.2f}"); self.s_tc.pack(**P)
        tk.Button(inner, text="↯  Zastosuj nowy cel", bg=ACCENT, fg=DARK,
                  font=("Segoe UI",9,"bold"), relief="flat", cursor="hand2",
                  command=self._apply_target).pack(padx=8, pady=(2,6), fill="x")

        S(); H("▶  CHROMOSOM").pack(**P)
        self.s_genes = LabeledSlider(inner, "Liczba genów",  2, 50,  10);  self.s_genes.pack(**P)
        self.s_blo   = LabeledSlider(inner, "Zakres min", -10,  0,   0, fmt="{:.1f}"); self.s_blo.pack(**P)
        self.s_bhi   = LabeledSlider(inner, "Zakres max",   0, 20,  10, fmt="{:.1f}"); self.s_bhi.pack(**P)

        S(); H("▶  POPULACJA").pack(**P)
        self.s_pop = LabeledSlider(inner, "Rozmiar populacji",  10, 1000, 150); self.s_pop.pack(**P)
        self.s_gen = LabeledSlider(inner, "Liczba pokoleń",     10, 2000, 300); self.s_gen.pack(**P)

        S(); H("▶  OPERATORY GA").pack(**P)
        self.s_mut   = LabeledSlider(inner, "Mut. rate",  0, .5,  .05, fmt="{:.3f}"); self.s_mut.pack(**P)
        self.s_sigma = LabeledSlider(inner, "Mut. sigma", .01,2., .3,  fmt="{:.2f}"); self.s_sigma.pack(**P)
        self.s_cross = LabeledSlider(inner, "Cross. rate",0, 1.,  .8,  fmt="{:.2f}"); self.s_cross.pack(**P)
        self.s_tour  = LabeledSlider(inner, "Turniej k",  2, 10,   3);                self.s_tour.pack(**P)
        self.s_elite = LabeledSlider(inner, "Elityzm n",  0, 20,   5);                self.s_elite.pack(**P)

        S(); H("▶  WYŚWIETLANIE").pack(**P)
        self.s_refresh = LabeledSlider(inner, "Odśwież co N gen.", 1, 50, 5); self.s_refresh.pack(**P)

        S()
        self._btn_run = tk.Button(inner, text="▶  START", bg=GREEN, fg=DARK,
                                   font=("Segoe UI",10,"bold"), relief="flat",
                                   cursor="hand2", height=2, command=self._on_start)
        self._btn_run.pack(padx=8, pady=(4,2), fill="x")
        self._btn_stop = tk.Button(inner, text="⏹  STOP", bg=RED, fg="white",
                                    font=("Segoe UI",10,"bold"), relief="flat",
                                    cursor="hand2", state="disabled", command=self._on_stop)
        self._btn_stop.pack(padx=8, pady=2, fill="x")
        tk.Button(inner, text="↺  RESET", bg=BORDER, fg=TEXT, font=FONT_UI,
                  relief="flat", cursor="hand2", command=self._on_reset).pack(padx=8, pady=(2,10), fill="x")

    # ── panel ewolucji ─────────────────────────────────────────────────────

    def _build_evo_panel(self, parent):
        # tytuł + info
        top = tk.Frame(parent, bg=PANEL); top.pack(fill="x", padx=8, pady=(4,0))
        tk.Label(top, text="  PRZEBIEG EWOLUCJI", bg=PANEL, fg=ACCENT,
                 font=FONT_H).pack(side="left")
        self._evo_info = tk.Label(top, text="gen: —  |  best: —  |  mean: —",
                                   bg=PANEL, fg=MUTED, font=FONT_MONO)
        self._evo_info.pack(side="right", padx=6)

        # matplotlib canvas
        self._evo_fig = Figure(figsize=(10, 2.7), facecolor=PANEL)
        self._evo_canvas = FigureCanvasTkAgg(self._evo_fig, master=parent)
        self._evo_canvas.get_tk_widget().pack(fill="both", expand=True)

        gs = gridspec.GridSpec(1, 2, figure=self._evo_fig, width_ratios=[4,1],
                               left=0.04, right=0.97, top=0.87, bottom=0.17, wspace=0.06)
        self._ax_heat    = self._evo_fig.add_subplot(gs[0,0])
        self._ax_profile = self._evo_fig.add_subplot(gs[0,1])

        for ax in [self._ax_heat, self._ax_profile]:
            ax.set_facecolor(DARK)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.tick_params(colors=MUTED, labelsize=7)

        self._ax_heat.set_title(
            "Ewolucja populacji  ·  🟢 bliski celu  ──  🔴 daleki od celu  ·  kliknij lub przeciągnij suwak",
            color=TEXT, fontsize=8, pad=3)
        self._ax_heat.set_xlabel("Generacja", color=MUTED, fontsize=7)
        self._ax_heat.set_ylabel("Rank osobnika", color=MUTED, fontsize=7)
        self._ax_profile.set_title("Profil", color=TEXT, fontsize=7, pad=2)
        self._ax_profile.set_xlabel("Fitness", color=MUTED, fontsize=6)
        self._ax_profile.yaxis.set_visible(False)

        # kontrolki scrubber
        ctrl = tk.Frame(parent, bg=PANEL); ctrl.pack(fill="x", padx=8, pady=(2,4))

        def btn(t, cmd, w=3):
            return tk.Button(ctrl, text=t, bg=BORDER, fg=TEXT, font=FONT_MONO,
                             relief="flat", cursor="hand2", width=w, command=cmd)

        btn("⏮", self._evo_first).pack(side="left", padx=(0,2))
        btn("◀", self._evo_prev ).pack(side="left", padx=2)
        btn("▶", self._evo_next ).pack(side="left", padx=2)
        btn("⏭", self._evo_last ).pack(side="left", padx=2)

        tk.Label(ctrl, text="  GEN:", bg=PANEL, fg=MUTED,
                 font=FONT_MONO).pack(side="left", padx=(8,2))
        self._evo_sv  = tk.IntVar(value=0)
        self._evo_sl  = ttk.Scale(ctrl, from_=0, to=1, variable=self._evo_sv,
                                   orient="horizontal", command=self._evo_sl_moved)
        self._evo_sl.pack(side="left", fill="x", expand=True, padx=6)
        self._evo_gl  = tk.Label(ctrl, text="0", bg=PANEL, fg=ACCENT,
                                  font=FONT_MONO, width=5)
        self._evo_gl.pack(side="left", padx=(2,8))

        self._evo_canvas.mpl_connect("button_press_event",  self._evo_click)
        self._evo_canvas.mpl_connect("motion_notify_event", self._evo_drag)
        self._evo_canvas.draw()

    # ── matplotlib główny ──────────────────────────────────────────────────

    def _init_mpl_main(self):
        plt.rcParams.update(MPL_STYLE)
        self._fig = Figure(figsize=(9,5.2), facecolor=DARK)
        gs = gridspec.GridSpec(2,2, figure=self._fig, hspace=0.42, wspace=0.3,
                               left=0.07, right=0.97, top=0.93, bottom=0.10)
        self._ax_fit  = self._fig.add_subplot(gs[0,:])
        self._ax_best = self._fig.add_subplot(gs[1,0])
        self._ax_cmp  = self._fig.add_subplot(gs[1,1])

        for ax in [self._ax_fit, self._ax_best, self._ax_cmp]:
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, alpha=0.25, color=BORDER)

        self._ax_fit.set_title("Historia Fitness", color=TEXT, fontsize=10, pad=5)
        self._ax_fit.set_xlabel("Generacja", color=MUTED, fontsize=8)
        self._ax_fit.set_ylabel("Odległość Hamminga", color=MUTED, fontsize=8)
        self._ax_best.set_title("Geny najlepszego osobnika", color=TEXT, fontsize=9, pad=4)
        self._ax_cmp.set_title("Cel  vs  Najlepszy", color=TEXT, fontsize=9, pad=4)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._top_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._ln_best, = self._ax_fit.plot([], [], color=GREEN,  lw=1.8, label="Best")
        self._ln_mean, = self._ax_fit.plot([], [], color=YELLOW, lw=1.2, ls="--", label="Mean", alpha=0.8)
        self._ax_fit.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, loc="upper right")
        self._canvas.draw()

    def _init_mpl_evo(self):
        ax = self._ax_heat
        self._evo_img = ax.imshow(np.zeros((1,1)), aspect="auto", origin="lower",
                                   cmap=EVO_CMAP, vmin=0, vmax=1, interpolation="nearest")
        self._evo_vl  = ax.axvline(x=0, color="white", lw=1.4, alpha=0.85, ls="--")
        self._evo_canvas.draw_idle()

    # ── GA logic ────────────────────────────────────────────────────────────

    def _build_cfg(self):
        lo, hi = self.s_blo.get(), self.s_bhi.get()
        if hi <= lo: hi = lo+1.
        v = sorted([self.s_ta.get(), self.s_tb.get(), self.s_tc.get()])
        return GAConfig(
            n_genes=max(2,int(self.s_genes.get())),
            pop_size=max(10,int(self.s_pop.get())),
            n_generations=max(10,int(self.s_gen.get())),
            mutation_rate=float(self.s_mut.get()),
            mutation_sigma=float(self.s_sigma.get()),
            crossover_rate=float(self.s_cross.get()),
            tournament_k=max(2,int(self.s_tour.get())),
            elitism_n=max(0,int(self.s_elite.get())),
            gene_bounds=(lo,hi),
            target_a=v[0], target_b=v[1], target_c=v[2],
        )

    def _on_start(self):
        if self._running: return
        cfg = self._build_cfg()
        self.ga = OFNGeneticAlgorithm(cfg)
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except: pass
        self._reset_plots()
        self._n_gen_total = cfg.n_generations
        self._refresh_every = max(1, int(self.s_refresh.get()))
        self._evo_matrix = None; self._evo_fit_max = 1.; self._evo_cursor = 0
        self._evo_sl.config(to=1); self._evo_sv.set(0)
        self._running = True
        self._btn_run.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._sb.upd("state", "▶ Działa...", GREEN)
        self._gen_lbl.config(text=f"GEN  0 / {cfg.n_generations}")
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _on_stop(self):
        self._running = False
        if self.ga: self.ga.stop()
        self._sb.upd("state","⏹ Zatrzymany", RED)
        self._btn_run.config(state="normal"); self._btn_stop.config(state="disabled")

    def _on_reset(self):
        self._on_stop(); self.ga = None
        self._reset_plots()
        self._evo_matrix = None; self._init_mpl_evo()
        for k,t,c in [("gen","GEN: —",MUTED),("best","BEST: —",GREEN),
                       ("mean","MEAN: —",YELLOW),("div","DIV: —",PURPLE),("speed","SPEED: —",MUTED)]:
            self._sb.upd(k,t,c)
        self._gen_lbl.config(text="GEN  0 / 0")
        self._evo_info.config(text="gen: —  |  best: —  |  mean: —")

    def _apply_target(self):
        if self.ga:
            v = sorted([self.s_ta.get(), self.s_tb.get(), self.s_tc.get()])
            self.ga.update_target(*v)

    def _worker(self):
        if not self.ga: return
        self.ga.initialize()
        cfg = self.ga.cfg; times = []
        for gen in range(1, cfg.n_generations+1):
            if not self._running: break
            t0 = time.perf_counter()
            stats = self.ga.step()
            times.append(time.perf_counter()-t0)
            if gen % self._refresh_every == 0 or gen == cfg.n_generations:
                stats.elapsed_ms = sum(times[-20:])/max(len(times[-20:]),1)*1000
                try: self._queue.put_nowait(stats)
                except queue.Full: pass
        self._running = False
        self.root.after(0, self._on_done)

    def _on_done(self):
        self._btn_run.config(state="normal"); self._btn_stop.config(state="disabled")
        self._sb.upd("state","✓ Zakończono", ACCENT)
        if self._last_stats: self._refresh_heatmap(self._last_stats)

    # ── polling ─────────────────────────────────────────────────────────────

    def _poll(self):
        updated = False
        while not self._queue.empty():
            try: s = self._queue.get_nowait(); updated = True; self._last_stats = s
            except queue.Empty: break
        if updated: self._update_display(self._last_stats)
        self.root.after(40, self._poll)

    # ── render ──────────────────────────────────────────────────────────────

    def _reset_plots(self):
        self._ln_best.set_data([],[]); self._ln_mean.set_data([],[])
        self._ax_fit.relim(); self._ax_fit.autoscale_view()
        for ax in [self._ax_best, self._ax_cmp]:
            ax.cla(); ax.set_facecolor(PANEL)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
            ax.tick_params(colors=MUTED, labelsize=8); ax.grid(True, alpha=0.25, color=BORDER)
        self._canvas.draw_idle()

    def _update_display(self, s: GAStats):
        gen = s.generation; n = self._n_gen_total
        self._gen_lbl.config(text=f"GEN  {gen} / {n}")
        self._sb.upd("gen",  f"GEN: {gen}/{n}")
        self._sb.upd("best", f"BEST: {s.best_fitness:.5f}", GREEN)
        self._sb.upd("mean", f"MEAN: {s.mean_fitness:.5f}", YELLOW)
        if self.ga: self._sb.upd("div", f"DIV: {self.ga.population_diversity():.3f}", PURPLE)
        self._sb.upd("speed", f"SPEED: {1000/max(s.elapsed_ms,.001):.0f} gen/s", MUTED)

        xs = list(range(1, len(s.history_best)+1))
        self._ln_best.set_data(xs, s.history_best)
        self._ln_mean.set_data(xs, s.history_mean)
        self._ax_fit.relim(); self._ax_fit.autoscale_view()
        self._draw_ofn_best(s); self._draw_ofn_cmp(s)
        self._canvas.draw_idle()
        self._refresh_heatmap(s)

    def _draw_ofn_best(self, s: GAStats):
        ax = self._ax_best; ax.cla(); ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8); ax.grid(True, alpha=0.2, color=BORDER)
        ax.set_title("Geny najlepszego osobnika", color=TEXT, fontsize=9, pad=4)
        ax.set_ylim(-0.05, 1.15)
        if s.best_individual is None or s.best_individual.size == 0: return
        y = np.linspace(0,1,180)
        cols = plt.cm.plasma(np.linspace(0.1,0.9,len(s.best_individual)))
        for p, c in zip(s.best_individual, cols):
            a,b,cc = p
            xu = a+y*(b-a); xd = cc-y*(cc-b)
            ax.fill(np.concatenate([xu,xd[::-1]]), np.concatenate([y,y[::-1]]), alpha=0.07, color=c)
            ax.plot(xu,y,color=c,lw=1.2,alpha=0.8)
            ax.plot(xd,y,color=c,lw=1.2,alpha=0.8,ls=":")

    def _draw_ofn_cmp(self, s: GAStats):
        ax = self._ax_cmp; ax.cla(); ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=8); ax.grid(True, alpha=0.2, color=BORDER)
        ax.set_title("Cel  vs  Najlepszy", color=TEXT, fontsize=9, pad=4)
        ax.set_ylim(-0.05,1.15)
        if not self.ga: return
        y = np.linspace(0,1,180)
        ta,tb,tc = self.ga.target[0]
        xu_t = ta+y*(tb-ta); xd_t = tc-y*(tc-tb)
        ax.fill(np.concatenate([xu_t,xd_t[::-1]]),np.concatenate([y,y[::-1]]),alpha=0.15,color=ACCENT)
        ax.plot(xu_t,y,color=ACCENT,lw=2.2,label="Cel"); ax.plot(xd_t,y,color=ACCENT,lw=2.2)
        if s.best_individual is not None and s.best_individual.size>0:
            av=s.best_individual.mean(axis=0); ba,bb,bc=av
            xu_b=ba+y*(bb-ba); xd_b=bc-y*(bc-bb)
            ax.fill(np.concatenate([xu_b,xd_b[::-1]]),np.concatenate([y,y[::-1]]),alpha=0.12,color=GREEN)
            ax.plot(xu_b,y,color=GREEN,lw=1.8,ls="--",label="Najlepszy"); ax.plot(xd_b,y,color=GREEN,lw=1.8,ls="--")
        ax.legend(facecolor=PANEL,edgecolor=BORDER,labelcolor=TEXT,fontsize=8)

    # ── heatmapa ewolucji ───────────────────────────────────────────────────

    def _refresh_heatmap(self, s: GAStats):
        if not self.ga or not s.fitness_snapshots: return
        mat = self.ga.get_fitness_matrix()        # (n_gens, pop_size)
        self._evo_matrix   = mat
        self._evo_fit_max  = max(s.fitness_global_max, 1e-9)
        n_g, n_p = mat.shape

        # normalizacja → [0,1]: 0=ideał (zielony), 1=najgorszy (czerwony)
        norm = np.clip(mat / self._evo_fit_max, 0, 1)   # (n_g, pop_size)

        # imshow: rows=osobnicy (oś Y), cols=generacje (oś X)
        self._evo_img.set_data(norm.T)                    # (n_p, n_g)
        self._evo_img.set_extent([-0.5, n_g-0.5, -0.5, n_p-0.5])
        self._ax_heat.set_xlim(-0.5, n_g-0.5)
        self._ax_heat.set_ylim(-0.5, n_p-0.5)
        self._ax_heat.set_ylabel(f"Rank osobnika (0=best, {n_p-1}=worst)", color=MUTED, fontsize=7)

        # kursor — śledź aktualną gen. jeśli scrubber na końcu
        at_end = (self._evo_cursor >= n_g-2) or (self._evo_cursor == 0 and n_g > 1)
        cursor = n_g-1 if at_end else self._evo_cursor
        self._evo_cursor = cursor
        self._evo_vl.set_xdata([cursor, cursor])

        # aktualizuj slider
        self._evo_sl.config(to=max(1, n_g-1))
        self._evo_sv.set(cursor)
        self._evo_gl.config(text=str(cursor))

        self._draw_profile(cursor)
        self._evo_canvas.draw_idle()

    def _draw_profile(self, gen_idx: int):
        """Poziomy wykres słupkowy fitness osobników dla wybranej generacji."""
        ax = self._ax_profile; ax.cla()
        ax.set_facecolor(DARK)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.tick_params(colors=MUTED, labelsize=6)
        ax.set_title(f"Gen {gen_idx}", color=TEXT, fontsize=7, pad=2)
        ax.set_xlabel("Fitness", color=MUTED, fontsize=6)
        ax.yaxis.set_visible(False)

        if self._evo_matrix is None or gen_idx >= len(self._evo_matrix): return
        fit = self._evo_matrix[gen_idx]   # już posortowane rosnąco
        n   = len(fit)
        norm = np.clip(fit / self._evo_fit_max, 0, 1)
        ax.barh(np.arange(n), fit, color=EVO_CMAP(norm), height=0.9, linewidth=0)
        ax.set_ylim(-0.5, n-0.5)
        if fit[-1] > 0: ax.set_xlim(0, fit[-1]*1.05)

        best, mean = float(fit[0]), float(fit.mean())
        self._evo_info.config(
            text=f"gen: {gen_idx}  |  best: {best:.4f}  |  mean: {mean:.4f}", fg=TEXT)

    # ── nawigacja scrubberem ────────────────────────────────────────────────

    def _evo_goto(self, gen: int):
        if self._evo_matrix is None: return
        n = len(self._evo_matrix)
        gen = max(0, min(gen, n-1))
        self._evo_cursor = gen
        self._evo_vl.set_xdata([gen, gen])
        self._evo_sv.set(gen); self._evo_gl.config(text=str(gen))
        self._draw_profile(gen)
        self._evo_canvas.draw_idle()

    def _evo_sl_moved(self, val):
        try: self._evo_goto(int(float(val)))
        except: pass

    def _evo_click(self, ev):
        if ev.inaxes is self._ax_heat and ev.xdata is not None:
            self._evo_goto(int(round(ev.xdata)))

    def _evo_drag(self, ev):
        if ev.button == 1 and ev.inaxes is self._ax_heat and ev.xdata is not None:
            self._evo_goto(int(round(ev.xdata)))

    def _evo_first(self): self._evo_goto(0)
    def _evo_last(self):
        if self._evo_matrix is not None: self._evo_goto(len(self._evo_matrix)-1)
    def _evo_prev(self): self._evo_goto(self._evo_cursor-1)
    def _evo_next(self): self._evo_goto(self._evo_cursor+1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    OFNGAApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
