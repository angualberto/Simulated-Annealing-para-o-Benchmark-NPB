#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera as figuras do artigo (desempenho, telemetria, arquitetura)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 11})
OUT = "figs/"

# ---------------- Figura 1: modos de execucao ----------------
modos = ["Híbrido\n(SA auto)", "GPU\n(frac=0)", "CPU\n(frac=1)"]
tempo = [163.1, 184.9, 611.1]
flops = [182.1, 160.7, 48.6]
cores = ["#2c6fbb", "#bb8c2c", "#bb2c2c"]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
b1 = ax[0].bar(modos, tempo, color=cores, width=0.55)
for b, v in zip(b1, tempo):
    ax[0].text(b.get_x() + b.get_width()/2, v + 10, f"{v:.1f} ms", ha="center", fontsize=9)
ax[0].set_ylabel("Tempo por varredura (ms)")
ax[0].set_ylim(0, 700); ax[0].set_yticks([0, 200, 400, 600])
ax[0].set_title("(a) Tempo por varredura", fontsize=9.5)
ax[0].grid(axis="y", alpha=0.3)

b2 = ax[1].bar(modos, flops, color=cores, width=0.55)
for b, v in zip(b2, flops):
    ax[1].text(b.get_x() + b.get_width()/2, v + 6, f"{v:.1f}", ha="center", fontsize=9)
ax[1].set_ylabel("GFLOPS (28 flops/ponto)")
ax[1].set_ylim(0, 215); ax[1].set_yticks([0, 50, 100, 150, 200])
ax[1].set_title("(b) Vazão", fontsize=9.5)
ax[1].grid(axis="y", alpha=0.3)
fig.savefig(OUT + "fig_modos.png", dpi=300)
plt.close(fig)

# ---------------- Figura 2: evolucao do run (log_1T.txt) ----------------
iters, ts, temps = [], [], []
for line in open("../hibrido/log_1T.txt", encoding="utf-8"):
    m = re.search(r"\[Iter\s+(\d+)/1000\] t=([\d.]+) ms.*GPU=(\d+)C", line)
    if m:
        iters.append(int(m.group(1))); ts.append(float(m.group(2))); temps.append(int(m.group(3)))

fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
ax.plot(iters, ts, color="#2c6fbb", lw=1.4, label="Tempo por iteração")
ax.axhline(163.1, color="#bb2c2c", ls="--", lw=1, label="Média: 163,1 ms")
ax.set_xlabel("Iteração"); ax.set_ylabel("Tempo (ms)", color="#2c6fbb")
ax.tick_params(axis="y", labelcolor="#2c6fbb")
ax.set_ylim(120, 240)
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(iters, temps, color="#3f7d3f", lw=1.2, label="Temperatura GPU")
ax2.set_ylabel("Temperatura (°C)", color="#3f7d3f")
ax2.tick_params(axis="y", labelcolor="#3f7d3f")
ax2.axhline(84, color="#bb2c2c", ls=":", lw=1, label="Trava térmica (84 °C)")
ax2.set_ylim(30, 90)
ax.set_xlim(0, 1200); ax.set_xticks([0, 200, 400, 600, 800, 1000])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
          bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8.5, frameon=True)
fig.savefig(OUT + "fig_telemetria.png", dpi=300)
plt.close(fig)

# ---------------- Figura 3: arquitetura hibrida ----------------
# Layout em grade: setas horizontais nos centros (y) das caixas ou
# verticais exatas; cotovelos de CPU/GPU para a saída passam pelo
# corredor x em (5.9, 6.7), sem cruzar nenhuma caixa.
fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()
ax.set_xlim(0, 10); ax.set_ylim(0, 6.5); ax.axis("off")

def box(x, y, w, h, text, fc="#eaf2fb", ec="#2c6fbb"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9.5)

def seg(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#444444", lw=1.1,
            solid_capstyle="round")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, lw=1.1, color="#444444"))

def elbow(x1, y1, xv, y2):
    seg(x1, y1, xv, y1)
    arrow(xv, y1, xv, y2)

boxes = {
    "swap":    (0.3, 5.8, 2.4, 0.6, "Memória Swap (SSD)\nOut-of-Core Dinâmico", "#fcf8e3", "#bb8c2c"),
    "malha":   (0.3, 4.4, 2.4, 1.1, "Malha NPB SP\n1.061e12 pontos/iteração\n1.040.384 linhas (n=1020)", "#fdf3e3", "#bb8c2c"),
    "livre":   (0.3, 2.6, 2.4, 1.3, "Lista ``livre''\nblocos de 256 linhas\nPERIOD=8", "#f3e3fd", "#7c3fbb"),
    "decisor": (3.5, 3.0, 2.4, 2.4, "Decisor SA\nfração CPU/GPU por janela\nE(x) medido em tempo real\n(Metropolis + resfriamento)", "#eafbf0", "#3f7d3f"),
    "cpu":     (6.7, 4.4, 2.9, 1.3, "Pool de threads CPU\n(16 threads POSIX)\nsolve serial bit-idêntico", "#fdeaea", "#bb2c2c"),
    "gpu":     (6.7, 2.4, 2.9, 1.4, "GPU RTX 5000 Ada\nmega-lotes de 128 blocos\nsolve_lines_kernel", "#eaf2fb", "#2c6fbb"),
    "tele":    (6.7, 0.7, 2.9, 1.1, "Telemetria NVML\ntemperatura / consumo\n(limite 84 °C)", "#f5f5f5", "#666666"),
    "saida":   (0.3, 0.7, 6.2, 1.0, "Saída h_out + checksum bit-idêntico (1,88943960e+08)", "#eafbf0", "#3f7d3f"),
}
for b in boxes.values():
    box(*b)

# fluxo malha <-> swap (mão dupla explícita, setas distintas)
arrow(1.3, boxes["malha"][1] + boxes["malha"][3], 1.3, boxes["swap"][1]) # malha -> swap
arrow(1.7, boxes["swap"][1], 1.7, boxes["malha"][1] + boxes["malha"][3]) # swap -> malha

# malha -> livre
arrow(1.5, boxes["malha"][1], 1.5, boxes["livre"][1] + boxes["livre"][3])

# malha -> decisor e livre -> decisor (horizontais, entram na borda esquerda)
arrow(2.7, 4.95, 3.5, 4.95)
arrow(2.7, 3.25, 3.5, 3.25)
# decisor -> CPU e decisor -> GPU (horizontais nos y exatos das caixas)
arrow(5.9, 5.05, 6.7, 5.05)
arrow(5.9, 3.1, 6.7, 3.1)
# CPU -> saída e GPU -> saída (cotovelos pelo corredor entre decisor e GPU)
elbow(6.7, 4.6, 6.2, 1.7)
elbow(6.7, 2.6, 6.45, 1.7)
# GPU -> telemetria (vertical no centro)
arrow(8.15, 2.4, 8.15, 1.8)
fig.savefig(OUT + "fig_arquitetura.png", dpi=300)
plt.close(fig)

print("Figuras geradas:", "OK")

# ---------------- Figura 4: Comparativo 5 Metodos (Mega-lote) ----------------
import csv
csv_path = "../hibrido/comparativo_megalote_5metodos.csv"
try:
    metodos = []
    tempos = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metodo = row["metodo"]
            # Shorten names for the plot
            if "Somente-GPU" in metodo: name = "GPU\n(100%)"
            elif "Somente-CPU" in metodo: name = "CPU\n(100%)"
            elif "Estatico" in metodo: name = "Estático\n(50/50)"
            elif "Reativo" in metodo: name = "EMA\n(Reativo)"
            elif "SA" in metodo: name = "SA\n(Proposto)"
            else: name = metodo
            metodos.append(name)
            tempos.append(float(row["tempo_min_ms"]))
            
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    
    cores_5 = []
    for m in metodos:
        if "SA" in m: cores_5.append("#bb8c2c")  # Destaque dourado
        elif "GPU" in m: cores_5.append("#2c6fbb") # Azul
        elif "CPU" in m: cores_5.append("#bb2c2c") # Vermelho
        else: cores_5.append("#777777") # Cinza
        
    b5 = ax.bar(metodos, tempos, color=cores_5, width=0.6)
    for b, v in zip(b5, tempos):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.2f} ms", ha="center", fontsize=9.5, fontweight="bold" if v < 40 else "normal")
        
    ax.set_ylabel("Tempo mínimo por varredura (ms)")
    ax.set_ylim(0, max(tempos) * 1.15)
    ax.set_title("Desempenho dos Métodos de Balanceamento no Mega-lote (Menor é Melhor)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    
    # Custom legend
    import matplotlib.patches as mpatches
    sa_patch = mpatches.Patch(color='#bb8c2c', label='SA Adaptativo (Mais rápido)')
    ax.legend(handles=[sa_patch], loc='upper left')

    fig.savefig(OUT + "fig_5metodos.png", dpi=300)
    plt.close(fig)
    print("Figura 5 métodos gerada: OK")
except Exception as e:
    print(f"Não foi possível gerar fig_5metodos.png: {e}")
