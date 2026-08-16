#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolida os resultados de hpc/resultados_hibrido e gera as figuras do artigo."""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 11})
ROOT = "/home/andre/Documentos/hpc/resultados_hibrido"
OUT = "scripts/artigo/figs/"

def carregar(*globs):
    out = []
    for g in globs:
        for f in glob.glob(os.path.join(ROOT, g)):
            try:
                d = json.load(open(f, encoding="utf-8"))
                d["_arquivo"] = os.path.basename(f)
                out.append(d)
            except Exception:
                pass
    return out

inter = carregar("resultados_servidor/revalidacao/intercalado/*.json")
base  = carregar("resultados_servidor/revalidacao/baselines/*.json")
sw2   = carregar("resultados_servidor/revalidacao/sweep2/*.json")

ema, sa = {}, {}
for j in inter:
    t = j.get("tempo_total_ms")
    nb = j.get("estrutura", {}).get("blocos", 0)
    if "controlador" not in j and t:          # e*.json = EMA
        ema.setdefault(nb, []).append(t)
    elif "controlador" in j and t:            # s*.json = SA
        sa.setdefault(nb, []).append(t)

bas = {}
for j in base:
    ff = j.get("controlador", {}).get("frac_fixo")
    t = j.get("tempo_total_ms")
    if ff is not None and t:
        bas.setdefault(float(ff), []).append(t)

sweep = {}
for j in sw2:
    ff = j.get("controlador", {}).get("frac_fixo")
    t = j.get("tempo_total_ms")
    if ff is not None and t:
        sweep.setdefault(float(ff), []).append(t)

print("== CAMPANHA EMA vs SA (min / JDmedia) ==")
for nb in sorted(set(ema) | set(sa)):
    e = ema.get(nb, []); s = sa.get(nb, [])
    print(f"  blocos={nb:5d}: EMA min={min(e) if e else 0:7.2f} ({len(e)} am) | SA min={min(s) if s else 0:7.2f} ({len(s)} am)")
print("== BASELINES forcados (min) ==")
for ff in sorted(bas):
    print(f"  frac={ff}: min={min(bas[ff]):7.2f} ({len(bas[ff])} am)")
print("== SWEEP2 (min) ==")
for ff in sorted(sweep):
    print(f"  frac={ff}: min={min(sweep[ff]):7.2f} ({len(sweep[ff])} am)")

# ================= FIGURAS (numeros oficiais do relatorio, min-of-5) =============
azul, verde, vermelho, cinza = "#2c6fbb", "#3f7d3f", "#bb2c2c", "#888888"

# --- FIG 1: campanha + sweep ---
fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True)
escalas = ["64", "512", "1024"]
ema_v = [6.66, 36.16, 70.00]; sa_v = [6.59, 36.73, 72.98]
x = range(3); w = 0.35
ax[0].bar([i - w/2 for i in x], ema_v, w, label="EMA", color=cinza)
ax[0].bar([i + w/2 for i in x], sa_v, w, label="SA", color=azul)
for i, (e, s) in enumerate(zip(ema_v, sa_v)):
    ax[0].text(i - w/2, e + 1.8, f"{e:.2f}", ha="center", fontsize=7.5)
    ax[0].text(i + w/2, s + 1.8, f"{s:.2f}", ha="center", fontsize=7.5)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(escalas)
ax[0].set_xlabel("Número de blocos (256 linhas)")
ax[0].set_ylabel("Tempo [ms]")
ax[0].set_title("(a) EMA vs SA (melhor de 5)", fontsize=9.5)
ax[0].legend(fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18)); ax[0].grid(axis="y", alpha=0.3)

fracs = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
tv = [127.12, 86.91, 72.49, 71.33, 73.12, 71.91, 147.63]
ax[1].plot(fracs, tv, "o-", color=azul, lw=1.5)
ax[1].axhspan(71, 74, color="#3f7d3f", alpha=0.12)
ax[1].axhline(72.98, color=vermelho, ls="--", lw=1, label="SA auto (72,98 ms)")
ax[1].set_xticks(fracs); ax[1].set_xlim(-0.08, 1.08); ax[1].set_ylim(50, 155)
ax[1].set_xlabel("Fração forçada ($HSP\\_FIXED\\_FRAC$)")
ax[1].set_ylabel("Tempo [ms]")
ax[1].set_title("(b) Varredura de fração forçada", fontsize=9.5)
ax[1].legend(fontsize=8.5, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.18)); ax[1].grid(alpha=0.3)
fig.savefig(OUT + "fig_campanhas.png", dpi=300); plt.close(fig)

# --- FIG 2: mega-lote + cuPentBatch ---
fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True)
modos = ["GPU-only", "CPU-only", "frac 0,5", "EMA", "SA"]
antes = [127.12, 147.63, 71.20, 70.00, 72.98]
mega  = [47.30, 147.87, 51.52, 47.12, 42.44]
x = range(5); w = 0.35
ax[0].bar([i - w/2 for i in x], antes, w, label="BATCH=8 (antes)", color=cinza)
ax[0].bar([i + w/2 for i in x], mega,  w, label="MEGA=128", color=azul)
for i, a, m in zip(x, antes, mega):
    ax[0].text(i - w/2, a + 2, f"{a:.0f}", ha="center", fontsize=6.8)
    ax[0].text(i + w/2, m + 2, f"{m:.0f}", ha="center", fontsize=6.8)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(modos, fontsize=8)
ax[0].set_ylabel("Tempo [ms]")
ax[0].set_title("(a) Mega-lote: 1 launch/128 blocos", fontsize=9.5)
ax[0].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20)); ax[0].grid(axis="y", alpha=0.3)

names = ["Produção\n(LU)", "cuPentBatch\n(p.Solve)", "cuPentBatch\n(p.Factor)", "cuPentBatch\n(F+S)"]
kv = [36.23, 38.31, 40.24, 78.50]
b = ax[1].bar(names, kv, color=[azul, cinza, cinza, vermelho], width=0.5)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names, fontsize=5.5)
for bb, v in zip(b, kv):
    ax[1].text(bb.get_x() + bb.get_width()/2, v + 1, f"{v:.2f}", ha="center", fontsize=8)
ax[1].set_ylim(0, 95)
ax[1].annotate("1,05×", xy=(1.5, 38.31), xytext=(1.5, 51), ha="center", fontsize=10, color=azul,
               arrowprops=dict(arrowstyle="-", lw=0.8, color=azul))
ax[1].annotate("2,17×", xy=(3.5, 78.5), xytext=(3.5, 90), ha="center", fontsize=10, color=vermelho,
               arrowprops=dict(arrowstyle="-", lw=0.8, color=vermelho))
ax[1].set_ylabel("Tempo de kernel [ms]");
ax[1].set_title("(b) 262.144 linhas, n=1020", fontsize=9.5)
ax[1].grid(axis="y", alpha=0.3)
fig.savefig(OUT + "fig_megabatch.png", dpi=300); plt.close(fig)

# --- FIG 3: projecao classes + estado da arte ---
fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True)
classes = ["C (162³)", "D (408³)", "E (1020³)"]
proj_e = [4.2, 66.6, 416.5]; proj_s = [4.4, 70.0, 437.5]
x = range(3); w = 0.35
ax[0].bar([i - w/2 for i in x], proj_e, w, label="EMA", color=cinza)
ax[0].bar([i + w/2 for i in x], proj_s, w, label="SA", color=azul)
for i, (e, s) in enumerate(zip(proj_e, proj_s)):
    ax[0].text(i - w/2, e + 8, f"{e}", ha="center", fontsize=7)
    ax[0].text(i + w/2, s + 8, f"{s}", ha="center", fontsize=7)
ax[0].set_ylim(0, 650); ax[0].set_yticks([0, 200, 400, 600]); ax[0].set_xticks(list(x)); ax[0].set_xticklabels(classes)
ax[0].set_ylabel("Tempo projetado [s]")
ax[0].set_title("(a) Projeção penta C/D/E", fontsize=9.5)
ax[0].legend(fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18)); ax[0].grid(axis="y", alpha=0.3)

sota = ["Ser. 1 core\n(E5-1660v2)", "6 cores\n(Xeon)", "C2070\n(GPU)", "K40\n(GPU)", "Titan\n(GPU)", "Este estudo\n(fase penta)*"]
sv = [483.24, 122.25, 106.3, 31.26, 29.3, 4.2]
b = ax[1].bar(range(len(sv)), sv, color=[cinza]*5 + [azul], width=0.6)
for bb, v in zip(b, sv):
    ax[1].text(bb.get_x() + bb.get_width()/2, v * 1.06, f"{v:.0f}", ha="center", fontsize=8)
ax[1].set_yscale("log"); ax[1].set_ylim(1, 650); ax[1].set_yticks([1, 10, 100, 500])
ax[1].set_xticks(range(len(sv))); ax[1].set_xticklabels(sota, fontsize=6.5)
ax[1].set_ylabel("Tempo [s] (log)")
ax[1].set_title("(b) SP C: publicado vs. projeção*", fontsize=9.5)
ax[1].grid(axis="y", alpha=0.3)
fig.savefig(OUT + "fig_classes.png", dpi=300); plt.close(fig)

print("FIGURAS OK")