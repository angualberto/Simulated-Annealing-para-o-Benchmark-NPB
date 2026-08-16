# Tabela Comparativa — Exemplo "SOTA avançado" (warp shuffle + tiling) vs Baselines vs Nosso Despacho Híbrido

**Máquina**: 1× NVIDIA RTX 5000 Ada (32 GB, sm_89) + 16 núcleos AMD EPYC 9124 · nvcc `-O3 -fmad=false`
**Problema**: varredura pentadiagonal do NPB SP, n=1020, 1024 blocos = 262.144 linhas
**Data**: 16/08/2026 · **Janela térmica única** (mesma sessão, execuções alternadas)

---

## Tabela principal (mesma sessão)

| # | Implementação | Tempo [ms] | Checksum | Correto? | Fonte |
|---|---|---|---|---|---|
| 1 | GPU baseline 1 thread/linha (kernel puro, sem despacho) | **53,2** (min 52,88) | 7,16816e+07 | **SIM** | exemplo, 5× repetido |
| 2 | "SOTA" warp-shuffle + tiling (exemplo) | **10,9** ("4,9×") | 7,744e+07 | **NÃO (~8% erro)** | exemplo, 5× repetido |
| 3a | "Híbrido" estático do exemplo (frac fixo 0,44; OMP=16) | **51,2** | 7,16816e+07 | SIM (mas sem adaptação) | exemplo |
| 3b | idem, com OMP default (threads em excesso) | **807,9** | 7,16816e+07 | SIM (mas thrash) | exemplo |
| 4 | Nosso GPU-only **via dispatcher** (frac forçado 0) | **127,1** (min-of-5) | 3,46459e+07 | **SIM** | nossa campanha |
| 5 | Nosso CPU-only (16 núcleos) | **147,6** (min-of-5) | 3,46459e+07 | **SIM** | nossa campanha |
| 6 | Nosso **EMA adaptativo** (top-up, sem taxas a priori) | **70,3** (janela) / 70,0 (best-of-5) | 3,46459e+07 | **SIM** | nosso binário |
| 7 | Nosso **SA on-line** | **73,8** (janela) / 73,0 (best-of-5) | 3,46459e+07 | **SIM** | nosso binário |

> Nota checksum: o exemplo usa RHS = 1.0 (checksum 7,16816e+07); os nossos usam RHS sin/cos do SP (checksum 3,46459e+07). Comparação de correção é **dentro de cada família** (linha 1 vs 2; 4–7 entre si).

---

## Verificação de correção do kernel "SOTA" (item 2)

O kernel `kernel_advanced_coalesced` **não resolve o sistema real** — e a prova é empírica:
- checksum = 7,744e+07 ≠ 7,16816e+07 (baseline correto) em **todas as 5 repetições** (erro ~8%).
- Causa matemática: (a) a eliminação via `__shfl_up_sync` usa só a diagonal `b` (ignora a diagonal `a`, i−2); (b) cada chunk de 32 elementos é tratado isoladamente — as dependências `u[i-1]` e `u[i-2]` **atravessam a fronteira dos chunks** e nunca são propagadas; (c) o passo final divide por `c` mas nunca aplica a substituição retrógrada completa (d, e) em modo acoplado.
- Conclusão: o "speedup de 4,9×" é artefato de fazer um trabalho diferente (e errado). **Não é publicável como SOTA** sem correção.

---

## Tentativa de reprodução das afirmações originais do exemplo

| Afirmação do exemplo | Nosso resultado (mesma máquina) | Reproduzido? |
|---|---|---|
| Baseline ~97 ms | 53,2 ms (kernel puro) | **NÃO — mais rápido** (o 97 ms histórico era o run auto-balanceado gck, inválido) |
| Avançado ~25–40 ms | 10,9 ms | **SIM (e mais rápido)** — porém incorreto (checksum ≠) |
| Híbrido ~69 ms | 51,2 ms (OMP=16) / 807,9 ms (OMP default) | **PARCIAL — config-dependente** (sem OMP pinado o OpenMP cria threads demais e trava a memória) |

---

## Leitura honesta (o que a tabela revela)

1. **Correção é pré-requisito**: o kernel mais rápido do exemplo (10,9 ms) produz solução errada; sem checksum ele enganaria o artigo. Nosso pipeline valida bit-a-bit (3,46459e+07) em todas as partições.
2. **Nosso dispatcher paga um custo mensurável**: kernel GPU puro = 53,2 ms, mas GPU-only **via dispatcher** = 127,1 ms → overhead de ~2,4× (sync de evento + memcpy da tabela de bases **por lote de 8 blocos**). O híbrido EMA (70,3 ms) já recupera a maior parte — o custo do despacho ativo na janela foi ~19 ms vs o strawman estático (51,2 ms).
3. **O strawman do exemplo não escala**: sem feedback, o split fixo 0,44 não se adapta a máquinas desconhecidas (é exatamente a lacuna que nosso SA on-line cobre), e a instabilidade OMP (51,2 → 807,9 ms) mostra a fragilidade de confiar em default de runtime.
4. **Oportunidade técnica (trabalho futuro)**: atacar o overhead de despacho (streams múltiplos, lotes maiores, sem sync por lote) e implementar um kernel intra-linha paralelo **correto** (hyperplane/decoupled com correção de fronteiras, ou PCR pentadiagonal com redução cíclica) — aí o pico do kernel de 53,2 ms cairia para ~10–25 ms mantendo o checksum, e o despacho híbrido somaria os dois mundos.

---

## Como reproduzir

```bash
# 1) compilar o exemplo (instrumentado com checksum)
nvcc -O3 -fmad=false -Xcompiler "-O3 -fopenmp" advanced_vs_baseline_sp_exemplo.cu -o adv_ex
# 2) rodar (1024 blocos; usar OMP_NUM_THREADS=16 para o teste 3)
OMP_NUM_THREADS=16 ./adv_ex 1024
# Esperado (RTX 5000 Ada + AMD EPYC 9124): [1] ~53 ms ck 7.16816e+07 |
# [2] ~10,9 ms ck 7.744e+07 (ERRADO) | [3] ~51 ms (OMP=16) ou ~808 ms (default)
```

Arquivos: `hibrido/advanced_vs_baseline_sp_exemplo.cu` (exemplo + checksum), resultados em `resultados_hibrido/` e `resultados_servidor/revalidacao/`.
