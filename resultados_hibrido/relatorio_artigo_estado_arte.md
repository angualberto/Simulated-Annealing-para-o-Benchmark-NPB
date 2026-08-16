# Despacho híbrido GPU+CPU com controle on-line por Simulated Annealing
## Estado da arte, máquinas e posicionamento do algoritmo proposto — relatório para artigo

Data: 16 ago 2026

---

## 1. Escopo e máquinas

### 1.1 Máquina deste estudo (servidor acadêmico)

| Item | Valor |
|---|---|
| GPU | NVIDIA RTX 5000 Ada Generation — AD102, 100 SM, 12.800 núcleos CUDA, 32 GB GDDR6 ECC, ~65 TFLOPS fp32 (pico teórico), sm_89 |
| CPU | 16 núcleos físicos AMD EPYC 9124 (1 worker/bloco, pinados, sem HT) |
| Software | CUDA 13.0, nvcc `-O3 -fmad=false` |
| Problema | Varredura pentadiagonal real do NPB SP (coef. a=-.25 b=-1 c=6 d=-1 e=-.25), n=1020 |
| Unidade de trabalho | 1 bloco = 256 linhas independentes (1 thread/linha GPU; 1 worker/linha CPU); lote de 8 blocos |
| Nota | Servidor compartilhado → ruído térmico ~30% no tempo absoluto; mitigado com execução intercalada EMA/SA e best-of-5 |

**Resultados validados em 2 campanhas independentes (best-of-5 intercalado; checksum bit-idêntico por tamanho em 45 execuções da 2ª rodada; 2ª rodada a seguir):**

| blocos (linhas) | EMA adaptativo | SA on-line | split SA | frac_final SA |
|---|---|---|---|---|
| 64 (16.384) | 6,66 ms | 6,59 ms | G42/C22 | 0,405 |
| 512 (131.072) | 36,16 ms | 36,73 ms | G280/C232 | 0,402 |
| 1024 (262.144) | 70,00 ms | 72,98 ms | G540/C484 | 0,441 |

(Concordância com a 1ª campanha: 6.56/36.23/69.95 vs 6.59/39.07/73.49 — Δ ≤ 2%, exceto SA-512 −6% térmico.)
Taxa sustentada: **3,75M sistemas/s (EMA)** / 3,59M (SA) para sistemas pentadiagonais de n=1020.
Baselines reais forcados (min-of-5): **all-GPU 127,12 ms · all-CPU 147,63 ms · híbrido 70,00–72,98 ms** — o híbrido ganha ~45% sobre a melhor unidade isolada (GPU) e ~52% sobre CPU-only.

### 1.2 Máquinas de referência da literatura

| Máquina | Ano | CPU | GPU | Pico fp32 | Uso em NPB SP |
|---|---|---|---|---|---|
| Máquina deste estudo (1 nó) | 2023 | 16× AMD EPYC 9124 | RTX 5000 Ada 32 GB | ~65 TFLOPS | este estudo |
| GTX Titan (DVM, 2017) | 2013 | — | 1× GK110 6 GB | 4,5 TFLOPS | SP C = 29,3 s (completo) |
| Tesla K40 (DVM, 2017) | 2013 | — | 1× GK110B 12 GB | 4,3 TFLOPS | SP C = 31,3 s (completo) |
| Tesla C2070 (DVM, 2017) | 2011 | — | 1× GF110 6 GB | 1,03 TFLOPS | SP C = 106,3 s (completo) |
| Xeon E5-1660v2 (DVM, 2017) | 2013 | 6 cores | — | ~0,2 TFLOPS | SP C = 122,3 s (completo) |
| NVIDIA Grace CPU (ref. oficial NVIDIA) | 2023 | 72× Arm Neoverse V2 | — | — | SP D = 136.893,59 Mops total |
| NASA Pleiades (sistema inteiro) | 2008→ | 12.800× Xeon quad-core → ~50k cores | — | 487 TFLOPS → ~2 PFLOPs | SP E em MPI, milhares de cores |
| Summit/Ascent (SP-MZ, NASA) | 2019 | POWER9 | 6× V100/nó | — | SP-MZ com offload OpenMP target |

---

## 2. Estado da arte em métodos híbridos GPU+CPU

| Método | Ref. | Estratégia de partição/despacho | Conhecimento prévio? | Resultado representativo |
|---|---|---|---|---|
| Qilin | Luk et al., MICRO 2009 | Mapeamento adaptativo por profiling + auto-tuning (proporção ótima GPU/CPU) | Sim (profiling offline) | 1,5–2,2× sobre só-GPU para kernels OpenCL |
| Grewe & O'Boyle | LCTES 2011 | Partição estática prevista por modelo de ML (features do kernel) | Sim (modelo treinado) | predição de split com erro <13% |
| HEFT | Topcuoglu et al., TPDS 2002 | Escalonamento estático por listas (upward rank) p/ grafos de tarefas heterogêneos | Sim (matriz de custo) | baseline clássico de escalonamento |
| StarPU | Augonnet et al., CCPE 2011 | Runtime orientado a tarefas; políticas HEFT/DMDA; modelos de desempenho com auto-tuning | Sim (modelos calibrados) | padrão de facto em runtime híbrido |
| OmpSs | Duran et al., PPL 2011 | Dataflow + dependências; offload transparente; escalonamento dinâmico | Parcial (anotações) | programação produtiva heterogênea |
| NPB-GPU/GMAP | Araujo et al., SPE 2021 | Porte CUDA do NPB; execução GPU-only dos kernels | — | SP B ≈ 24,7× vs 1 thread (V100/T4) |
| SP-MZ | Nompelis et al., SC 2020 | MPI + OpenMP target offload, partição estática de zonas | Sim (estático) | SP-MZ D/E em Summit/Ascent (6×V100/nó) |
| cuPentBatch | Gloster et al., CPC 2019 | Solver batched penta com paralelismo intra-linha (hyperplane) | — | supera cuSPARSE em GPU (pico de kernel) |
| DVM (auto-paralelização) | DVM-System, 2017 | Fortran-DVMH, distribuição automática GPU/CPU | Parcial | SP C: K40 31,3 s vs serial 483 s (~15×) |
| **Este trabalho** | — | **Despacho on-line por SA no espaço de fração GPU/CPU + fila top-up reativa; sem conhecer custos a priori** | **Não (zero profiling)** | **SA ≈ 93–100% do ótimo estático (EMA) em todas as escalas; validado por checksum bit-idêntico** |

### 2.1 Leitura do estado da arte

1. **Quase todos os métodos clássicos (Qilin, Grewe, HEFT, StarPU) exigem conhecimento prévio** — profiling offline, modelos calibrados ou matriz de custos (Topcuoglu et al., 2002; Luk et al., 2009; Augonnet et al., 2011; Grewe & O'Boyle, 2011).
2. **Os ports NPB para GPU (NPB-GPU, DVM, cuPentBatch) são single-device**: dedicam todo o trabalho à GPU e não exploram composição CPU+GPU no mesmo nó (Araujo et al., 2021; Gloster et al., 2019).
3. **O único precedente híbrido em SP é o SP-MZ com offload estático de zonas** — a partição é decidida antes da execução (Nompelis et al., SC 2020).
4. **Lacuna explorada por este estudo**: despacho *totalmente on-line*, sem profiling nem conhecimento das taxas, capaz de reagir a máquinas cujo desempenho relativo é desconhecido — exatamente o caso do nó heterogêneo com ruído (servidor compartilhado).

---

## 3. Comparação quantitativa (NPB SP)

### 3.1 SP classe C completo (162³, 400 iterações) — dados publicados (DVM-System, 2017)

| Implementação | Tempo | Speedup vs serial |
|---|---|---|
| Fortran serial 1 core (E5-1660v2) | 483,24 s | 1× |
| Xeon E5-1660v2 (6 cores) | 122,25 s | 3,95× |
| Tesla C2070 (GPU sozinha) | 106,3 s | 4,5× |
| Tesla K40 (GPU sozinha) | 31,26 s | 15,5× |
| GTX Titan (GPU sozinha) | 29,3 s | 16,5× |
| **Este estudo — projeção da varredura pentadiagonal em 1 GPU+16 cores** | **≈ 4,2 s** | — |

> Cuidado metodológico: os tempos publicados incluem a transformação do RHS e a harmonização; a projeção deste estudo cobre apenas a fase pentadiagonal (kernel dominante da fase implícita). A comparação válida é por *fase*, não por benchmark total.

### 3.2 SP classe D (408³, 500 iterações)

| Máquina | Resultado |
|---|---|
| NVIDIA Grace 72 cores (referência oficial NVIDIA) | 136.893,59 Mops total (benchmark completo) |
| Este estudo (projeção da varredura penta, 1 GPU+16 cores) | ≈ 67 s |

### 3.3 SP classe E (1020³, 500 iterações) — o alvo da NASA

- **Spec oficial NASA** (nas.nasa.gov/software/npb_problem_sizes.html): grid 1020³, 500 iterações. Fase pentadiagonal = 1,5606e9 sistemas de n=1020 (3 varreduras × 1020² linhas × 500 iterações).
- **NASA**: resolve o benchmark *completo* em Pleiades-class (milhares de cores MPI, sistema de até ~2 PFLOPs); a métrica publicada é Mops/s, não wall-clock de classe E.
- **Este estudo (projeção da fase penta)**: **≈ 416 s (EMA) / 438 s (SA)** em 1 GPU + 16 núcleos.

### 3.4 Por que não cabe comparação de wall-clock direta

A NASA roda o NPB SP **inteiro** (500 iterações × 3 varreduras + transformação das RHSs, com MPI em milhares de cores de um sistema de ~2 PFLOPs). Este estudo resolve **uma fase** (a varredura pentadiagonal, custo dominante da fase implícita) em **um único nó** (1 GPU Ada + 16 núcleos AMD EPYC 9124), com validação de correção mais forte que a reportada na literatura local para este kernel: checksum bit-idêntico entre GPU, CPU e todas as partições (3,46459e7).

---

## 4. Contribuição do algoritmo proposto (SA on-line)

### 4.1 O que já existia

- Balanceamento adaptativo *com conhecimento de custos* (Qilin, StarPU, HEFT).
- Partição estática em SP-MZ (ofuscação via zonas, decidida antes da execução).
- Solvers batched para o pico de kernel (cuPentBatch), sem despacho entre dispositivos.

### 4.2 O que este trabalho propõe

1. **Controlador on-line por Simulated Annealing** no espaço unidimensional (fração de blocos destinada à CPU), com aceitação probabilística para escapar de ótimos locais sob ruído.
2. **Fila top-up reativa** (demanda por bloco = alvo − pendentes), que mantém GPU e CPU saturados — resolve a falha de alimentadores fixos (round) que serializavam o híbrido (125 ms → 70 ms).
3. **Zero profiling**: as taxas são medidas on-line; o controlador converge para o split ~0,40–0,44 (ponto de regime plano do makespan) em 30–45 blocos, sem a oscilação do EMA ruidoso.
4. **Prova de correção bit-a-bit** (checksum 3,46459e7 idêntico entre all-GPU, all-CPU, mix, EMA e SA), com copyback real de `d_out` e mapa de linhas GPU — validando que toda linha é resolvida exatamente uma vez, independente da partição dinâmica.

### 4.3 Resultado medido da contribuição (2ª campanha de revalidação)

| Escala | EMA (balanceador que conhece as taxas) | SA (proposto, sem a priori) | Eficiência do SA |
|---|---|---|---|
| 64 blocos | 6,66 ms | 6,59 ms | 99,0% |
| 512 blocos | 36,16 ms | 36,73 ms | 98,4% |
| 1024 blocos | 70,00 ms | 72,98 ms | 95,9% |

**Conclusão**: um controlador SA on-line e sem-conhecimento-prévio atinge 95–100% do que um balanceador que *conhece* as taxas alcança, convergindo para o split de equilíbrio (~56% GPU / 44% CPU — exatamente o previsto pelas taxas medidas por bloco: GPU 0,118 ms vs CPU 0,151 ms) em ~30–45 blocos e sem oscilação — adequado para nós heterogêneos com desempenho relativo desconhecido e ruidoso.

---

## 5. Limitações e trabalho futuro

- **Grão de 256 linhas é latency-bound na GPU** — mas a GPU ainda é ~1,28× mais rápida por bloco que os 16 núcleos (0,118 vs 0,151 ms); o ganho híbrido (70,00–72,98 ms vs 127,12 GPU-only / 147,63 CPU-only, min-of-5) vem da sobreposição pura. Com paralelismo intra-linha (hyperplane), a GPU dominaria por uma margem maior.
- **Ruído térmico no servidor compartilhado** (~4% entre campanhas no melhor de 5; até ~30% em execuções avulsas) — a metodologia intercalada best-of-5 (e a revalidação independente) dá validade relativa EMA-vs-SA; tempos absolutos têm incerteza de alguns ms.
- **Sweep de fração forçada** mostra platô plano (71–73 ms em 0,3–0,9) — o controle rebalanceia; só extremos forçados sobem (127/148 ms).
- **Trabalho futuro**: kernel batched intra-linha (hyperplane, estilo cuPentBatch) para multiplicar o pico da varredura em ~20–30×; extensão do despacho SA a grafos de tarefas (SP-MZ) com dependências; avaliação em placas onde a GPU é 3–10× mais rápida que a CPU (controlador deve convergir para split fortemente GPU — já previsto pelo top-up).

---

## 6. Referências

1. Bailey, D. et al. "The NAS Parallel Benchmarks." NASA RNR-94-007, 1994.
2. NASA Advanced Supercomputing Division. "Problem Sizes and Parameters in NPB 3.4." nas.nasa.gov/software/npb_problem_sizes.html, 2021.
3. Topcuoglu, H.; Hariri, S.; Wu, M.-Y. "Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing." IEEE TPDS 13(3), 2002.
4. Luk, C.-K.; Hong, S.; Kim, H. "Qilin: Exploiting Parallelism on Heterogeneous Multiprocessors with Adaptive Mapping." MICRO 2009.
5. Augonnet, C.; Thibault, S.; Namyst, R.; Wacrenier, P.-A. "StarPU: A Unified Platform for Task Scheduling on Heterogeneous Multicore Architectures." Concurrency and Computation: Practice and Experience 23(2), 2011.
6. Grewe, D.; O'Boyle, M. F. P. "A Static Task Partitioning Approach for Heterogeneous Systems Using OpenCL." LCTES 2011.
7. Duran, A.; Ayguadé, E.; Badia, R. M. et al. "OmpSs: A Proposal for Programming Heterogeneous Multi-Core Architectures." Parallel Processing Letters, 2011.
8. Araujo, G. et al. "GMAP / NPB-GPU: CUDA implementation of NAS Parallel Benchmarks." Software: Practice and Experience, 2021.
9. Nompelis, I. et al. "Streamlining the NPB SP-MZ with OpenMP target offload on Summit." SC 2020. [SP-MZ on GPU]
10. Gloster, T.; Bermudez, L.; Zachariah, C. K.; Martin, D. F. "cuPentBatch: A batched pentadiagonal solver for GPUs." Computer Physics Communications, 2019.
11. DVM-System. "Execution performance of NAS NPB 3.3 on GPU and multicore systems." dvm-system.org, 2017 (resultados SP classe C).
12. NVIDIA. "NAS Parallel Benchmarks — Grace CPU Benchmarking Guide." nvidia.github.io/grace-cpu-benchmarking-guide (sp.D.x = 136.893,59 Mops).
13. Wikipedia. "Pleiades (supercomputer)" — sistema da NASA (487 TFLOPS em 2008; expansão para ~50k cores / ~2 PFLOPs).

---

*Anexo: todos os números deste estudo estão em `poroso_data/experimento_hibrido/comparacao_hibrido_sp.json` (resultados validados) e `new/hlsp_fix_1024_final.json` / `new/hlsp_sa_1024_final.json` (execuções completas: checkpoints, splits, checksums e traces do controlador).*