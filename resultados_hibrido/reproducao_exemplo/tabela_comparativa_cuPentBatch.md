# Comparacao: cuPentBatch (Gloster, O'Naraigh & Pang, CPC 2019) vs Este Trabalho

- **cuPentBatch** — arXiv:1807.07382 (v3), github.com/munstermonster/cuPentBatch; Titan X Pascal (12 GB), i7-6850K (6 HT), CUDA 9.2, doubles, kernel-only.
- **Este trabalho** — despacho hibrido GPU+CPU on-line (EMA adaptativo + SA), NPB SP pentadiagonal, n=1020; RTX 5000 Ada (sm_89, CUDA 13.0) + 16 nucleos AMD EPYC 9124; `nvcc -O3 -fmad=false`.

## Tabela 1 - Visao geral

| Criterio | cuPentBatch | Este trabalho |
|---|---|---|
| Escopo | Biblioteca de solver batched pentadiagonal em GPU (1 dispositivo) | Runtime de despacho on-line hibrido GPU+CPU |
| Algoritmo | Thomas estendido (LU sem pivoteamento), O(N)/sistema, 1 thread/sistema, varredura serial | Mesmo esquema: `factor_penta` (CPU, 1x) + solve (kernel), 1 thread/linha, serial |
| Layout memoria | Interleaved (coalescido: warp le elementos consecutivos de m sistemas) | `gpu_base` por bloco (nao coalescido, stride N*8 B) |
| Matriz constante | `cuPentBatchConstant`: factoriza A 1x, resolve m RHSs | idem (coefs do SP constantes, factor 1x) |
| Dispositivos | GPU somente | GPU + CPU co-agendados |
| Controle/adaptacao | Nenhum (fila estatica) | EMA adaptativo + SA on-line com top-up reativo |
| Validacao | L2 vs solucao exata da hyperdiffusion (erro ~ N^-2) | Checksum bit-identico GPU==CPU==particoes |
| Aplicacao-alvo | Cahn-Hilliard / hyperdiffusion / estudos parametricos | Varredura pentadiagonal do NPB SP (classe E) |
| N por sistema | ate O(10^4) | 1020 |

## Tabela 2 - Resultados relatados vs equivalentes medidos

| Metrica | cuPentBatch (Titan X Pascal) | Nosso (RTX 5000 Ada) |
|---|---|---|
| vs `gpsvInterleavedBatch` (cuSPARSE, QR+Householder) | 1,2-2,0x+ (Rewrite 1,2-1,3x em lotes 10^4-10^5) | nao usado (sem cuSPARSE) |
| vs serial 1 core | 10-20x (batch > 2048) | nao medido (16 nucleos sao a base) |
| vs OpenMP 8 threads | 5-6x (GPU vs 8 threads) | GPU-only 127,1 ms vs CPU-only 147,6 ms -> 1,16x (16 threads) |
| Ganho hibrido | NAO EXISTE (single-device) | 70,0-73,0 ms vs GPU 127,1 (-45%) e CPU 147,6 (-52%) |
| Validacao numerica | erro L2 N^-2 | checksum bit-identico (3,46459e+07), 45 execucoes |

> Hardware/era: Titan X Pascal fp64 = 1/32 (~0,35 TFLOPS) vs Ada (~2 TFLOPS fp64); eles medem kernel-only (250 passos, sem PCIe, sem despacho).

## Tabela 3 - Complementaridade

**cuPentBatch ataca o pico do kernel:**
- layout interleaved -> acesso coalescido a DRAM;
- fatoracao unica p/ matriz constante (so o RHS muda);
- 1 thread/sistema O(N) serial — mesmo custo algoritmico que o nosso.

**Este trabalho ataca o agendamento on-line:**
- decidir quem resolve cada bloco (GPU vs CPU) sem taxas a priori;
- top-up reativo mantem ambos saturados (fila, nao round-robin);
- SA on-line converge ~0,44 (equilibrio de fila previsto pelas taxas);
- prova de correcao bit-a-bit entre GPU e CPU (cuPentBatch nao tem o conceito).

**Lacuna identificada (proxima etapa do nosso plano):**
nosso kernel atual (1 thread/linha descoalescido) roda ~53,2 ms (262.144 linhas, kernel puro) / 127,1 ms via dispatcher. Kernel estilo cuPentBatch (interleaved + factor 1x) no mesmo tamanho deve cair para ~10-25 ms mantendo o checksum; encaixado no dispatcher, o sistema soma pico de kernel batched (GPU) + reserva CPU adaptativa.

## Tabela 4 - Caveats e diferencas honestas

1. **Estabilidade**: ambos usam LU sem pivoteamento. cuPentBatch vale para SPD/diag. dominantes; o SP e estritamente diag. dominante (|c|=6 > 2,5), logo LU sem pivoteamento e seguro. cuPentBatch admite QR (gpsv) como mais estavel a priori.
2. **O que medem**: kernel-only (sem transferencia, sem dispatcher). Nos medimos o sistema completo (despacho + sync por lote + checksum) — por isso GPU-only 127,1 ms vs kernel puro 53,2 ms, fenomeno ausente dos numeros deles.
3. **Validacao**: eles validam o esquema PDE (L2 vs exata); nos validamos a concordancia device-a-device (checksum bit-identico em todas as particoes).
4. **Complemento**: cuPentBatch = kernel SOTA (candidato a plugar no dispatcher); este trabalho = despacho SOTA (o que cuPentBatch nao faz).

## Resultados medidos (16/08/2026, RTX 5000 Ada, CUDA 13.0, kernels inalterados do repo v1.0)

| Linhas | Producao (ms) | cpb factor | cpb solve | cpb F+S | Speedup S | Speedup F+S |
|---|---|---|---|---|---|---|
| 2.048 | 0,96 | 1,61 | 1,07 | 2,68 | 0,90x | 0,36x |
| 16.384 | 1,94 | 2,52 | 2,29 | 4,81 | 0,85x | 0,40x |
| 131.072 | 18,39 | 20,76 | 21,30 | 42,06 | 0,86x | 0,44x |
| 262.144 | 36,25 | 40,24 | 38,29 | 78,53 | 0,95x | 0,46x |

**Validacao (262.144 linhas)**: checksums identicos (7,63448874e+07); max|diff| = 1,11e-16; residuo ||Ax-b||inf: cuPentBatch 5,41e-16, producao 4,44e-16 → ambos resolvem o mesmo sistema exato.

**DMA RHS 2,14 GB H2D = 86,0 ms > solve (36-38 ms)** — o pipeline "tudo em GPU" e dominado pela transferencia, nao pelo kernel.

**Leitura honesta**: o layout interleaved do cuPentBatch nao paga no nosso problema (n=1020) porque ele rele os 5 arrays de fatoracao (10,7 GB) da DRAM a cada solve; nosso kernel pre-fatora 1x no CPU e le fatores via argumentos (L1/const). Os speedups do paper (2x vs gpsv, 5-6x vs OpenMP, 10-20x vs serial) sao em Titan X Pascal com n ate 1e4 (regime latency-bound). **Revisao do plano**: nao plugar kernel interleaved; o alvo real e o overhead por lote do dispatcher (GPU-only 127,1 ms vs kernel puro 36,2 ms = 3,5x) → variante de lancamento unico por fatia GPU (mega-lote) com top-up reativo.

## Proxima etapa proposta

1. Clonar cuPentBatch (Apache 2.0) e compilar no servidor (CUDA 13); **FEITO** — numeros acima.
2. benchmark no mesmo tamanho (n=1020, lote 262.144, doubles): `cuPentBatchConstant` vs nosso kernel; **FEITO** — 38,3 vs 36,2 ms; kernel interleaved refutado por medicao.
3. **Revisado**: variante mega-lote no dispatcher (lancamento unico por fatia GPU, top-up reativo mantido) + revalidacao best-of-5 EMA/SA com checksum bit-identico (3,46459e+07).
4. publicar pico do kernel (LU banda, fatores em registrador) x despacho hibrido + reproducao cuPentBatch.
