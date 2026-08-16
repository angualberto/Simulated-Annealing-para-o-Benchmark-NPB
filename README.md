# Simulated Annealing para o Benchmark NPB

Solver pentadiagonal híbrido CPU--GPU com ajuste dinâmico por *Simulated
Annealing* (SA) para a varredura do benchmark NAS Parallel Benchmark (NPB SP).

## Destaques

- Balanceamento CPU/GPU decidido em tempo de execução por SA (Metropolis),
  sem calibração manual (*zero-profiling*);
- Mega-lotes de 128 blocos de kernel para minimizar *launches*;
- Telemetria térmica via NVML com trava de segurança (84 °C);
- Reproducibilidade bit-a-bit: checksum `1,88943960e+08` idêntico em todos
  os modos e iterações;
- Run de referência: **1,061 trilhão de pontos** (1.040.384.000 sistemas de
  ordem 1020) em 213,9 s, a 163 ms/varredura (~182 GFLOPS, 28 flops/ponto).

## Estrutura

| Caminho            | Conteúdo |
|--------------------|----------|
| `hibrido/`         | Código CUDA do solver (evolução `hybrid_*.cu` até `hybrid_hlist_sp_sa_monitored.cu`), logs e JSONs do run de 1T |
| `artigo/`          | Artigo SBC (LATeX + PDF), bibliografia e figuras |
| `resultados_hibrido/` | Campanhas de revalidação (JSONs crus, relatório consolidado, comparações com cuPentBatch e mega-lote) |
| `sa_mesh.f90`, `sp_class_e.f90`, `sp_class_e_cuda.cu` | Malha e referências do NPB SP |

## Compilação (no servidor com GPU)

```bash
nvcc -O3 -arch=sm_89 -fmad=false -Xcompiler "-O3 -fopenmp" \
  hibrido/hybrid_hlist_sp_sa_monitored.cu -o sp_sa_monitored -lpthread \
  -L/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs -lnvidia-ml
```

Execução: `./sp_sa_monitored n out.json nb lines_per_block iteracoes temp_limit`
com variáveis de ambiente `HSP_CK_EVERY` (checksum amostrado) e
`HSP_FIXED_FRAC` (0 = somente GPU, 1 = somente CPU).

## Autores

André Gualberto · Gabriel Thomaz do Nascimento (LNCC)