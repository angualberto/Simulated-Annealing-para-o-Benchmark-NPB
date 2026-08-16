# PACOTE FINAL - Despacho Hibrido GPU+CPU (NPB SP pentadiagonal) + Comparacoes

Data: 16/08/2026 (re-run completo e validado)

## Especificacoes da maquina de medicao (servidor LNCC)

- Host: virtual22.prjsieh.lncc.br (LNCC - Laboratorio Nacional de Computacao Cientifica)
- SO: Rocky Linux 9.6 (Blue Onyx) | kernel 5.14.0-570.32.1.el9_6 | x86_64
- CPU: AMD EPYC 9124 (Zen 4) - 16 cores / 32 threads, 1 socket
  (base 1,5 GHz / max 3,71 GHz | L1d 512 KiB, L1i 512 KiB, L2 16 MiB, L3 64 MiB)
- RAM: 125 GiB (NUMA node 0; CPUs 0-31)
- GPU: 2x NVIDIA RTX 5000 Ada Generation (experimentos no device 0)
  32 GB GDDR6 (32.760 MiB) | driver 580.65.06 | TDP 250 W | sm_89
- CUDA: 13.0 (V13.0.48)
- Compilacao: nvcc -O3 -fmad=false -arch=sm_89

## Quantidade de maquinas / recursos usados

| Recurso | Quantidade |
|---|---|
| Maquinas (nos) usadas | 1 (no unico) |
| Hosts que executaram | 1 (virtual22.prjsieh.lncc.br) |
| CPU do host | 1 socket AMD EPYC 9124, 16 cores / 32 threads (HT) |
| GPUs instaladas no no | 2 x NVIDIA RTX 5000 Ada (32 GB cada) |
| GPUs USADAS nos experimentos | 1 (device 0; a 2a ficou ociosa) |
| RAM do no | 125 GiB |
Execucoes no servidor: agualber@146.134.87.2:/prj/prjsieh/agualber/poroso/poroso/seg

## 1. O problema

Varredura pentadiagonal real do NPB SP classe E: A*u=b com n=1020,
a=(-0.25), b=(-1), c=6, d=(-1), e=(-0.25) (estrit. diag. dominante).
262.144 linhas independentes = 1024 blocos x 256 linhas. Precision: double.

## 2. O que ha aqui

- codigo/ ............ fontes COMPILADOS-COMPATIVEIS (vistos abaixo)
- resultados/ ........ saidas cruas do re-run final (logs + 6 JSONs)
- comparativos/ ...... comparativo_geral.txt (resumo mestre) + docs de
                       reproducao (cuPentBatch, "SOTA", exemplo) + JSONs
                       de resultados consolidados

## 3. Como compilar (no servidor, CUDA 13, sm_89)

    # Dispatcher hibrido (controlador SA online + HSP_FIXED_FRAC para baselines)
    nvcc -O3 -fmad=false -arch=sm_89 -o sp_sa_mega hybrid_hlist_sp_sa.cu -lpthread

    # Dispatcher hibrido (controlador EMA adaptativo)
    nvcc -O3 -fmad=false -arch=sm_89 -o sp_ema_mega hybrid_hlist_sp.cu -lpthread

    # Benchmark cuPentBatch (repo munstermonster/cuPentBatch v1.0, kernels
    # inalterados) vs nosso kernel de producao - REQUER cuPentBatch.cu/.h
    # no mesmo diretorio do fonte
    nvcc -O3 -fmad=false -arch=sm_89 -std=c++17 -o bench_cupentbatch bench_cupentbatch.cu

## 4. Como rodar

    # Matriz reduzida de validacao (1024 blocos, todos os modos)
    export HSP_FIXED_FRAC=0.0   && ./sp_sa_mega 1020 g0.json   1024 256   # all-GPU
    export HSP_FIXED_FRAC=1.0   && ./sp_sa_mega 1020 g1.json   1024 256   # all-CPU
    export HSP_FIXED_FRAC=0.5   && ./sp_sa_mega 1020 g05.json  1024 256   # frac fixa
    unset  HSP_FIXED_FRAC       && ./sp_sa_mega 1020 sa1.json  1024 256   # SA auto
                                  ./sp_ema_mega 1020 ema.json 1024 256    # EMA auto

    # Benchmark de kernel (262144 linhas, min-of-5, janela termica)
    ./bench_cupentbatch 262144 5

    # Exemplo reproduzido do tutorial "SOTA" (warp-shuffle invalido, 10,9 ms ERRADO)
    nvcc -O3 -fmad=false -arch=sm_89 -Xcompiler -fopenmp -o adv_vs_base \
         advanced_vs_baseline_sp_exemplo.cu && OMP_NUM_THREADS=16 ./adv_vs_base 1024

## 5. Numeros finais (re-run 16/08 11:36-11:37)

KERNEL (262.144 linhas, min-of-5):
  producao (LU exata, fatores CPU 1x)      36,23 ms
  cuPentBatch solve                        38,31 ms  (0,95x)
  cuPentBatch factor+solve                 78,50 ms  (0,46x)

DISPATCHER MEGA-LOTE (1024 blocos):
  all-GPU 48,08 ms | all-CPU 147,87 ms | frac 0,5 = 51,68 ms
  SA auto 42,72 ms (recorde) | EMA auto 47,29 ms
  vs antes (BATCH=8): GPU-only -63%, SA -42%, EMA -33%, frac0,5 -28%

VALIDACAO:
  checksum do projeto (RHS sin/cos): 3,46459000e+07 BIT-IDENTICO em 100%
    dos modos/particoes (GPU==CPU==splits)
  cuPentBatch vs producao (RHS=1.0): checksums identicos 7,63448874e+07,
    max |diff| 1,1e-16, residuo ||A x - b||inf: 5,4e-16 / 4,4e-16
    -> ambos resolvem o MESMO sistema exato; prova a exatidao da producao.
  Kernel "baseline simplificado" do exemplo: NAO exato (residuo ~3,75).
  "SOTA" warp-shuffle do tutorial: invalido (checksum ~8% divergente).
  DMA RHS 2,14 GB H2D = 86 ms > solve (36-38 ms): tudo-em-GPU e DMA-bound.

## 6. Fontes primarias

- resultados/*.log/.json (re-run final)
- comparativos/comparacao_cupentbatch.json, comparacao_mega_lote.json
- comparativos/comparativo_geral.txt (tabelas A-D completas)