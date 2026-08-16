#!/bin/bash
set -e

# ==============================================================================
# BENCHMARK COMPARATIVO HETEROGÊNEO (MESMA MÁQUINA / RTX 5000 Ada + AMD EPYC)
# Métodos: GPU Pura, CPU Pura, Estático (50/50), Reativo (EMA), SA (Proposto)
#
# Protocolo: melhor-de-5 com descarte de warm-up; checksum bit-a-bit por rodada.
# Uso: ./benchmark_comparativo.sh [num_iteracoes_por_rodada]
#   (default: 1 iteracao/rodada -> varredura unica de 262.144 sistemas)
# ==============================================================================

BIN="./sp_sa_monitored"
SRC="hybrid_hlist_sp_sa_monitored.cu"
N_PENTA=1020
LINES_PER_BLK=256
N_BLOCOS=1024       # 1024 blocos x 256 linhas = 262.144 sistemas lineares
REPETICOES=5
WARMUP=1
TEMP_LIMITE=84
ITER_POR_RODADA="${1:-1}"
CSV_OUT="comparativo_metodos_heterogeneos.csv"

NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"

# 1. Compilação com precisão estrita e suporte NVML (se nvcc disponivel)
if [ -x "$NVCC" ]; then
    echo "[+] Compilando binário heterogêneo (-fmad=false, NVML, OpenMP)..."
    "$NVCC" -O3 -arch=sm_89 -fmad=false -Xcompiler "-O3 -fopenmp" \
        "$SRC" -o "$BIN" -lpthread -lnvml || \
        "$NVCC" -O3 -arch=sm_89 -fmad=false -Xcompiler "-O3 -fopenmp" \
            "$SRC" -o "$BIN" -lpthread -L/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs -lnvidia-ml
else
    echo "[!] nvcc nao encontrado. Tentando binario pré-existente: $BIN"
    [ -x "$BIN" ] || { echo "ERRO: sem nvcc e sem $BIN. Compile no servidor com GPU."; exit 1; }
fi

echo "metodo,frac_forcada,tempo_min_ms,tempo_medio_ms,gflops_pico,temp_gpu_c,consumo_w,checksum_ok" > "$CSV_OUT"

executar_teste() {
    local nome_metodo="$1"
    local frac_env="$2"
    local tempos=()
    local temp_max=0
    local pwr_max=0
    local checksum_ref=""
    local chk_ok="SIM"
    local total=$(( N_BLOCOS * LINES_PER_BLK * N_PENTA ))

    echo -e "\n>>> Testando: ${nome_metodo} (melhor-de-${REPETICOES}, +${WARMUP} warm-up)..."
    export HSP_FIXED_FRAC="$frac_env"

    # rodada de warm-up (descartada) + bateria
    for r in $(seq 0 $REPETICOES); do
        $BIN $N_PENTA "tmp_bench.json" $N_BLOCOS $LINES_PER_BLK $ITER_POR_RODADA $TEMP_LIMITE > /dev/null

        read -r t_ms temp_c pwr_w chk gflops <<< \
            "$(python3 -c 'import json; d=json.load(open("tmp_bench.json")); print(d["tempo_medio_ms"], d["telemetria"]["temp_c"], d["telemetria"]["power_w"], d["checksum"], d["gflops"])')"

        if [ "$r" -eq 0 ]; then
            printf "  Warm-up: %6.2f ms | Temp: %2d°C | Consumo: %5.1f W\n" "$t_ms" "$temp_c" "$pwr_w"
            continue
        fi

        tempos+=("$t_ms")
        if [ "$temp_c" -gt "$temp_max" ]; then temp_max=$temp_c; fi
        if awk -v a="$pwr_w" -v b="$pwr_max" 'BEGIN{exit !(a>b)}'; then pwr_max=$pwr_w; fi

        if [ -z "$checksum_ref" ]; then
            checksum_ref="$chk"
        elif [ "$checksum_ref" != "$chk" ]; then
            chk_ok="DIVERGENTE"
        fi

        printf "  Rodada %d: %6.2f ms | Temp: %2d°C | Consumo: %5.1f W | Checksum: %.8e\n" \
               "$r" "$t_ms" "$temp_c" "$pwr_w" "$chk"
    done

    # Melhor tempo e média (melhor-de-5)
    min_t=$(printf "%s\n" "${tempos[@]}" | sort -n | head -n 1)
    sum_t=$(printf "%s\n" "${tempos[@]}" | awk '{s+=$1} END {printf "%.2f", s/NR}')

    # GFLOPS (28 flops/ponto; mesma formula do solver)
    gflops=$(TOTAL=$total MIN_T=$min_t python3 -c 'import os
print("%.2f" % ((int(os.environ["TOTAL"]) * 28.0) / (float(os.environ["MIN_T"]) * 1e6)))')

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
           "$nome_metodo" "$frac_env" "$min_t" "$sum_t" "$gflops" "$temp_max" "$pwr_max" "$chk_ok" >> "$CSV_OUT"
}

# ==============================================================================
# BATERIA DE TESTES COMPARATIVOS
# ==============================================================================
executar_teste "Somente-GPU (100% GPU)"      "0.0"
executar_teste "Somente-CPU (100% CPU)"      "1.0"
executar_teste "Estatico Balanceado (50/50)" "0.5"
executar_teste "Reativo por Taxa (EMA)"      "0.42"
executar_teste "SA Adaptativo (Proposto)"    "-1.0"  # -1 ativa a heuristica de busca online

rm -f tmp_bench.json

# ==============================================================================
# EXIBIÇÃO CONSOLIDADA DOS RESULTADOS
# ==============================================================================
echo -e "\n======================================================================================"
echo "        CONSOLIDADO EXPERIMENTAL NA MESMA MAQUINA ($ITER_POR_RODADA iteracao(ões)/rodada)"
if command -v column > /dev/null 2>&1; then
    column -s, -t "$CSV_OUT"
else
    cat "$CSV_OUT"
fi
echo "======================================================================================"
echo "Resultados exportados para: $CSV_OUT"