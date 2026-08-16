#!/bin/bash
set -e

# ==============================================================================
# BENCHMARK MEGA-LOTE (MEGA=128) - 5 METODOS HETEROGENEOS NA MESMA MAQUINA
# Usa o binario sp_sa_mega (variante 1 kernel+sync por 128 blocos, campanha
# do relatorio TABELA 8). Sem telemetria NVML nesse binario -> temp/consumo N/A.
# Protocolo: warm-up + melhor-de-5; checksum bit-a-bit por rodada.
# ==============================================================================

BIN="/prj/prjsieh/agualber/poroso/poroso/seg/sp_sa_mega"
N_PENTA=1020
LINES_PER_BLK=256
N_BLOCOS=1024
REPETICOES=5
WARMUP=1
CSV_OUT="comparativo_megalote_5metodos.csv"

echo "metodo,frac_forcada,tempo_min_ms,tempo_medio_ms,blocos_gpu,blocos_cpu,checksum_ok" > "$CSV_OUT"

executar_teste() {
    local nome_metodo="$1"
    local frac_env="$2"
    local tempos=()
    local checksum_ref=""
    local chk_ok="SIM"
    local total=$(( N_BLOCOS * LINES_PER_BLK * N_PENTA ))
    local bgpu=0; local bcpu=0

    echo -e "\n>>> Testando: ${nome_metodo} (MEGA=128; melhor-de-${REPETICOES}, +${WARMUP} warm-up)..."
    export HSP_FIXED_FRAC="$frac_env"

    for r in $(seq 0 $REPETICOES); do
        $BIN $N_PENTA "tmp_mega.json" $N_BLOCOS $LINES_PER_BLK > /dev/null 2>&1
        read -r t_ms chk g cpu <<< \
            "$(python3 -c 'import json; d=json.load(open("tmp_mega.json")); print(d["tempo_total_ms"], d["checksum"], d["blocos"]["gpu"], d["blocos"]["cpu"])')"

        if [ "$r" -eq 0 ]; then
            printf "  Warm-up: %7.2f ms | Checksum: %.8e | G%d/C%d\n" "$t_ms" "$chk" "$g" "$cpu"
            continue
        fi

        tempos+=("$t_ms")
        if [ -z "$checksum_ref" ]; then
            checksum_ref="$chk"
        elif [ "$checksum_ref" != "$chk" ]; then
            chk_ok="DIVERGENTE"
        fi
        printf "  Rodada %d: %7.2f ms | Checksum: %.8e | G%d/C%d\n" "$r" "$t_ms" "$chk" "$g" "$cpu"
    done

    min_t=$(printf "%s\n" "${tempos[@]}" | sort -n | head -n 1)
    sum_t=$(printf "%s\n" "${tempos[@]}" | awk '{s+=$1} END {printf "%.2f", s/NR}')
    gflops=$(TOTAL=$total MIN_T=$min_t python3 -c 'import os
print("%.2f" % ((int(os.environ["TOTAL"]) * 28.0) / (float(os.environ["MIN_T"]) * 1e6)))')

    : "${bgpu:=$g}"; : "${bcpu:=$cpu}"
    printf "%s,%s,%s,%s,%s,%s,%s\n" \
           "$nome_metodo" "$frac_env" "$min_t" "$sum_t" "$bgpu" "$bcpu" "$chk_ok" >> "$CSV_OUT"
}

executar_teste "Somente-GPU (100% GPU)"      "0.0"
executar_teste "Somente-CPU (100% CPU)"      "1.0"
executar_teste "Estatico Balanceado (50/50)" "0.5"
executar_teste "Reativo por Taxa (EMA)"      "0.42"
executar_teste "SA Adaptativo (Proposto)"    "-1.0"

rm -f tmp_mega.json

echo -e "\n==========================================================================="
echo "      CONSOLIDADO MEGA-LOTE (5 METODOS, MESMA MAQUINA / RTX 5000 Ada)"
if command -v column > /dev/null 2>&1; then
    column -s, -t "$CSV_OUT"
else
    cat "$CSV_OUT"
fi
echo "==========================================================================="
echo "Exportado para: $CSV_OUT"