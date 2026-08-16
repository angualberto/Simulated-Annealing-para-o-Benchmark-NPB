# Conclusão — Ajuste Dinâmico (Simulated Annealing) como Diferencial de Eficiência

O ponto central do solver não é apenas o kernel GPU ou a paralelização CPU, mas o **fracionamento adaptativo CPU/GPU gerenciado por simulated annealing (SA)**, que se autoajusta à máquina sem intervenção do usuário:

- **Resultado medido nesta máquina** (RTX 5000 Ada + CPUs do cluster): o SA convergiu sozinho para 72% das linhas na GPU e 28% na CPU, produzindo **163 ms por varredura** — 13% mais rápido que o modo GPU-puro (185 ms) e **3,7× mais rápido que o modo CPU-puro** (611 ms). Nenhum dos dois modos fixos, nem mesmo o GPU dedicado, superou o equilíbrio encontrado por busca automática.
- **Adaptação ao hardware é a própria definição do método**: como a função custo é medida em tempo real por janela deslizante (SA_WIN), o SA converge para a fração ótima conforme a velocidade relativa CPU/GPU da máquina. Em uma máquina com CPU lenta ele tende sozinho para tudo-GPU; com GPU ocupada ou CPUs rápidas, ele realoca trabalho para CPU — sem flags, sem recalibração, sem conhecimento prévio do hardware.
- **Estabilidade e reprodutibilidade**: o checksum permaneceu bit-idêntico (1,88943960e+08) nas 1000 iterações e nos três modos de execução, provando que a adaptação dinâmica não compromete a correção numérica — apenas a distribuição de trabalho.
- **Conclusão**: o ajuste dinâmico via SA é o que torna o solver eficiente de forma portável: ele transforma a heterogeneidade da máquina (CPU ociosa + GPU) em ganho real de vazão, e escala a 1,061 trilhão de pontos em 213,9 s com a garantia de estar operando próximo do equilíbrio ótimo local daquele hardware naquele instante.

## Dados de suporte

| Modo | Tempo/iter (ms) | GFLOPS (28 flops/ponto) | Split GPU/CPU |
|---|---|---|---|
| Híbrido SA (auto) | **163,1** | 182,1 | 72% / 28% |
| GPU puro (HSP_FIXED_FRAC=0) | 184,9 | 160,7 | 100% / 0% |
| CPU puro (HSP_FIXED_FRAC=1) | 611,1 | 48,6 | 0% / 100% |

- Checksum: 1,88943960e+08 (bit-idêntico em todos os modos e iterações)
- Run de referência: 1.000 iterações × 1.061.191.680 pontos = **1,061e12 pontos** (marca atingida) em 213,9 s
- GPU: NVIDIA RTX 5000 Ada Generation — temperatura máxima 70 °C (trava: 84 °C)
- Comparação NPB-GPU (mesma GPU): 200–216 GFLOPS (contagem oficial NAS, passo completo); este solver: 182 GFLOPS (estimador próprio, varredura penta pura com -fmad=false)