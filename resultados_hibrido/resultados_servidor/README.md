# resultados_servidor — espelho dos resultados do servidor (146.134.87.2)

Baixado em 16/08/2026 de `/tmp/final.tar.gz` do servidor agualber@146.134.87.2
(compilado e executado em /prj/prjsieh/agualber/poroso/poroso/seg, JSONs em /tmp do servidor).

## Convenção de nomes

| Prefixo | Significado |
|---|---|
| `e64_k`, `e512_k`, `e1024_k` | EMA adaptativo, 64/512/1024 blocos (256 linhas/bloco), invocação k=1..5 (best-of-3 interno) — **intercalado com SA** (campanha final) |
| `s64_k`, `s512_k`, `s1024_k` | SA on-line, mesmo desenho, invocação k=1..5 |
| `gck.json` | all-GPU 1024 blocos (baseline só-GPU) |
| `ccck.json` | all-CPU 1024 blocos (baseline só-CPU) |
| `ema_1..3`, `ema_t` | repetições EMA + trace (campanha anterior, mesma correção de kernel) |
| `sa_1..3` | repetições SA (campanha anterior) |
| `sw_0.25_1..3` etc. | sweep de fração estática 0.25/0.45/0.65/0.85, min-of-3 |
| `swp_0.05`..`swp_0.95` | sweep de fração estática, uma invocação cada (0.05 a 0.95) |
| `b_1..5` | execuções "best-of" avulsas (checksum correto 3.46459e7); não usadas nos números finais |
| `ck_c/ck_g/ck_m.json` | validações extras do checksum (CPU/GPU/mix) — todas 3.46459e7 (= correto) |
| `orig_1024.json` | **versão com bugs (antes dos fixes)**: t=56.49 ms, checksum 1.44182e7 ≠ correto — evidência do erro antigo (cache GPU + b não transformado) |
| `tmp_1020*.json` | execuções de debug/temporárias |
| `hl_dbg*.json` | runs de debug (lentos, ck ausente) — históricos |

### `seg/` (45 JSONs — saídas de execução do diretório do projeto no servidor)
- `hlsp_fix_1020_1024_1..3` / `hlsp_sa_1020_1024_1..3` (idem 512, 64): repetições 3× (EMA/SA corrigidos) — os best dessas séries deram origem aos números finais
- `hlsp_fix_1020_*.json` / `hlsp_sa_1020_*.json`: uma execução por size (séries antigas)
- `hl_*`, `ll_*`, `resultado_*.json`, `tmp_1020*`: campanhas anteriores (pré-correção) — históricos; **não usar como números finais**

## Números-síntese (best-of-5 intercalado)

| blocos | EMA (ms) | SA (ms) | split SA | checksum |
|---|---|---|---|---|
| 64 | 6.56 | 6.59 | G43/C21 | 1.881050e+06 |
| 512 | 36.23 | 39.07 | G266/C246 | 1.611680e+07 |
| 1024 | 69.95 | 73.49 | G537/C487 | 3.464590e+07 |

- Checksum bit-idêntico entre all-GPU (gck), all-CPU (ccck), mix, EMA e SA → correção provada.
- Arquivos finais canonical (cópia dos best): `../new/hlsp_fix_1024_final.json` e `../new/hlsp_sa_1024_final.json`.

## Campos de cada JSON
- `tempo_total_ms`, `blocos.gpu`, `blocos.cpu` (split final), `traco[]` (frac_cpu por checkpoint — convergência do controlador), `checksum` (soma validada do repeat), `taxa_gpu_s`, `taxa_cpu_s` (taxas medidas on-line).

## Pendência
Espelho completo baixado em 16/08/2026 (2ª tentativa): 84 JSONs de /tmp + 45 de
/prj/prjsieh/agualber/poroso/poroso/seg/ → subpasta `seg/`. Nada pendente.