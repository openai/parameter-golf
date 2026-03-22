# Propuesta de Contribucion Original: Parameter Golf

*Actualizado: 2026-03-22 — Post-descarte de depth recurrence*

## Estado del Arte (2026-03-22)

### Merged SOTA: val_bpb 1.1428
- 10L, 512d, int5-MLP/int6-attn, BigramHash(10240), SWA(0.4), WD=0.04, zstd-22

### PR Frontier: val_bpb 1.1246 (PR #374)
- 11L, 512d, Tight SWA (scale<0.2), Shared VE128, Partial RoPE, LN Scale, XSA4, Late QAT

### Mejor resultado con TTT: val_bpb 1.1254 (PR #338)
- 11L, XSA4 + EMA + TTT SGD (3 epochs, freeze 2 blocks)

### Stack Convergido
Todos los top-20: 11L/512d/MLP3x/relu²/GQA(8H/4KV)/int6+zstd-22/SmearGate/BigramHash/OrthoInit/WD=0.04/sliding eval stride=64.

---

## Lecciones de Nuestros Experimentos

### Depth Recurrence: DESCARTADA
- **RingGolf Phase 1b**: 8 bloques unicos × 3 loops = 16 capas efectivas, dim=576
- **Resultado**: val_bpb 1.2631 post-quant (30 min, 1×H100) vs Consensus 1.2235
- **Per-step**: RingGolf converge mejor (+0.04 BPB ventaja a step 1000)
- **Per-wallclock**: Pierde por throughput (1080ms/paso vs 578ms)
- **PR #363 confirma**: cuantizacion se amplifica 900× a traves de recurrence
- **PR #386 confirma**: shared block × 12 = 1.4061 (no competitivo)
- **PR #375 meta-insight**: cada 1ms de overhead/paso cuesta ~0.006 BPB
- **Conclusion**: La profundidad extra no compensa el throughput perdido en regimen de 10 min

### Consensus Stack v3 (nuestro mejor resultado)
- 11L, 512d, MLP3x, SmearGate, mixed int6/int8, SWA, WD=0.038
- **val_bpb 1.3341** (TTT LoRA) en 1×H100, 10 min (1035 pasos)
- **val_bpb 1.2087** (TTT LoRA) en 1×H100, 30 min (2850 pasos)
- Degradacion post-quant: +0.0075 BPB (buena)

### XSA + EMA + Late QAT (v4)
- Funciona con suficientes pasos (30 min: val_bpb 1.2235 post-quant)
- Falla con pocos pasos (10 min 1×H100: val_bpb 1.4519 — peor que v3)
- Necesita 8×H100 (~7K pasos) para ser competitivo

---

## Nueva Direccion: Eval-Time Optimization Stack

### Por que eval-time

La arquitectura base (11L/512d/MLP3x) esta **saturada**. Las mejoras restantes vienen de:
1. Post-training quantization optimization (GPTQ-lite, gradient-guided)
2. Weight averaging (Tight SWA, EMA)
3. Eval-time adaptation (TTT, PPM-C mixer)
4. Eval strategy (stride 32, sliding window)

### Mapa de Tecnicas Eval-Time

| Tecnica | Mejora | Costo | PR | Estado |
|---------|--------|-------|-----|--------|
| **Tight SWA** (scale<0.2) | Zero penalty | 0 | #374 | Validado |
| **Shared VE128** (capas 9,10) | ~0.002 BPB | ~32KB | #374 | Validado |
| **GPTQ-lite** (clip search) | Reduce quant gap | 0 runtime | #379 | Validado |
| **TTT 8 epochs + stride 32** | -0.001 BPB | 285s eval | #390 | Validado |
| **PPM-C context mixer** | ~0.015 BPB | ~60 LOC, 0 artifact | #283 | Parcial |
| **Neural Cache** (KV cross-window) | Desconocido | Complejo | #318 | Bug, sin resultados |

### Interacciones Criticas

- **TTT + XSA = NEGATIVO** (PR #303: +0.016 peor). Son redundantes.
- **TTT + SmearGate = POSITIVO** (PR #254, #338 lo resuelve congelando 2 bloques)
- **EMA > SWA** por 0.003 BPB (PR #375 verifica)
- **786K batch > 524K** por 0.004 BPB (PR #375 — total tokens > gradient steps)
- **MTP, INT4, Canon layers = NEGATIVOS** (PR #375, $500 de compute)

---

## Propuesta: Eval-Time Ensemble Stack

### Idea Central

Combinar las mejores tecnicas de eval-time que son **complementarias y no exploradas juntas**:

1. **Base training**: Reproducir PR #374 stack (el SOTA actual sin TTT)
   - 11L, 512d, MLP3x, XSA4, Tight SWA, Partial RoPE, LN Scale, Late QAT, VE128
   - Estimado: ~1.1246 BPB (reproduccion)

2. **GPTQ-lite**: Busqueda de clip optimo por capa durante cuantizacion
   - Zero training cost, reduce quant gap ~0.002 BPB
   - Estimado: ~1.1226 BPB

3. **PPM-C context mixer** (NOVEL para esta combinacion):
   - Mezcla de probabilidades neural + PPM order-2 por documento
   - alpha=0.95 neural / 0.05 PPM
   - PR #283 muestra ~0.015 BPB mejora en subset, nadie lo combino con SOTA stack
   - Zero artifact cost, ~60 LOC
   - Estimado: ~1.1076-1.1176 BPB

4. **Eval stride 32** (no 64):
   - PR #390 muestra mejora con stride mas fino
   - Costo: 2× mas eval time (factible dentro de 600s)

### Lo que es Original

1. **PPM-C + GPTQ-lite + SOTA stack**: Nadie ha combinado PPM context mixing con el stack XSA/VE/Partial RoPE. El PPM opera a nivel de probabilidades (no representaciones), asi que es ortogonal a XSA.

2. **Si PPM funciona**: Seria la primera submission que mezcla un modelo clasico (PPM) con un neural model para mejorar BPB. El argumento teorico es solido — PPM captura patrones locales exactos que el neural model promedia.

3. **Si PPM no funciona**: Caemos a reproduccion de SOTA + GPTQ-lite, que aun seria competitiva (~1.1226) y documenta la negative result.

### Presupuesto de Eval Time (600s)

| Fase | Tiempo estimado |
|------|----------------|
| Sliding window eval (stride 32) | ~170s |
| GPTQ-lite clip search (5 ratios × ~90 tensors) | ~30s |
| PPM-C construction per-doc | ~60s |
| PPM-C probability mixing | ~20s |
| **Total** | **~280s** (dentro de 600s) |

---

## Plan de Ejecucion (Revisado)

### Fase 1: Reproducir PR #374 base (8×H100, 10 min)
1. Implementar Tight SWA, Shared VE128, Partial RoPE, LN Scale
2. Usar LRs y hiperparametros exactos de PR #374
3. Necesitamos zstd-22 (agregar dependencia `zstandard`)
4. **Criterio**: val_bpb < 1.13 (sliding s64)
5. **Costo**: ~$3.50 (1 run 8×H100)

### Fase 2: Agregar GPTQ-lite (solo post-training)
1. Implementar busqueda de clip ratio por tensor
2. 5 ratios: [99.9, 99.99, 99.999, 99.9999, 100.0] percentiles
3. Seleccionar el que minimiza reconstruction error L2
4. **Criterio**: quant gap < 0.006 BPB

### Fase 3: Implementar PPM-C mixer (eval-time)
1. PPM-C order-2 construido per-document a partir de tokens ya evaluados
2. Log-probability mixing: `log_p = alpha * log_neural + (1-alpha) * log_ppm`
3. Alpha sweep: [0.90, 0.93, 0.95, 0.97, 0.99]
4. Per-doc mode (no cross-document, cumple reglas)
5. **Criterio**: mejora > 0.005 BPB sobre base

### Fase 4: Submission (3 seeds)
1. Correr 3 seeds en 8×H100
2. Verificar reproducibilidad (std < 0.002)
3. Documentar resultados con ablation de cada tecnica
4. PR al upstream

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigacion |
|--------|-------------|------------|
| PPM-C no mejora sobre SOTA stack | Media | Contribucion como negative result + GPTQ-lite submission |
| No tenemos 8×H100 | Alta | Solicitar compute grant de OpenAI o probar en 1×H100 con 30min |
| Reproduccion de PR #374 falla | Baja | Stack bien documentado, multiples PRs lo reproducen |
| Eval time > 600s | Baja | PPM es O(n) per-doc, stride 32 ya validado en 170s |
| zstd no disponible en RunPod | Baja | Template tiene pip, `pip install zstandard` |

## Fallback

Si PPM-C falla: submission de reproduccion PR #374 + GPTQ-lite como "non-record: systematic quantization optimization". Si GPTQ-lite tambien falla: documentar ambos negative results como contribucion de investigacion.

---

## Apendice: Resultados de Nuestros Experimentos

| Run | Config | Pasos | val_bpb | Artifact | Hardware |
|-----|--------|-------|---------|----------|----------|
| mlx_smoke_baseline | 9L/512d baseline | 200 | 2.3244 | 10.1MB | Apple Silicon |
| consensus_v3 | 11L/512d/MLP3x/SmearGate/int6mix | 1035 | 1.3501→1.3341(TTT) | 15.2MB | 1×H100 10min |
| consensus_v4_30m | +XSA+EMA+Late QAT | 2850 | 1.2235→1.2087(TTT) | 17.5MB | 1×H100 30min |
| ringgolf_phase1b | 8blk×576d, 16 eff layers | 1610 | 1.2631 | 15.5MB | 1×H100 30min |

---

*Documento vivo — actualizar despues de cada fase*
