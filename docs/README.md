# Parameter Golf - Base de Conocimientos

## El Problema

**Parameter Golf** es un challenge de OpenAI que plantea una pregunta fundamental en deep learning: **dado un presupuesto fijo de parámetros, cual es el mejor modelo de lenguaje que puedes entrenar?**

### Definicion Formal

Optimizar **L(N)** — la menor loss posible para un numero fijo de parametros N — sin restricciones de datos, compute, pasos o arquitectura. Es el dual de otros challenges:

| Challenge | Optimiza | Restriccion |
|-----------|----------|-------------|
| **Parameter Golf** | L(N) — menor loss | Parametros fijos (16MB artifact) |
| NanoGPT Speedrun | L(T) — menor tiempo | Loss fija (3.28 val_loss) |
| NanoGPT Slowrun | L(D) — menor loss | Dataset fijo |

### Restricciones del Challenge

| Restriccion | Valor |
|-------------|-------|
| **Tamano del artifact** | ≤ 16,000,000 bytes (decimal, no MiB) |
| **Composicion del artifact** | codigo (`train_gpt.py`) + pesos int8 comprimidos con zlib |
| **Tiempo de entrenamiento** | ≤ 10 minutos en 8×H100 SXM |
| **Tiempo de evaluacion** | ≤ 10 minutos adicionales en 8×H100 |
| **Red durante eval** | Prohibida (no downloads, no network calls) |
| **Metrica** | val_bpb (bits per byte) en FineWeb validation set fijo |
| **Significancia estadistica** | Mejorar SOTA por ≥0.005 nats con p < 0.01 (tipicamente 3 runs) |

### Que es val_bpb?

**Bits per byte (BPB)** mide la capacidad de compresion del modelo sobre texto crudo, independiente del tokenizador. Es la metrica definitiva porque:

1. **Tokenizer-agnostic**: Permite comparar modelos con vocabularios distintos (1024, 4096, etc.)
2. **Interpretacion directa**: Cuantos bits necesita el modelo para predecir cada byte del texto
3. **Relacion con val_loss**: `val_bpb = val_loss * (tokens_in_val / bytes_in_val)` — ajusta la cross-entropy loss por la razon de compresion del tokenizador

Menor es mejor. El baseline logra ~1.224 val_bpb; el SOTA merged es ~1.175, pero la frontera de PRs abiertos alcanza ~1.132.

---

## Arquitectura Base

### Modelo: GPT con optimizaciones modernas

El modelo base es un transformer decoder-only con las siguientes caracteristicas:

| Componente | Implementacion | Proposito |
|------------|---------------|-----------|
| **Atencion** | Grouped Query Attention (GQA) | 8 heads, 4 KV heads — reduce memoria manteniendo calidad |
| **Posicion** | Rotary Position Embeddings (RoPE) | Codificacion posicional relativa, extendible a contextos mas largos |
| **MLP** | SwiGLU | Activacion gated que mejora expresividad vs ReLU/GELU |
| **Embeddings** | Tied (input = output head) | Ahorra parametros significativamente con vocab pequeno |
| **Logits** | Softcap (30.0) | Estabiliza entrenamiento evitando logits extremos |

### Configuracion Base (Naive Baseline)

```
9 capas transformer, dimension 512
8 attention heads, 4 KV heads
Vocabulario 1024 tokens (BPE)
Secuencia 1024 tokens
~524K tokens por batch
~13,500 pasos en 10 minutos
```

**val_bpb baseline: 1.2244**

### Optimizador: Muon

El optimizador principal es **Muon** (basado en modded-nanogpt), que usa **ortogonalizacion Newton-Schulz** para precondicionamiento:

- Aplica una aproximacion polinomial a la descomposicion polar de los gradientes
- Mas eficiente que Adam para matrices de pesos grandes
- Adam se usa solo para parametros escalares y embeddings
- Los learning rates son separados: `MATRIX_LR` (Muon), `SCALAR_LR` (Adam), `TIED_EMBED_LR`, `EMBED_LR`

### Pipeline de Cuantizacion

El modelo entrenado en FP32/BF16 se comprime para el artifact final:

```
Baseline:  Pesos FP32 → int8           → zlib   → Artifact
Frontera:  Pesos BF16 → int6 (QAT/STE) → zstd-22 → Artifact
```

La **degradacion post-cuantizacion** es un factor critico:
- Baseline (int8 + zlib): ~0.007 BPB de penalizacion (pre-quant 1.217 → post-quant 1.224)
- Frontera (int6 + QAT + SWA): ~0.001 BPB con tecnicas especializadas
- Int6 comprime ~1.5× mas que int8, liberando espacio para mas capas/params en el artifact de 16MB

---

## Leaderboard y Progresion de Tecnicas

### Estado Actual (2026-03-20)

| # | Run | val_bpb | Delta vs Baseline | Tecnicas Clave |
|---|-----|---------|-------------------|----------------|
| 1 | Muon WD + 10L + OvertoneInit | **1.1748** | -0.0496 | Sliding window eval, FP16 emb, 10 capas, Muon WD, spectral init |
| 2 | Sliding Window Eval | 1.1925 | -0.0319 | Evaluacion con ventana deslizante stride=64 |
| 3 | LoRA TTT | 1.1929 | -0.0315 | Test-time training con LoRA por documento |
| 4 | Seq4096 + Muon Tuning | 1.2014 | -0.0230 | Secuencia 4096, momentum 0.99, LR bajo |
| 5 | Seq2048 | 1.2060 | -0.0184 | Secuencia 2048, LR tuning |
| 6 | int6 Mixed Precision | 1.2147 | -0.0097 | 10 capas, cuantizacion selectiva int6/int8 |
| 7 | FP16 Embed | 1.2197 | -0.0047 | Embedding FP16, warmdown tuning |
| 8 | Naive Baseline | 1.2244 | — | 9×512, vocab 1024, baseline |

### Anatomia del SOTA (1.1748)

El record actual combina multiples mejoras ortogonales:

1. **Sliding Window Eval** (stride=64): Cada token se evalua con ~960+ tokens de contexto en vez del promedio 0-1023. Mejora ~0.032 BPB sin cambiar entrenamiento.

2. **FP16 Tied Embedding**: Mantiene el embedding en FP16 en vez de int8. La matriz de embedding es la mas sensible a cuantizacion porque tiene doble funcion (input + output). Mejora ~0.006 BPB, cuesta ~500KB extra.

3. **10 capas** (vs 9): Una capa adicional aprovechando el espacio liberado por cuantizacion mixta.

4. **Muon Weight Decay** (0.02): Regularizacion desacoplada que mejora generalizacion y produce distribuciones de pesos mas amigables con cuantizacion.

5. **Overtone Spectral Init**: Inicializacion con SVD donde los valores singulares siguen ley de potencias (S_k ~ k^{-0.5}). Mejora convergencia temprana.

6. **Phase-Transition Residual Mixing**: Mezcla residual programada con sigmoid entre capas.

---

## Insights Fundamentales

### 1. La Cuantizacion es el Cuello de Botella

La penalizacion post-cuantizacion (0.001–0.014 BPB) domina sobre la mayoria de ajustes de hiperparametros. El run de 4 horas (non-record) demuestra esto dramaticamente:

```
Pre-quant val_bpb:  1.1749  (excelente)
Post-quant val_bpb: 1.2074  (penalizacion de 0.0325 BPB)
```

Mas entrenamiento no ayuda si los pesos no son cuantizacion-friendly. Las estrategias efectivas son:
- **FP16 embedding**: Evitar cuantizar la capa mas sensible
- **Warmdown extendido**: Producir distribuciones de pesos mas estrechas
- **Cuantizacion mixta int6/int8**: Capas intermedias toleran menor precision
- **Quantization-aware training**: Entrenar con la cuantizacion en mente

### 2. La Evaluacion Importa Tanto Como el Entrenamiento

Sliding window eval con stride=64 proporciona **0.032 BPB** de mejora — mas que cualquier cambio individual de arquitectura o hiperparametros. Cada token se predice con contexto casi completo en vez del promedio progresivo.

Test-time training (LoRA TTT) anade otra dimension: adaptar el modelo por documento durante evaluacion. La clave es la **evaluacion aislada por documento** — no mezclar contextos entre documentos distintos.

### 3. Contexto Mas Largo Compensa Menos Pasos

| Seq Length | Pasos en 10min | ms/paso | val_bpb |
|-----------|---------------|---------|---------|
| 1024 | ~13,500 | 43.5 | 1.224 |
| 2048 | ~11,500 | 51.9 | 1.206 |
| 4096 | ~8,400 | 71.5 | 1.201 |

A pesar de completar 38% menos pasos, secuencias 4× mas largas mejoran la loss porque cada paso proporciona senales mas ricas.

### 4. Los Learning Rates Default Son Demasiado Altos

Los LR default de Muon (0.04) son excesivos para el regimen de 10 minutos. LR sweeps sistematicos muestran que el optimo esta en ~0.02 (la mitad). Esto tambien mejora la calidad post-cuantizacion porque produce distribuciones de pesos mas suaves.

### 5. La Inicializacion No Es Trivial

Spectral shaping (Overtone init con SVD y ley de potencias) proporciona mejoras ortogonales a todos los demas cambios. Con solo ~13K pasos de entrenamiento, cada ventaja temprana en convergencia se traduce directamente en mejor val_bpb final.

---

## Analisis de Envios: Leaderboard vs Frontera

### El Gap entre SOTA Merged y PRs Abiertos

El leaderboard oficial (records merged) no refleja la frontera real del challenge. Los PRs abiertos han avanzado significativamente:

```
Baseline merged:     1.2244  (9L, int8, zlib)
SOTA merged:         1.1748  (10L, FP16 emb, Muon WD, OvertoneInit, sliding eval)
Frontera PRs:        1.1318  (11L, int6 QAT, SWA, FA3, BigramHash, SmearGate)
                     ──────
Delta merged:        0.0496 BPB  (baseline → SOTA merged)
Delta no-merged:     0.0430 BPB  (SOTA merged → frontera PRs)
Delta total:         0.0926 BPB  (baseline → frontera)
```

El gap no-merged (0.043 BPB) es casi tan grande como todo el progreso merged. Las tecnicas responsables:

| Tecnica | Contribucion estimada | Fuente |
|---------|----------------------|--------|
| Int6 cuantizacion (mas params en 16MB) | ~0.015-0.020 BPB | PG records, PRs #198, #219 |
| QAT/STE + SWA (mejor cuantizacion) | ~0.008-0.010 BPB | PRs #192, #194 |
| MLP 3× + 11L (mas capacidad) | ~0.008-0.010 BPB | PRs #198, #219 |
| SmearGate + BigramHash | ~0.003-0.005 BPB | Speedrun records #34, #62 |
| OrthoInit + muP + WD tuning | ~0.003-0.005 BPB | PRs #198, #206 |

### PRs Frontera (Top 5)

| PR | val_bpb | Tecnicas | Autor |
|----|---------|----------|-------|
| #198 | **1.1318** | 11L Int6 + WD=0.04 + SWA + FA3 + BigramHash + SmearGate + OrthoInit/muP | jfprincz |
| #194 | 1.1480 | 11L Int6 QAT + Per-Dim SmearGate + SWA | baudrillardsgh0st |
| #206 | 1.1507 | Int6 STE + SmearGate + U-Net + RoPE50K + SWA | dexhunter |
| #219 | 1.1541 | 12L Int5-MLP + Int6-Attn mixed quant | alertcat |
| #215 | 1.1548 | 11L Low-Rank Q192 (Q factorizado 512→192→512) | JayCheng113 |

### Stack Consenso Emergente

Casi todo PR competitivo comparte este stack base:

| Componente | Detalle | Origen |
|------------|---------|--------|
| Int6 QAT (STE) | Fake 6-bit quantization durante training | PG PRs |
| zstd-22 | Reemplaza zlib; ~35% mejor compresion para int6 | PG PRs |
| 11 capas / 512 dim | Sweet spot para 16MB con int6 | PG PRs |
| MLP 3× (hidden=1536) | Ratio SwiGLU optimo ~2.7× | [HF blog](https://huggingface.co/blog/codelion/optimal-model-architecture) |
| SmearGate | Gate que mezcla token actual + anterior | [Speedrun record #34](nanogpt-speedrun.md) |
| SWA | Stochastic Weight Averaging en warmdown | [Slowrun](nanogpt-slowrun.md) |
| Muon WD = 0.038-0.04 | WD alto mantiene pesos pequenos para int6 | PG PRs + [Speedrun](nanogpt-speedrun.md) |
| FP16 tied embeddings | No cuantizar embedding/unembedding | PG record FP16Embed |
| Sliding window eval stride=64 | Contexto casi completo para cada token | PG record SlidingWindowEval |
| OrthoInit + muP | Inicializacion ortogonal + maximal update param | PG PRs |
| Seq 2048 + RoPE | 2× contexto de entrenamiento | PG record Seq2048 |

### Hallazgos Sorpresa (No Anticipados)

1. **zstd-22 reemplaza zlib**: Los PRs usan zstd nivel 22 en vez de zlib — comprime ~35% mejor para valores int6 en containers int8.

2. **Low-rank Q factorization** (PR #215): Las matrices Q tienen condition numbers de 100M+ y naturalmente viven en un subespacio low-rank. Factorizar Q (512→192→512) ahorra 25% params por capa y baja step time 22%. K, V, O no deben factorizarse.

3. **Per-dimension SmearGate** (PR #194): Cada dimension del embedding tiene su propio ratio de mezcla aprendido (vs gate escalar). Mejora sobre SmearGate basico.

4. **WD=0.04 es optimo (no 0.02)**: Contraintuitivo — el WD merged era 0.02, pero int6 necesita WD mas alto para mantener pesos en rango cuantizable. La regularizacion y la cuantizacion estan acopladas.

---

## Contraste: Investigacion vs Practica

### Predicciones Correctas

| Prediccion | Fuente | Validacion |
|------------|--------|------------|
| Int6 es el game-changer | [small-model-research.md](small-model-research.md) | Todos los top PRs usan int6 |
| QAT con STE mejora cuantizacion | [small-model-research.md](small-model-research.md) | PRs #192, #194, #206 |
| SWA/EMA mejora pesos para quant | [nanogpt-slowrun.md](nanogpt-slowrun.md) | Consenso en todo PR competitivo |
| SmearGate del speedrun es transferible | [nanogpt-speedrun.md](nanogpt-speedrun.md) | Presente en todos los top PRs |
| 11-12 capas (mas profundidad ayuda bajo D_crit) | [small-model-research.md](small-model-research.md) | 11L es sweet spot, 12L en PR #219 |
| MLP mas ancho (ratio SwiGLU ~2.7×) | [small-model-research.md](small-model-research.md) | MLP 3× (ratio 3) es consenso |
| WD mas alto mejora cuantizacion | [nanogpt-slowrun.md](nanogpt-slowrun.md) | WD 0.038-0.04 es consenso |
| BigramHash del speedrun es transferible | [nanogpt-speedrun.md](nanogpt-speedrun.md) | Presente en PRs #198, #208 |
| MTP no ayuda a esta escala (<1B) | [small-model-research.md](small-model-research.md) | Nadie lo usa |

### No Predichas (Sorpresas)

| Hallazgo | Por que no se predijo |
|----------|----------------------|
| zstd-22 reemplaza zlib | Enfoque en ML, no en compresion de datos |
| Low-rank Q (PR #215) | Analisis de condition number es empirico, no teorico |
| Per-dim SmearGate | Extension natural pero no obvia del SmearGate escalar |
| WD=0.04 > WD=0.02 con int6 | La interaccion WD↔cuantizacion no es intuitiva |
| OrthoInit + muP como combo | muP es de 2022 pero no se menciono en analisis previos |

### Aun No Validadas (Oportunidades)

| Tecnica | Fuente | Estado en PG | Oportunidad |
|---------|--------|-------------|-------------|
| NorMuon + Polar Express | [Speedrun records #38, #41](nanogpt-speedrun.md) | No implementado | Los PRs usan Muon basico — actualizarlo podria mejorar convergencia |
| Value projections desde x0 | [Slowrun record #3](nanogpt-slowrun.md) | No intentado | Proyeccion 512→64 = 32KB, validado en slowrun |
| Per-head attention gating | [Slowrun record #6](nanogpt-slowrun.md) | Sub-explorado | Solo PR #206 usa algo similar (U-Net) |
| RingFormer (shared + low-rank local) | [arXiv:2502.13181](https://arxiv.org/abs/2502.13181) | No intentado | 5× eficiencia de parametros en paper |
| Layer looping (capas intermedias) | [Slowrun records #7, #8](nanogpt-slowrun.md) | Resultados mixtos en PRs | Compite contra "mas params unicos via int6" |
| NoPE-RoPE hybrid | [arXiv:2501.18795](https://arxiv.org/abs/2501.18795) | No explorado | Cero costo, diversifica representaciones |
| BitNet ternario capas medias | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | No explorado | Int5 (PR #219) es el mas agresivo actual |
| ROOT/CANS coeficientes optimos NS | [arXiv:2511.20626](https://arxiv.org/abs/2511.20626) | No explorado | Mejor ortogonalizacion para Muon |

---

## Direcciones de Investigacion y Desarrollo

### Tier 1 — Implementar (validado en PRs y/o challenges hermanos)

| # | Tecnica | Costo | Impacto | Fuente |
|---|---------|-------|---------|--------|
| 1 | **Int6 cuantizacion completa + QAT/STE** | 0 params extra | Muy alto | PRs #192-198 |
| 2 | **SWA en warmdown** (cada 50-100 pasos) | 0 params, ~30MB RAM | Alto | PRs + [Slowrun](nanogpt-slowrun.md) |
| 3 | **11 capas + MLP 3×** (habilitado por int6) | +~5M params (caben en int6) | Alto | PRs #198, #219 |
| 4 | **SmearGate** | ~512 params | Medio | [Speedrun #34](nanogpt-speedrun.md) |
| 5 | **Cautious WD = 0.038-0.04** | 0 params | Medio | PRs + [Speedrun #43](nanogpt-speedrun.md) |
| 6 | **OrthoInit + muP** | 0 params | Medio | PRs #198, #206 |
| 7 | **BigramHash embedding** | ~1-2.5MB | Medio | [Speedrun #62](nanogpt-speedrun.md), PRs |

### Tier 2 — Experimentar (evidencia en challenges hermanos, no validado en PG)

| # | Tecnica | Costo | Impacto esperado | Fuente |
|---|---------|-------|-----------------|--------|
| 8 | **NorMuon + Polar Express** | 0 params | Medio | [Speedrun #38, #41](nanogpt-speedrun.md) |
| 9 | **Value projections desde x0** | ~32KB | Medio | [Slowrun #3](nanogpt-slowrun.md) |
| 10 | **Per-head attention gating** | ~4K params | Medio | [Slowrun #6](nanogpt-slowrun.md) |
| 11 | **U-Net skip connections** | ~10 params | Medio | [Slowrun](nanogpt-slowrun.md), [Speedrun](nanogpt-speedrun.md) |
| 12 | **Partial key offset** | 0 params | Bajo-Medio | [Slowrun tiny #6](nanogpt-slowrun.md) |
| 13 | **Low-rank Q factorization** (Q: 512→192→512) | Ahorra 25%/capa | Medio | PR #215 |
| 14 | **Batch size schedule** (pequeño→grande) | 0 params | Bajo-Medio | [Speedrun #46](nanogpt-speedrun.md) |

### Tier 3 — Investigar (evidencia teorica, alto potencial, alto riesgo)

| # | Tecnica | Potencial | Riesgo | Fuente |
|---|---------|-----------|--------|--------|
| 15 | **RingFormer** (shared + low-rank per-depth) | Muy alto (5× eficiencia) | Alto (compite con int6+mas capas unicas) | [arXiv:2502.13181](https://arxiv.org/abs/2502.13181) |
| 16 | **Int5/Int4 para MLP** | Alto (aun mas params en 16MB) | Medio (calidad a int4 incierta) | PR #219, [small-model-research.md](small-model-research.md) |
| 17 | **BitNet ternario capas medias** | Alto (compresion extrema) | Alto (calidad a 22M ternario) | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) |
| 18 | **NoPE-RoPE hybrid** | Bajo-Medio | Bajo | [arXiv:2501.18795](https://arxiv.org/abs/2501.18795) |
| 19 | **ROOT/CANS coeficientes para NS** | Medio | Medio | [arXiv:2511.20626](https://arxiv.org/abs/2511.20626) |
| 20 | **Depth/width sweep** (12×480 vs 8×576 vs 14×448) | Medio | Bajo (solo hipers) | [Depth Delusion](https://arxiv.org/abs/2601.20994), [MobileLLM](https://arxiv.org/abs/2402.14905) |

### No Recomendado

| Tecnica | Razon | Fuente |
|---------|-------|--------|
| Multi-token prediction | No ayuda bajo 1-3B params (confirmado ICML 2024, BabyLM 2025) | [arXiv:2404.19737](https://arxiv.org/abs/2404.19737) |
| Modelos byte-level | Secuencias 4× mas largas, contraproducente con limite de 10 min | [small-model-research.md](small-model-research.md) |
| MoE con multiples expertos | Router overhead domina a 22M params | [small-model-research.md](small-model-research.md) |
| Dropout | 1 epoca sobre 8B tokens, no hay sobreajuste | [nanogpt-slowrun.md](nanogpt-slowrun.md) |
| Mamba/SSM puro | Ventaja en secuencias largas, no a 1024-2048 tokens con Flash Attention | [small-model-research.md](small-model-research.md) |
| Knowledge distillation online | Requiere teacher dentro del presupuesto de 10 min | [small-model-research.md](small-model-research.md) |

---

## Referencias

### Documentos Internos
- [Analisis NanoGPT Speedrun](nanogpt-speedrun.md) — 77 records, tecnicas de optimizacion L(T)
- [Analisis NanoGPT Slowrun](nanogpt-slowrun.md) — 27 records, tecnicas data-efficient L(D)
- [Investigacion Modelos Sub-100M](small-model-research.md) — papers, arquitecturas, cuantizacion

### Papers Fundamentales
- **Neural Scaling Laws** (Kaplan et al., 2020): [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **Scaling Laws in the Tiny Regime** (Marzo 2026): [arXiv:2603.07365](https://arxiv.org/abs/2603.07365)
- **MobileLLM** (Meta, ICML 2024): [arXiv:2402.14905](https://arxiv.org/abs/2402.14905)
- **The Depth Delusion** (Enero 2026): [arXiv:2601.20994](https://arxiv.org/abs/2601.20994)
- **RingFormer** (Febrero 2025): [arXiv:2502.13181](https://arxiv.org/abs/2502.13181)
- **BitNet b1.58** (2024): [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)

### Optimizadores
- **Muon**: [Blog](https://kellerjordan.github.io/posts/muon/) | [GitHub](https://github.com/KellerJordan/Muon)
- **NorMuon**: [arXiv:2510.05491](https://arxiv.org/abs/2510.05491)
- **ROOT**: [arXiv:2511.20626](https://arxiv.org/abs/2511.20626)
- **CANS**: [arXiv:2506.10935](https://arxiv.org/abs/2506.10935)

### Comunidad
- **NanoGPT Speedrunning**: [GitHub](https://github.com/KellerJordan/modded-nanogpt)
- **NanoGPT Slowrun**: [GitHub](https://github.com/qlabs-eng/slowrun)
- **Parameter Golf Research Garden**: [golf.agustif.com](https://golf.agustif.com/)
- **OpenAI Discord**: #parameter-golf-discussions, #parameter-golf-announcements
- **Compute Grants**: $1M via [formulario](https://openai.com/index/parameter-golf/#credit-form)

---

*Ultima actualizacion: 2026-03-20*
