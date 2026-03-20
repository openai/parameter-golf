# Investigacion: Tecnicas para Modelos Sub-100M

## Nuestro Modelo en Contexto

El modelo de Parameter Golf tiene **~22M parametros** (9 capas, dim=512, vocab=1024, tied embeddings). Esto lo situa en la escala mas pequeña de modelos de lenguaje viables:

| Modelo | Params | Capas | Dim | Vocab | Tokens entrenamiento |
|--------|--------|-------|-----|-------|---------------------|
| **Parameter Golf** | **~22M** | **9** | **512** | **1,024** | **~5.5B (10min 8×H100)** |
| MobileLLM-125M | 125M | 30 | 576 | 32K | 1T |
| SmolLM2-135M | 135M | 30 | 576 | 49K | 2T |
| TinyLlama-1.1B | 1.1B | 22 | 2048 | 32K | 3T |
| Phi-1 | 1.3B | 24 | 2048 | 51K | 7B |

Nuestro modelo es **6× mas pequeño** que los mas pequeños publicados (SmolLM2-135M, MobileLLM-125M). Sin embargo, tiene una ventaja unica: **vocabulario ultra-pequeño** (1024 vs 32-50K tipico), lo que significa que el embedding consume solo ~2.4% de los parametros vs ~20-40% en modelos tipicos. Esto maximiza la capacidad del transformer trunk.

---

## 1. Profundidad vs Anchura: El Gran Debate

### MobileLLM (Meta, ICML 2024): "Deeper is Better"

**Paper**: [arXiv:2402.14905](https://arxiv.org/abs/2402.14905)

A 125M params, un modelo de 30 capas con dim=576 supera significativamente a uno de 12 capas con dim mas grande. MobileLLM-125M usa:
- 30 capas, dim 576, 9 heads, 3 KV heads (GQA 3:1)
- SwiGLU, tied embeddings
- **MobileLLM-LS** (Layer Sharing): pares de bloques consecutivos comparten pesos, ganando 0.7% accuracy gratis

**Conclusion**: Para modelos sub-1B, profundidad > anchura.

### "The Depth Delusion" (Enero 2026): "Wider is Better"

**Paper**: [arXiv:2601.20994](https://arxiv.org/abs/2601.20994)

Validado en 30 arquitecturas (17M–7B, R²=0.922):
- La profundidad optima escala como D* ~ C^0.12, la anchura como W* ~ C^0.34
- La anchura debe crecer **2.8× mas rapido** que la profundidad
- Mas alla de D_crit ~ W^0.44, agregar capas **aumenta** la loss a pesar de agregar parametros
- Mecanismo: "gradient starvation" — decaimiento exponencial del gradiente con la profundidad relativa a la anchura

### Resolucion de la Contradiccion

Ambos papers son correctos en sus regimenes:

- **MobileLLM** compara a param count fijo — mas capas finas vs menos capas anchas. A 125M params, 30 capas × dim=576 gana porque la profundidad permite mas composicion de funciones.
- **Depth Delusion** encuentra un optimo absoluto — dado compute infinito, hay un ratio depth/width ideal. Mas alla de ese ratio, la profundidad perjudica.

**Para nuestro modelo de ~22M**: Con dim=512, D_crit ~ 512^0.44 ≈ 14 capas. Nuestras 9-10 capas estan por debajo del limite critico, asi que **agregar profundidad aun ayuda** (lo que confirma el record int6 de 10 capas). Pero ir a 20+ capas con dim=256 seria arriesgado.

**Configuraciones a explorar**:

| Config | Capas | Dim | Params ~aprox | Dentro de D_crit? |
|--------|-------|-----|---------|-------------------|
| Actual | 9-10 | 512 | 22M | Si (D_crit~14) |
| Deeper-A | 14 | 448 | ~22M | Limite |
| Deeper-B | 12 | 480 | ~22M | Si |
| Wider | 8 | 576 | ~22M | Si |

---

## 2. Parameter Sharing y Depth Recurrence

### MobileLLM-LS: Block-wise Weight Sharing

Bloques consecutivos comparten todos los pesos. A 125M: +0.7% accuracy sin costo en parametros, overhead de latencia marginal.

**Para PG**: Con 10 capas, compartir en pares (0-1, 2-3, ...) daria 5 bloques unicos con 10 capas efectivas. Ahorra ~50% de parametros del trunk, que se pueden reinvertir en mas profundidad o anchura.

### RingFormer (Febrero 2025)

**Paper**: [arXiv:2502.13181](https://arxiv.org/abs/2502.13181)

Un solo bloque transformer compartido se reutiliza en todas las posiciones de profundidad, con "level signals" dependientes de la entrada generados por transformaciones low-rank especificas por profundidad. **Iguala o supera transformers estandar usando solo ~20% de los parametros**.

```
Parametros globales (compartidos): ~4M  (un bloque transformer)
Parametros locales (por profundidad): ~200K  (adaptadores low-rank por capa)
Profundidad efectiva: 20+ capas
Total: ~6M params → rendimiento de ~22M params
```

**Implicacion para PG**: Si un RingFormer de 6M params rinde como 22M, un RingFormer de 22M podria rendir como ~80M. El presupuesto de 16MB se usa para maximizar profundidad efectiva. **Esta es potencialmente la tecnica de mayor impacto.**

### Relaxed Recursive Transformers (Octubre 2024, ICLR 2026)

**Paper**: [arXiv:2410.20672](https://arxiv.org/abs/2410.20672)

Capas unicas se repiten en loop, con adaptadores LoRA por iteracion para diferenciar las pasadas. Recursive Gemma 1B iguala Gemma 2B estandar — **~50% de reduccion de parametros**.

### Transferibilidad: **MUY ALTA**

| Tecnica | Costo en params | Ganancia esperada | Dificultad |
|---------|----------------|-------------------|------------|
| Block-wise sharing (MobileLLM-LS) | 0 (ahorra 50%) | +0.7% | Baja |
| RingFormer (shared + low-rank local) | Ahorra ~80% | Potencialmente masiva | Media-Alta |
| Recursive + LoRA adapters | Ahorra ~40-50% | Alta | Media |

---

## 3. Estado Actual del Challenge (Marzo 2026)

### Frontera de PRs Abiertos

Los PRs mas recientes estan empujando val_bpb significativamente por debajo del SOTA merged (1.1748):

| PR | val_bpb | Tecnicas Clave |
|----|---------|---------------|
| #198 | **1.1318** | 11L Int6 + WD=0.04 + SWA + FA3 |
| #194 | 1.1480 | 11L Int6 QAT + Per-Dim SmearGate + SWA |
| #192 | 1.1502 | 11L Int6 QAT + SmearGate + WD 0.038 |
| #206 | 1.1507 | Int6 STE + SmearGate + OrthoInit + SWA |
| #219 | 1.1541 | 12L Int5-MLP + Int6-Attn mixed quant |
| #208 | 1.1568 | Int6 MLP3x + SmearGate + BigramHash + SWA + DocSliding |

### Tecnicas Emergentes en el Challenge

1. **Int6 cuantizacion** (step=4 rounding): Permite ~1.5× mas parametros en 16MB. Habilita 11-12 capas y/o MLPs mas anchas (3× en vez de 2×).
2. **QAT con STE** (Straight-Through Estimator): Entrenar con ruido de cuantizacion in-the-loop. Los pesos aprenden a ser cuantizacion-friendly.
3. **SmearGate**: Gate aprendido que desplaza embeddings 1 posicion (del speedrun).
4. **SWA** (Stochastic Weight Averaging): Promedia pesos durante las ultimas epocas para mejor generalizacion y distribucion de pesos mas suave.
5. **Flash Attention 3**: Mayor throughput, mas pasos en 10 minutos.
6. **BigramHash**: Embeddings hash-based de bigramas (del speedrun).
7. **OrthoInit**: Inicializacion ortogonal de pesos.
8. **DocSliding**: Evaluacion con ventana deslizante por documento.
9. **Muon Weight Decay ~0.038**: WD mas agresivo que el baseline.

**Insight critico**: Int6 es el game-changer — al comprimir los pesos mas agresivamente, se liberan ~4MB que se traducen en capas adicionales y MLPs mas anchos. La combinacion Int6 + QAT + SWA es el stack dominante.

---

## 4. Cuantizacion Agresiva: Int6, Int4, BitNet

### Int6 (Ya Validado en PG)

El record int6 mixed precision usa 64 niveles (vs 256 de int8) para capas intermedias:
- Ahorra ~1.6MB por capa de 512×512
- Penalizacion de cuantizacion: solo 0.0018 BPB (vs 0.0093 en int8 baseline)
- Habilitador para 10-12 capas dentro de 16MB

### QAT con STE

**Papers**: [PyTorch QAT Blog](https://pytorch.org/blog/quantization-aware-training/) | [arXiv:2411.02530](https://arxiv.org/abs/2411.02530)

```python
# Forward pass con fake quantization:
w_quant = fake_quantize(w, num_bits=6)  # Cuantiza y descuantiza
output = F.linear(x, w_quant)

# Backward pass: STE ignora la cuantizacion
# Gradientes fluyen como si no hubiera cuantizacion
```

QAT recupera hasta 96% de la degradacion de accuracy vs PTQ (post-training quantization). Para PG, esto significaria:
- Entrenar con simulacion de int6 en el loop
- Los pesos convergen a distribuciones que cuantizan mejor
- La penalizacion post-quant baja de ~0.007 a ~0.001 BPB

### BitNet b1.58 (Pesos Ternarios)

**Paper**: [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)

Pesos restringidos a {-1, 0, +1} usando cuantizacion absmean:
```python
w_ternary = round(w / mean(|w|))  # Clamp a {-1, 0, +1}
# Cada peso necesita solo 1.58 bits
```

A 3B: iguala FP16 LLaMA en perplexity. A 100M: funcional pero con degradacion notable.

**Para PG**: Un modelo ternario de 22M comprimiria a ~4.3MB (vs ~22MB int8). Eso deja ~11.7MB para **mas parametros** — potencialmente un modelo de ~60M params ternario en 16MB. Pero la calidad a 22M con pesos ternarios es cuestionable.

**Hibrido prometedor**: Ternario para capas intermedias, int8 para primera/ultima capa y embedding. Similar al mixed int6/int8 actual pero mas agresivo.

### Int5 Mixed Quantization (PR #219)

El PR #219 explora int5 para MLP + int6 para atencion. A 12 capas, logra val_bpb=1.1541. Esto sugiere que el MLP tolera cuantizacion mas agresiva que la atencion.

### Transferibilidad: **MUY ALTA**

| Tecnica | Estado en PG | Impacto |
|---------|-------------|---------|
| Int6 capas medias | Ya validado (record) | Alto |
| QAT con STE | En PRs abiertos | Alto |
| Int5 para MLP | PR #219 experimental | Medio-Alto |
| BitNet ternario | No explorado | Potencialmente muy alto |
| Mixed ternario/int8 | No explorado | Alto potencial |

---

## 5. Optimizador: Mas Alla de Muon Basico

### NorMuon (Octubre 2025)

**Paper**: [arXiv:2510.05491](https://arxiv.org/abs/2510.05491)

Normaliza los updates de Muon, permitiendo LR mas altos y convergencia mas rapida. Ya usado en modded-nanogpt (record #41+).

### Polar Express (Septiembre 2025)

Reemplaza la iteracion Newton-Schulz con formula precomputada, mas estable en bfloat16. Ya en modded-nanogpt (record #38).

### ROOT Optimizer (Noviembre 2025)

**Paper**: [arXiv:2511.20626](https://arxiv.org/abs/2511.20626)

Reemplaza los coeficientes fijos de Newton-Schulz con coeficientes adaptativos por dimension. Superior a Muon estandar en convergencia.

### CANS — Chebyshev-Optimized Newton-Schulz (Junio 2025)

**Paper**: [arXiv:2506.10935](https://arxiv.org/abs/2506.10935)

Coeficientes teoricamente optimos via teorema de alternancia de Chebyshev. Mejor ortogonalizacion para paisajes anisotropicos.

### IFNSO — Iteration-Free Newton-Schulz (Febrero 2026)

**Paper**: [arXiv:2602.02500](https://arxiv.org/abs/2602.02500)

Consolida la estructura iterativa en un solo polinomio con coeficientes aprendidos. Elimina el loop interno de Newton-Schulz.

### Cautious Weight Decay (del Speedrun)

Solo aplica weight decay cuando el gradiente y el parametro tienen el mismo signo:
```python
mask = (grad * param > 0).float()
param -= lr * wd * mask * param
```

Ya demostrado en speedrun (record #43, #50). Muon WD schedule en PG.

### Transferibilidad: **ALTA**

| Optimizador | Mejora sobre Muon basico | Dificultad | Estado en PG |
|-------------|-------------------------|------------|-------------|
| NorMuon | Mejor convergencia | Media | No implementado |
| Polar Express | Estabilidad numerica | Media | No implementado |
| Cautious WD | Mejor regularizacion | Baja | Parcial (WD=0.02) |
| ROOT/CANS | Coeficientes optimos | Media | No implementado |
| IFNSO | Elimina loop NS | Media | No implementado |

---

## 6. Multi-Token Prediction: NO Recomendado

**Paper**: Gloeckle et al., [arXiv:2404.19737](https://arxiv.org/abs/2404.19737) (ICML 2024)

Resultado clave: **MTP no mejora modelos por debajo de 1-3B parametros**. El beneficio solo aparece por encima de un umbral de capacidad. Confirmado en BabyLM 2025 con datasets de 10M tokens.

A 22M params, MTP anade overhead computacional sin beneficio en calidad. El speedrun lo usa a 124M (donde si ayuda), pero no es transferible a nuestra escala.

**Veredicto: No implementar.**

---

## 7. Low-Rank FFN

**Paper**: [arXiv:2407.09835](https://arxiv.org/abs/2407.09835) (ICML 2024 Workshop)

Parametrizacion low-rank de capas FFN con 32% de parametros da solo 1 PPL de aumento a 1.3B. Curvas de scaling mas pronunciadas — modelos low-rank mejoran mas rapido con mas datos.

```python
# FFN estandar: W1 (dim × 4*dim), W2 (4*dim × dim)
# FFN low-rank: W1 = A @ B, donde A (dim × r), B (r × 4*dim)
# Ahorro: r < dim (tipicamente r = dim/4)
```

**Para PG**: Con SwiGLU, el MLP tiene 3 matrices de 512×1024. Low-rank con r=128:
- Estandar: 3 × 512 × 1024 = 1.57M params/capa
- Low-rank: 3 × (512×128 + 128×1024) = 0.59M params/capa (ahorro 62%)
- Los params ahorrados pueden reinvertirse en mas capas

**Transferibilidad**: **MEDIA-ALTA**. El ahorro depende del rank — demasiado bajo degrada calidad, demasiado alto no ahorra suficiente.

---

## 8. Byte-Level Models

### EvaByte (2025)
Primer modelo byte-level open-source que iguala modelos basados en tokenizador. Entrenado en 1.5T bytes.

### ByteFlow (Marzo 2026)
Opera en bytes crudos con objetivo information-theoretic para identificar unidades significativas al vuelo.

**Para PG**: Con vocab=1024, nuestro tokenizador ya es muy eficiente. Un modelo byte-level (vocab=256) ahorraria embedding params pero requeriria secuencias ~4× mas largas para cubrir el mismo texto. Con el limite de 10 minutos, esto es contraproducente. **No recomendado.**

---

## 9. Curriculum Learning y Data Ordering

**Paper**: "Curriculum Learning for LLM Pretraining" (Enero 2026, [arXiv:2601.21698](https://arxiv.org/pdf/2601.21698))

- Ordenamiento easy-to-hard: +5-8% en tareas de razonamiento para modelos 1.5B/30B tokens
- **AutoCurriculum** (Google, Nov 2025): RL para ajustar data mixtures durante entrenamiento, +9.3% en razonamiento complejo
- Ordenamiento inverso (hard-to-easy) a veces gana para modelos mas capaces

**Para PG**: Los datos son FineWeb pre-procesados en shards. Reordenar shards por dificultad (medida por perplexity de un modelo pequeño pre-entrenado) podria ayudar, pero requeriria pre-procesamiento offline. Con solo ~13K pasos, el impacto del curriculum es incierto.

**Transferibilidad**: **BAJA-MEDIA**. Requiere pre-procesamiento de datos que no esta en el scope inmediato.

---

## 10. Layer-wise Learning Rates

**Paper**: "Layerwise Learning Rate in the Era of Large Language Models" (ICLR 2025)

Usa teoria HT-SR (Heavy-Tailed Self-Regularization) para asignar LR por capa:
- Capas con heavy-tailedness debil → LR mas alto
- Capas con heavy-tailedness fuerte → LR mas bajo

Alternativa simple: LR decreciente con la profundidad (capas tempranas aprenden features basicas rapido, capas profundas necesitan ajuste fino).

**Para PG**: El codebase ya tiene LR separados (MATRIX_LR, SCALAR_LR, EMBED_LR, TIED_EMBED_LR). Agregar LR por capa seria una extension natural. Con Muon, cada capa podria tener un multiplicador de LR.

**Transferibilidad**: **MEDIA**. Implementacion simple, impacto incierto a 22M params.

---

## 11. Architectural Details Optimos

### FFN Ratio con SwiGLU

SwiGLU tiene gating que duplica la transformacion, por lo que el ratio optimo es **~2.7×** (no 4×). Nuestro modelo usa `mlp_mult=2`, lo que da FFN hidden=1024 (ratio 2×). Los PRs con Int6 usan MLP 3× (ratio 3, hidden=1536) — esto esta mas cerca del optimo.

### GQA Ratio

A escala pequeña, MobileLLM usa 9 heads / 3 KV heads (ratio 3:1). Nuestro modelo usa 8 heads / 4 KV heads (ratio 2:1). Reducir a 2 KV heads (ratio 4:1) ahorraria parametros de KV projection con impacto minimo.

### Normalizacion

QK-norm + logit softcap permite LR 1.5× mas altos sin divergencia. Ya implementado (logit softcap=30). Explorar cap mas bajo (15-23, como en speedrun) con LR mas alto.

### NoPE-RoPE Hybrid

**Paper**: [arXiv:2501.18795](https://arxiv.org/html/2501.18795v1)

Intercalar capas con y sin codificacion posicional:
- Capas NoPE aprenden representaciones position-agnostic
- Capas RoPE manejan relaciones posicionales
- 4× reduccion en KV cache, mejor generalizacion

**Para PG**: Con solo 9-10 capas, hacer 5 NoPE + 5 RoPE podria mejorar la diversidad de representaciones sin costo en parametros.

---

## 12. Stochastic Weight Averaging (SWA) y EMA

### SWA
Promedia pesos en multiples puntos del entrenamiento (tipicamente en la fase de warmdown). Produce distribuciones de pesos mas suaves que cuantizan mejor.

### EMA (del Slowrun)
Mantener shadow weights con decay=0.95, blend 70/30 con pesos finales. **Especialmente valioso para cuantizacion** — menos outliers en la distribucion de pesos.

**Para PG**: SWA ya aparece en los PRs abiertos mas competitivos. Implementar SWA o EMA en la fase de warmdown podria ser el "quick win" mas facil para mejorar calidad post-cuantizacion.

**Transferibilidad**: **MUY ALTA**. Cero parametros extra, mejora cuantizacion directamente.

---

## 13. Depth Recurrence en PG: Resultados Mixtos

Varios PRs de PG han explorado depth recurrence (#5, #8, #11, #15, #29, #30, #213) con **resultados mixtos**. La razon: con 16MB de presupuesto, la recurrencia compite contra simplemente tener mas parametros unicos (via int6).

**El trade-off**:
- Sin recurrencia, int6: ~22M params unicos, 11-12 capas
- Con recurrencia: ~11M params unicos, 20+ capas efectivas (via looping)
- ¿Cual tiene mejor loss? Depende de si la profundidad efectiva compensa la reduccion en capacidad por capa

**Cuando la recurrencia gana**:
- Modelos donde la profundidad esta por debajo de D_crit (si, nuestro caso)
- Con adaptadores low-rank por iteracion (RingFormer) para diferenciar pasadas
- Cuando el overhead de compute por paso no reduce demasiado los pasos totales

**Cuando pierde**:
- Si int6/int5 ya libera suficiente espacio para mas capas unicas
- Si el modelo es tan pequeño que cada parametro unico importa mas que la profundidad

---

## Sintesis: Prioridades de Investigacion

### Tier 1 — Implementar Inmediatamente (alta confianza, bajo riesgo)

| # | Tecnica | Fuente | Impacto | Esfuerzo |
|---|---------|--------|---------|----------|
| 1 | **Int6 cuantizacion** (todo el modelo) | PG records + PRs | Mas capas/params en 16MB | Medio |
| 2 | **QAT con STE** | PRs #192, #194 | Menor penalizacion quant | Medio |
| 3 | **SWA / EMA** en warmdown | Slowrun + PRs | Mejor cuantizacion | Bajo |
| 4 | **SmearGate** | Speedrun record #34 | Info local sin atencion | Bajo |
| 5 | **Cautious Weight Decay** | Speedrun record #43 | Mejor regularizacion | Bajo |
| 6 | **11-12 capas** (habilitado por int6) | PG PRs abiertos | Mas profundidad | Bajo |

### Tier 2 — Experimentar (evidencia prometedora, riesgo moderado)

| # | Tecnica | Fuente | Impacto | Esfuerzo |
|---|---------|--------|---------|----------|
| 7 | **MLP 3×** (SwiGLU ratio optimo) | HF blog + PRs | Ratio FFN optimo | Bajo |
| 8 | **NorMuon + Polar Express** | Speedrun records | Mejor convergencia | Medio |
| 9 | **Block-wise weight sharing** (pares) | MobileLLM | Ahorro 50% → mas capas | Medio |
| 10 | **Value projections desde x0** | Slowrun record #3 | Mejor atencion, pocos params | Medio |
| 11 | **Per-head attention gating** | Slowrun record #6 | Atencion adaptativa | Bajo |
| 12 | **BigramHash embedding** | Speedrun record #62 | Info bigramica barata | Medio |

### Tier 3 — Investigar (alto potencial, alto riesgo)

| # | Tecnica | Fuente | Impacto | Esfuerzo |
|---|---------|--------|---------|----------|
| 13 | **RingFormer** (shared + low-rank local) | arXiv 2502.13181 | Potencialmente transformativo | Alto |
| 14 | **Int5/Int4 para MLP** | PR #219 | Aun mas params en 16MB | Medio |
| 15 | **NoPE-RoPE hybrid** | arXiv 2501.18795 | Diversidad de representacion | Medio |
| 16 | **BitNet ternario para capas medias** | arXiv 2402.17764 | Compresion extrema | Alto |
| 17 | **Depth/width exploration** (12×480 vs 8×576) | Depth Delusion + MobileLLM | Ratio optimo | Bajo (solo hipers) |
| 18 | **ROOT/CANS coeficientes** optimizados para NS | arXiv 2511.20626 | Mejor ortogonalizacion | Medio |

### NO Recomendado

| Tecnica | Razon |
|---------|-------|
| Multi-token prediction | No ayuda bajo 1-3B params |
| Modelos byte-level | Secuencias 4× mas largas, contraproducente |
| MoE con multiples expertos | Overhead de routing a 22M params |
| Mamba/SSM | Ventaja en secuencias largas, no en 1024 tokens |
| Knowledge distillation online | Requiere teacher en presupuesto de 10 min |
| Dropout | 1 epoca, no hay sobreajuste |

---

## Ruta Critica Sugerida

Basado en lo que los PRs abiertos ya estan demostrando, el stack ganador parece ser:

```
Int6 cuantizacion (todo el modelo)
+ QAT con STE (entrenar con ruido de cuantizacion)
+ SWA/EMA en warmdown (pesos suaves para cuantizacion)
+ 11-12 capas (habilitado por int6)
+ MLP 3× (ratio SwiGLU optimo)
+ SmearGate + Cautious WD
+ NorMuon con Polar Express
+ Sliding window eval + Doc-sliding
+ FP16 tied embeddings
```

Esto ya esta llevando a val_bpb ~1.13 en PRs abiertos. Para ir mas alla:

```
+ RingFormer (shared base + low-rank per-depth) → profundidad efectiva 20+
+ Int5 para MLP + Int6 para atencion (mixed quant mas agresivo)
+ Value projections desde x0 + per-head gating
+ BigramHash para info local
```

---

## Referencias

### Papers Fundamentales
- [MobileLLM](https://arxiv.org/abs/2402.14905) — Meta, ICML 2024
- [SmolLM2](https://huggingface.co/papers/2502.02737) — HuggingFace, 2025
- [The Depth Delusion](https://arxiv.org/abs/2601.20994) — Enero 2026
- [RingFormer](https://arxiv.org/abs/2502.13181) — Febrero 2025
- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) — ICLR 2026
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — 2024
- [Multi-token Prediction](https://arxiv.org/abs/2404.19737) — ICML 2024
- [Low-Rank Training in Transformers](https://arxiv.org/abs/2407.09835) — ICML 2024 Workshop
- [Scaling Laws](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020
- [Scaling Laws in the Tiny Regime](https://arxiv.org/abs/2603.07365) — Marzo 2026

### Optimizadores
- [Muon](https://kellerjordan.github.io/posts/muon/) — Keller Jordan
- [NorMuon](https://arxiv.org/abs/2510.05491) — Octubre 2025
- [ROOT](https://arxiv.org/abs/2511.20626) — Noviembre 2025
- [CANS](https://arxiv.org/abs/2506.10935) — Junio 2025
- [IFNSO](https://arxiv.org/abs/2602.02500) — Febrero 2026

### Cuantizacion
- [PyTorch QAT](https://pytorch.org/blog/quantization-aware-training/)
- [QAT Comprehensive Study](https://arxiv.org/abs/2411.02530) — 2024
- [Continual QAT for BitNet](https://arxiv.org/abs/2502.11895) — 2025

### Arquitectura
- [NoPE-RoPE Hybrid](https://arxiv.org/abs/2501.18795) — Enero 2025
- [Optimal Architecture for Small LMs](https://huggingface.co/blog/codelion/optimal-model-architecture) — HuggingFace
- [Vocabulary Scaling Laws](https://arxiv.org/abs/2407.13623) — NeurIPS 2024

### Parameter Golf
- [GitHub Repository](https://github.com/openai/parameter-golf/)
- [DeepWiki Overview](https://deepwiki.com/openai/parameter-golf)
- [Research Garden](https://golf.agustif.com/)

---

*Ultima actualizacion: 2026-03-20*
