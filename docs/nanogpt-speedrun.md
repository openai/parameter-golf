# NanoGPT Speedrun — Analisis de Mejoras Incrementales

## Contexto

El [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) es el challenge hermano de Parameter Golf. Mientras Parameter Golf optimiza **L(N)** (menor loss dado parametros fijos), el speedrun optimiza **L(T)** (menor tiempo para alcanzar una loss fija de 3.28 en FineWeb).

Ambos challenges comparten el mismo dataset (FineWeb), hardware (8×H100), y codebase base. Las tecnicas que funcionan en uno frecuentemente son transferibles al otro, aunque con trade-offs distintos:

| | NanoGPT Speedrun | Parameter Golf |
|---|---|---|
| **Objetivo** | Alcanzar val_loss ≤ 3.28 lo mas rapido posible | Minimizar val_bpb con artifact ≤ 16MB |
| **Restriccion principal** | Tiempo (actualmente ~1.4 min) | Tamano de modelo (16MB comprimido) |
| **Modelo** | ~124M params (GPT-2 small scale) | ~15M params (vocab 1024, dim 512) |
| **Vocabulario** | 50,304 (GPT-2 tokenizer) | 1,024 (BPE custom) |
| **Optimizador** | NorMuon + Adam | Muon + Adam |

**Implicacion clave**: El speedrun tiene ~8× mas parametros y un presupuesto de compute ilimitado por paso. Las tecnicas que funcionan ahi deben evaluarse cuidadosamente para su transferencia a Parameter Golf, donde cada byte cuenta.

---

## Progresion Historica: De 45 Minutos a 1.4 Minutos

77 records en ~22 meses. La progresion revela patrones claros sobre que tipos de mejoras dan los mayores saltos:

### Fase 1: Fundamentos (Records 1-10, 45min → 7.8min, ~6× speedup)

| Record | Tiempo | Tecnica | Categoria |
|--------|--------|---------|-----------|
| 1 | 45.0 min | llm.c baseline (GPT-2 replication) | Baseline |
| 2 | 31.4 min | LR tuning + Rotary Embeddings | Arquitectura + Hiper |
| 3 | 24.9 min | **Muon optimizer** | Optimizador |
| 4 | 22.3 min | Muon improvements | Optimizador |
| 5 | 15.2 min | Pad embed, ReLU², zero-init projections, QK-norm | Arquitectura |
| 6 | 13.1 min | Distributed Muon overhead | Sistemas |
| 7 | 12.0 min | Upgrade PyTorch 2.5 | Sistemas |
| 8 | 10.8 min | Untied embedding/head | Arquitectura |
| 9 | 8.2 min | Value + embed skip connections, momentum warmup, softcap | Arquitectura |
| 10 | 7.8 min | BFloat16 activations | Precision |

**Insight**: Los fundamentos — optimizador (Muon), arquitectura moderna (RoPE, ReLU², QK-norm), y skip connections — proporcionan el 83% de la mejora total. Estas son las tecnicas "obvias" que toda implementacion deberia incluir.

### Fase 2: Atencion y Contexto (Records 11-20, 7.8min → 2.99min, ~2.6× speedup)

| Record | Tiempo | Tecnica | Categoria |
|--------|--------|---------|-----------|
| 11 | 7.2 min | U-net skip connections + doble LR | Arquitectura |
| 12 | 5.03 min | **FlexAttention 64K contexto** | Atencion |
| 13 | 4.66 min | Attention window warmup | Schedule |
| 14 | 4.41 min | **Value Embeddings** | Arquitectura |
| 15 | 3.95 min | U-net value embeddings + optimizaciones | Arquitectura |
| 16 | 3.80 min | Split value embeds, block sliding window | Atencion |
| 17 | 3.57 min | Sparsify value embeds, mejorar RoPE | Arquitectura |
| 18 | 3.40 min | Logit softcap 30→15 | Estabilizacion |
| 19 | 3.14 min | FP8 head, offset logits, lr decay a 0.1 | Precision + Schedule |
| 20 | 2.99 min | Merged QKV, long-short attention, batched Muon | Sistemas + Atencion |

**Insight**: FlexAttention (record #12) es el salto mas grande individual (7.2→5.03 min, -30%). La sliding window attention con ventanas variables por capa y warmup progresivo es un patron extremadamente poderoso.

### Fase 3: Refinamiento Algoritmico (Records 21-40, 2.99min → 2.36min)

| Record | Tiempo | Tecnica | Categoria |
|--------|--------|---------|-----------|
| 26 | 2.86 min | Alinear batch con EoS | Datos |
| 27 | 2.82 min | Triton kernel para matmul simetrico | Kernels |
| 28 | 2.81 min | **Sparse attention gate** | Atencion |
| 29 | 2.73 min | Flash Attention 3, max_doc_len=2048 | Atencion |
| 31 | 2.66 min | **YaRN durante entrenamiento** | Posicion |
| 34 | 2.55 min | **Smear module** (token shift 1 posicion) | Arquitectura |
| 38 | 2.48 min | **Polar Express** (reemplazo Newton-Schulz) | Optimizador |
| 40 | 2.36 min | **Backout** + hyperparameter tuning | Arquitectura |

**Insight**: Cada mejora individual es pequeña (~2-5%) pero se componen. Las innovaciones clave son: sparse attention gating, YaRN para extender contexto, smear para induccion de 1-token, y Polar Express como alternativa numerica a Newton-Schulz.

### Fase 4: Optimizador Avanzado + MTP (Records 41-60, 2.36min → 1.77min)

| Record | Tiempo | Tecnica | Categoria |
|--------|--------|---------|-----------|
| 41 | 2.35 min | **NorMuon** | Optimizador |
| 43 | 2.28 min | **Cautious Weight Decay** con schedule | Optimizador |
| 46 | 2.20 min | **Batch size schedule** | Schedule |
| 49 | 2.15 min | **Partial Key Offset** | Atencion |
| 51 | 2.08 min | Retie embed a lm_head | Arquitectura |
| 53 | 1.99 min | **Multi-token prediction** + untie a 2/3 | Objetivo |
| 54 | 1.94 min | Asymmetric logit rescale | Estabilizacion |
| 58 | 1.82 min | **Paired Head Attention** | Atencion |

**Insight**: Multi-token prediction (record #53) cruza la barrera de 2 minutos. NorMuon normaliza los updates de Muon y permite LR mas altos. El batch size schedule (pequeño→grande) mejora convergencia temprana.

### Fase 5: Kernels y Representaciones (Records 61-77, 1.77min → 1.44min)

| Record | Tiempo | Tecnica | Categoria |
|--------|--------|---------|-----------|
| 62 | 1.66 min | **Bigram Hash Embedding** | Representacion |
| 59 | 1.78 min | Fused Triton kernel para ReLU² MLP | Kernels |
| 60 | 1.77 min | Fused softcapped cross-entropy kernel | Kernels |
| 72 | 1.50 min | **Max seq length schedule** | Schedule |
| 73 | 1.49 min | **Partitioned Hyperconnections** | Arquitectura |
| 77 | 1.44 min | Simplificar hyperconnections a single saved activation | Arquitectura |

**Insight**: Los Triton kernels fusionados (ReLU² MLP, softcapped CE) liberan tiempo para mas pasos. Bigram hash embedding aporta informacion de bigramas de forma eficiente.

---

## Tecnicas Detalladas y Transferibilidad a Parameter Golf

### 1. Muon / NorMuon / Polar Express

**Que es**: Optimizador que usa ortogonalizacion de gradientes (descomposicion polar) en lugar de adaptacion diagonal (Adam). NorMuon anade normalizacion del update. Polar Express reemplaza la iteracion Newton-Schulz con una formula precomputada mas estable en bfloat16.

**Como funciona**:
```
g = gradient
g = g + momentum * g_prev                    # Nesterov momentum
g = polar_express(g)                          # Ortogonalizar: g → U (de SVD g=UΣV^T)
param -= lr * g + cautious_wd * param         # Update con weight decay cauteloso
```

**Polar Express** (reemplazo de Newton-Schulz):
- Para matrices altas (H > W): Computa X^T·X, aplica polinomio de grado 5
- Para matrices anchas (W > H): Computa X·X^T, aplica polinomio
- Precomputa coeficientes para 5 iteraciones de convergencia
- Mas estable que Newton-Schulz en bfloat16

**Transferibilidad a Parameter Golf**: **ALTA**. Ya esta implementado parcialmente (Muon sin NorMuon). Actualizar a NorMuon y Polar Express podria mejorar convergencia. El weight decay cauteloso (solo aplica cuando gradient y parametro tienen el mismo signo) es directamente transferible.

**Cautious Weight Decay**:
```python
# Solo decae si gradient "esta de acuerdo" con el parametro
mask = (grad * param > 0).float()
param -= lr * wd * mask * param
```

---

### 2. Value Embeddings

**Que es**: Embeddings adicionales que se inyectan en los valores de atencion, no en los queries/keys. Cada token tiene un "valor inherente" independiente del contexto.

**Como funciona**:
```
V = W_v @ x + gate * value_embed[token_id]   # Gate es learnable por head
```

- 5 value embedding matrices compartidas entre capas selectas (patron U-net: capas 1,2,8,9,10)
- Gated con parametros learnable por head (ve_gate_bank)
- Inicializacion esferica (0.01 × randn)

**En modded-nanogpt**: 5 × 50304 × 768 = ~193M parametros extra (!) — viable con 124M params base, impensable en Parameter Golf con vocab 1024.

**Transferibilidad a Parameter Golf**: **MEDIA**. Con vocab=1024 y dim=512, cada value embedding seria solo 1024×512 = 512KB. Cabrian 2-3 facilmente en el presupuesto de 16MB. Sin embargo, la ganancia depende de tener suficiente capacidad en las otras capas para aprovechar la informacion adicional. **Vale la pena experimentar con 1-2 value embeddings en las capas finales.**

---

### 3. Sliding Window Attention + Schedule

**Que es**: Atencion causal donde cada token solo atiende a los ultimos W tokens (ventana), no a toda la secuencia. Con schedule: las ventanas crecen durante el entrenamiento.

**Schedule de modded-nanogpt**:
```
Stage 1: short=128,  long=384   tokens (context limitado, pasos rapidos)
Stage 2: short=384,  long=896   tokens
Stage 3: short=640,  long=1408  tokens
Extension: short=768, long=1664 tokens (ventanas finales)
```

- Diferentes capas usan short vs long windows (alternadas)
- Evaluacion usa ventanas extendidas respecto a entrenamiento

**Transferibilidad a Parameter Golf**: **ALTA**. El baseline ya usa contexto completo (1024 tokens). Con secuencias mas largas (2048/4096), sliding window permitiria entrenar con contexto largo sin el costo O(n²) completo. El schedule progresivo (ventanas pequeñas al inicio, grandes al final) es directamente aplicable.

---

### 4. Bigram Hash Embedding

**Que es**: Un embedding que codifica pares de tokens consecutivos (bigramas) usando hashing.

**Como funciona**:
```python
# Para cada posicion, computa hash del bigrama (token actual, token previo)
hash = (rand1 * current_token) ^ (rand2 * prev_token) % vocab_size
bigram_embed = embedding_table[hash]
x = x + lambda * bigram_embed  # Se suma al residual stream
```

- Tabla de embedding: vocab_size × 5 entries (hash con 5 "buckets")
- Comunicacion sparse: solo se actualizan las filas accedidas
- Aporta informacion local (que token vino antes) sin atencion

**Transferibilidad a Parameter Golf**: **MEDIA-ALTA**. Con vocab=1024, la tabla seria 1024×5×512 = 2.5MB — significativo pero viable. El beneficio es que aporta informacion bigramica "gratis" (sin capa de atencion). Podria ser especialmente valioso con pocas capas. **La preocupacion es cuantizacion**: los embeddings de bigramas tendrian que cuantizarse bien a int8.

---

### 5. Smear Module

**Que es**: Desplaza los embeddings normalizados 1 posicion hacia adelante con un gate aprendido. Permite al modelo "mirar" el token inmediatamente siguiente de forma implicita.

**Como funciona**:
```python
x_norm = norm(x)
x_shift = roll(x_norm, 1)           # Shift tokens 1 posicion
x_shift[0] = 0                       # Primer token no tiene predecesor
gate = sigmoid(linear(x_norm))       # Gate 12→1, zero-init
x = x + gate * x_shift
```

- Gate inicializado a cero (empieza como no-op, se activa gradualmente)
- LR multiplicador bajo (0.01) para estabilidad
- Colocado antes de las capas transformer

**Transferibilidad a Parameter Golf**: **ALTA**. Costo minimo en parametros (solo un linear 512→1 = 512 params + bias). Proporciona una forma barata de induccion local sin capa de atencion adicional. **Implementacion trivial, alto valor esperado.**

---

### 6. Multi-Token Prediction (MTP)

**Que es**: En lugar de predecir solo el siguiente token, predecir los siguientes 2-3 tokens simultaneamente. Las perdidas se ponderan y decaen durante entrenamiento.

**Schedule de modded-nanogpt**:
```
Etapa 1 (1/3 entrenamiento): predecir tokens +1, +2, +3 con pesos [1.0, 0.5, 0.25]
Etapa 2 (1/3):               predecir tokens +1, +2 con pesos [1.0, 0.5]
Etapa 3 (1/3):               predecir solo token +1 con peso [1.0]
```

**Beneficio**: Gradientes mas ricos por paso. Predecir tokens futuros fuerza al modelo a aprender representaciones mas profundas del contexto.

**Transferibilidad a Parameter Golf**: **MEDIA**. MTP requiere un lm_head mas grande o compartido, pero con tied embeddings y vocab=1024, el overhead es minimo. El schedule decreciente (reducir MTP en las etapas finales) es crucial — se quiere que las etapas finales optimicen exactamente la metrica de evaluacion (next-token prediction). **Potencialmente valioso dado que Parameter Golf tiene pocos pasos (~13K).**

---

### 7. Skip Connections (Patron U-Net)

**Que es**: Conexiones residuales que saltan multiples capas, siguiendo un patron similar a U-Net.

**Patron en modded-nanogpt**:
```
Capa 0 → Capa 1 → Capa 2 → Capa 3 → Capa 4 → Capa 5 → ... → Capa 10
                                ↓                   ↑
                                └── skip_connection ─┘
                                    (capa 3 → capa 6, gated)
```

- Capa 3 guarda su estado como `skip_connection`
- Capa 6 **omite atencion completamente** y usa el skip: `x += skip_gate * skip_3`
- Esto crea un "shortcut" profundo y ahorra el costo de una capa de atencion

**Backout**:
```
x_backout = estado de capa 7
output = x - backout_lambda * x_backout
```
Permite al modelo "deshacer" contribuciones de capas intermedias antes de la prediccion final.

**Transferibilidad a Parameter Golf**: **ALTA**. Con solo 9-10 capas, un skip de capa 3→6 ahorria una capa de atencion manteniendo profundidad efectiva. Backout es un parametro escalar, costo nulo. **El record SOTA de Parameter Golf ya usa residual mixing (phase-transition), pero no skip U-net ni backout.**

---

### 8. Paired Head Attention

**Que es**: Pares de heads de atencion que comparten keys pero con offsets, permitiendo patrones de atencion complementarios.

**Como funciona**:
- Capas seleccionadas (0, 2, 5, 9) usan paired-head attention
- Heads adyacentes (0-1, 2-3, 4-5) comparten la misma key
- Un head atiende a posiciones pares, el otro a impares (efecto de duplicar secuencia)
- YaRN adaptado para las frecuencias angulares del modo pareado

**Transferibilidad a Parameter Golf**: **BAJA-MEDIA**. Requiere Flash Attention con soporte especifico y complejidad de implementacion significativa. Con solo 4 KV heads en Parameter Golf, la reduccion a "pares" podria ser contraproducente. Sin embargo, **la idea de compartir keys entre heads con offsets es interesante** y podria explorarse de forma simplificada.

---

### 9. Batch Size Schedule

**Que es**: Incrementar el batch size durante el entrenamiento en lugar de mantenerlo fijo.

**Schedule de modded-nanogpt**:
```
Stage 1: 131K tokens/step  (batch pequeño → updates frecuentes, exploracion)
Stage 2: 262K tokens/step
Stage 3: 393K tokens/step  (batch grande → gradientes precisos, convergencia)
```

**Beneficio**: Batch pequeño al inicio permite mas actualizaciones con gradientes ruidosos (bueno para exploracion). Batch grande al final da gradientes mas precisos para convergencia fina.

**Transferibilidad a Parameter Golf**: **ALTA**. El baseline usa 524K tokens/step fijo. Un schedule de 256K→524K→786K podria mejorar convergencia sin costo adicional en parametros. **Facil de implementar, solo cambia `TRAIN_BATCH_TOKENS` en el loop.**

---

### 10. Partitioned Hyperconnections

**Que es**: Reemplaza las skip connections estandar con conexiones parametrizadas que mezclan la entrada y salida de cada sublayer con coeficientes aprendidos.

**Como funciona**:
```python
# En lugar de: x = x + sublayer(x)
# Usa:         x = lambda_resid * x + lambda_out * sublayer(x) + lambda_x0 * x0
```

- `lambda_resid`: Cuanto del residual preservar (init ~1.05)
- `lambda_out`: Cuanto de la salida del sublayer mezclar
- `lambda_x0`: Cuanto del input original (embedding) inyectar
- Parametros por capa y por sublayer (atencion vs MLP)

**Simplificacion final** (record #77): Reemplazado por "single saved activation" — guardar una activacion intermedia y reusar, sin los coeficientes completos.

**Transferibilidad a Parameter Golf**: **ALTA**. El SOTA actual ya usa phase-transition residual mixing (similar). Los coeficientes aprendidos son escalares (~20 params total), costo nulo en el artifact. **Explorar inyeccion de x0 (input original) en capas profundas podria ser valioso.**

---

### 11. Logit Softcapping

**Que es**: Limitar la magnitud de los logits antes del softmax para estabilizar entrenamiento.

**Implementacion en modded-nanogpt**:
```python
# Entrenamiento (fused kernel):
loss = FusedSoftcappedCrossEntropy(logits, targets, cap=23)

# Inferencia:
logits = 23 * sigmoid((logits + 5) / 7.5)
```

**Evolucion**: Cap inicio en 30 (baseline), reducido a 15 (record #18), luego a 23 con asymmetric rescale (record #54). El rescale asimetrico (offset +5, escala /7.5) permite saturacion diferente para logits positivos vs negativos.

**Transferibilidad a Parameter Golf**: **ALTA**. Ya esta implementado (cap=30). **Experimentar con cap=15-23 y asymmetric rescale podria mejorar estabilidad con pocos pasos de entrenamiento.**

---

### 12. Max Sequence Length Schedule

**Que es**: Comenzar con secuencias cortas y aumentar durante entrenamiento.

**Schedule de modded-nanogpt**:
```
Stage 1: 896 tokens
Stages 2+: 2048 tokens
```

**Beneficio**: Secuencias cortas al inicio = mas pasos por segundo = mas updates = mejor exploracion temprana. Secuencias largas al final = mejor modelado de dependencias largas.

**Transferibilidad a Parameter Golf**: **ALTA**. Combinable con el hallazgo de que seq_len=4096 mejora val_bpb. Un schedule de 512→1024→2048 podria dar lo mejor de ambos mundos: muchos pasos tempranos + contexto largo final.

---

## Resumen de Transferibilidad

### Prioridad ALTA (implementar primero)

| Tecnica | Costo en params | Dificultad | Impacto estimado |
|---------|----------------|------------|-----------------|
| Cautious Weight Decay | 0 | Baja | Medio |
| Batch size schedule | 0 | Baja | Medio |
| Max seq length schedule | 0 | Baja | Medio |
| Smear module | ~512 params | Baja | Medio |
| Skip connection U-net + Backout | ~2 params | Media | Alto |
| Logit softcap tuning | 0 | Baja | Bajo-Medio |
| NorMuon + Polar Express | 0 | Media | Medio |

### Prioridad MEDIA (experimentar)

| Tecnica | Costo en params | Dificultad | Impacto estimado |
|---------|----------------|------------|-----------------|
| Value Embeddings (1-2) | ~1MB | Media | Medio |
| Multi-token prediction | ~0 (tied embed) | Media | Medio |
| Bigram Hash Embedding | ~2.5MB | Media | Medio-Alto |
| Sliding window schedule | 0 | Alta (Flash Attn) | Medio |
| Partitioned hyperconnections (x0 injection) | ~20 params | Baja | Bajo-Medio |

### Prioridad BAJA (investigar)

| Tecnica | Razon |
|---------|-------|
| Paired Head Attention | Complejidad vs beneficio incierto con 4 KV heads |
| FP8 matmul | No aplica — PG usa int8 post-training, no FP8 training |
| Sparse attention gate | Requiere FlexAttention/custom mask |
| Triton kernels fusionados | Beneficio en throughput, no en calidad del modelo |

---

## Diferencias Criticas con Parameter Golf

### Lo que NO es transferible directamente

1. **Velocidad por paso**: El speedrun optimiza ms/paso; PG optimiza calidad/byte. Kernels Triton fusionados ayudan en throughput pero no cambian la calidad del modelo para un numero fijo de pasos.

2. **Precision de entrenamiento**: El speedrun usa FP8 agresivamente para velocidad. PG necesita pesos que cuanticen bien a int8 — priorizar estabilidad sobre velocidad.

3. **Tamano de vocabulario**: El speedrun usa vocab 50K con embeddings masivos. Con vocab 1024, muchas tecnicas de embedding (value embeds, bigram hash) son proporcionalmente mucho mas baratas en parametros pero potencialmente menos impactantes (menos tokens distintos que distinguir).

4. **Untie/Retie embed**: El speedrun untie a 2/3 del entrenamiento. En PG con tied embeddings y vocab 1024, el embedding es ~512KB — untie duplicaria eso. El trade-off es diferente.

### Lo que ES transferible con adaptacion

1. **Schedules** (batch size, seq length, LR warmdown): Directamente aplicables, solo requieren tuning de los breakpoints.

2. **Optimizador** (NorMuon, Polar Express, Cautious WD): Mejora la convergencia sin costo en tamano del artifact.

3. **Arquitectura** (smear, skip U-net, backout, x0 injection): Costo minimo en parametros, potencial alto.

4. **MTP decreciente**: El schedule 3→2→1 token prediction es transferible con tied embeddings.

---

## Plan de Accion Sugerido

### Iteracion 1: Quick Wins (sin cambiar arquitectura)
1. Implementar Cautious Weight Decay en Muon
2. Batch size schedule: 256K → 524K → 786K
3. Seq length schedule: 512 → 1024 → 2048
4. Ajustar logit softcap a 20-23 con offset
5. Extender warmdown (ya validado en PG con WARMDOWN_ITERS=20000)

### Iteracion 2: Mejoras Arquitecturales Baratas
1. Smear module (512 params)
2. Skip connection capa 3→6 con gate + backout
3. x0 injection en capas profundas (escalares)
4. NorMuon con Polar Express

### Iteracion 3: Tecnicas de Mayor Impacto
1. Value embeddings (1-2 matrices, capas finales)
2. Multi-token prediction con schedule decreciente
3. Bigram hash embedding (evaluar costo vs beneficio en presupuesto 16MB)
4. Sliding window attention con schedule

---

*Fuente: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — analisis basado en 77 records (mayo 2024 – marzo 2026)*
*Ultima actualizacion: 2026-03-20*
