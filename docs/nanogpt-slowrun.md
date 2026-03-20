# NanoGPT Slowrun — Analisis de Tecnicas Data-Efficient

## Contexto

El [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) es el tercer challenge de la familia, optimizando **L(D)** — la menor loss posible con un dataset fijo de **100M tokens** de FineWeb, sin limites de compute. Es complementario a Parameter Golf y al Speedrun:

| | Speedrun | **Slowrun** | Parameter Golf |
|---|---|---|---|
| **Optimiza** | L(T) — menor tiempo | **L(D) — menor loss** | L(N) — menor loss |
| **Restriccion** | Tiempo (~1.4 min) | **Datos (100M tokens)** | Parametros (16MB) |
| **Modelo** | ~124M params | **2.7B params** | ~15M params |
| **Compute** | 8×H100, limitado | **8×H100, 1h (limited) / ilimitado** | 8×H100, 10 min |
| **Datos** | ~10B tokens (1 epoca) | **100M tokens (12+ epocas)** | ~8B tokens (1 epoca) |

**Implicacion clave para Parameter Golf**: El slowrun demuestra que tecnicas pueden exprimir mas informacion de datos limitados. Aunque Parameter Golf no esta limitado en datos (8B tokens disponibles), las tecnicas de regularizacion y eficiencia de datos pueden mejorar lo que un modelo pequeño aprende en ~13K pasos.

---

## Tres Tracks: Diferentes Regimenes de Compute

### Limited Track (1 hora, 8×H100)

**Modelo**: 2.7B params, 30 capas, dim=1792, 14 heads
**SOTA**: 3.252 val_loss (11 records)

Un modelo ~180× mas grande que el de Parameter Golf, entrenado 12 epocas sobre 100M tokens. La regularizacion pesada es esencial — sin ella, modelos grandes sobreajustan catastroficamente.

### Tiny Track (15 minutos, 8×H100)

**Modelo**: ~300M params, 16 capas, dim=1024, 8 heads
**SOTA**: 3.365 val_loss (7 records)

Mas cercano en escala temporal a Parameter Golf (10 min). Las tecnicas del tiny track son las mas directamente transferibles.

### Unlimited Track (sin limites)

**Modelo**: Ensemble de 10-12 modelos × 2.7B, con distilacion en cadena
**SOTA**: 3.045 val_loss (9 records, ~44 horas en 2 nodos)

Demuestra el limite inferior alcanzable con compute infinito sobre datos fijos. La distilacion y ensembles son las tecnicas clave.

---

## Progresion de Records

### Limited Track

| # | Val Loss | Delta | Tecnica | Categoria |
|---|---------|-------|---------|-----------|
| 1 | 3.402 | — | Baseline: 2.7B, Muon, dropout 0.1, WD 1.6 | Baseline |
| 2 | 3.376 | -0.026 | Shuffle por epoca | Datos |
| 3 | 3.349 | -0.027 | **Value projections desde x0** (no tablas) | Arquitectura |
| 4 | 3.335 | -0.014 | **SwiGLU** activation | Arquitectura |
| 5 | 3.314 | -0.021 | **U-Net skip connections** | Arquitectura |
| 6 | 3.295 | -0.019 | **Gating por head de atencion** | Arquitectura |
| 7 | 3.285 | -0.010 | **Layer looping** (repetir capas 15-20 ultimas epocas) | Compute |
| 8 | 3.278 | -0.007 | Layer looping expandido (3× antes de capas finales) | Compute |
| 9 | 3.276 | -0.002 | **Exclusive Self Attention (XSA)** | Atencion |
| 10 | 3.270 | -0.006 | LR tuning + warmdown tuning | Hiperparametros |
| 11 | 3.252 | -0.018 | **EMA de pesos** + hyperparameter tuning | Regularizacion |

**Patron**: Las mejoras arquitecturales (records 3-6) dan los saltos mas grandes. Layer looping y EMA son los "trucos" finales.

### Tiny Track

| # | Val Loss | Tecnica |
|---|---------|---------|
| 1 | 3.428 | Baseline 300M |
| 2 | 3.410 | SwiGLU |
| 3 | 3.395 | U-Net |
| 4 | 3.385 | Attention gating |
| 5 | 3.383 | Warmdown tuning |
| 6 | 3.374 | **Half-truncated RoPE** + **partial key offset** + residual lambdas=1.1 |
| 7 | 3.365 | **Weight decay schedule** (3 fases) |

### Unlimited Track

| # | Val Loss | Tecnica | Tiempo |
|---|---------|---------|--------|
| 1 | 3.402 | Baseline single model | 47 min |
| 2 | 3.264 | **Ensemble 8 modelos** + logit averaging | 6h 44m |
| 7 | 3.089 | Ensemble 10 + looping + distilacion | 19h |
| 9 | 3.045 | Ensemble + mas looping + modelo mas grande | 44h |

**Insight brutal**: El ensemble de 8 modelos (record #2) mejora de 3.402→3.264 en un solo salto (-0.138). Esto demuestra que la diversidad de modelos captura mas informacion del dataset limitado que cualquier tecnica individual.

---

## Tecnicas Detalladas y Transferibilidad

### 1. Regularizacion Pesada (Weight Decay + Dropout)

**El descubrimiento central del slowrun**: En regimenes data-limited, el weight decay optimo es **30× mayor** que la practica estandar.

```
Weight Decay tipico:  0.01 - 0.1
Weight Decay slowrun: 0.8 - 1.6 (!)
Dropout:              0.1
```

**Fundamento teorico** (Kim et al., 2025 — "Pre-training under infinite compute"):
- Modelos grandes tienen un sesgo de simplicidad fuerte
- La regularizacion amplifica este sesgo, forzando representaciones mas generalizables
- Sin regularizacion, un modelo 1.4B supera a un 2.7B en datos limitados
- Con regularizacion, el scaling monotono se recupera: mas parametros → menor loss

**Weight Decay Schedule (tiny track)**:
```
Fase 1 (epocas 0-2):  WD = 0.8    (establecer representaciones)
Fase 2 (epocas 2-8):  WD: 0.8→0.1 (relajar para explorar)
Fase 3 (epocas 8-16): WD: 0.1→1.25 (endurecer para convergencia)
```

**Transferibilidad a Parameter Golf**: **ALTA, pero con matices**. Parameter Golf NO esta en regimen data-limited (8B tokens, 1 epoca). Sin embargo:
- El modelo es muy pequeño (~15M params) y tiene ~13K pasos — es un regimen "step-limited"
- El Muon WD record de PG ya usa WD=0.02, que es minimo
- **El weight decay schedule (bajo→alto en fase final)** podria ayudar con cuantizacion — el warmdown del slowrun es analogo al warmdown de PG
- Dropout probablemente NO ayuda en PG — con 1 epoca no hay sobreajuste a datos

---

### 2. Value Projections desde x0

**Que es**: En lugar de tablas de value embedding (como modded-nanogpt), proyectar el embedding original `x0` a traves de una capa lineal para inyectarlo en los valores de atencion.

**Como funciona**:
```python
# En capas seleccionadas (patron alternado):
ve = linear_projection(x0)    # x0 es el embedding inicial, no el residual
v = W_v @ x + gate * ve       # gate = 2 * sigmoid(linear(x[:32]))
```

**Diferencia con modded-nanogpt**: En el speedrun, las value embeddings son tablas de lookup por token. En el slowrun, son proyecciones del embedding. Las proyecciones:
- Comparten informacion entre tokens similares (generalizacion)
- Cuestan menos parametros que tablas completas
- Son mas expresivas (combinan informacion del embedding, no solo buscan por ID)

**Transferibilidad a Parameter Golf**: **ALTA**. Con vocab=1024 y dim=512:
- Tabla de value embed: 1024 × 512 = 512KB por tabla (costo fijo, no se beneficia de compresion)
- Proyeccion lineal: 512 × 64 = 32KB (mucho mas compacta, y cuantiza bien a int8)
- **La proyeccion desde x0 es claramente superior para Parameter Golf** donde el presupuesto es limitado

---

### 3. U-Net Skip Connections

**Que es**: Division del modelo en encoder (primera mitad) y decoder (segunda mitad), con skip connections directas de encoder_layer[i] a decoder_layer[n-1-i].

```
Layer 0  → Layer 1  → ... → Layer 14 → Layer 15 → ... → Layer 29
  ↓          ↓                  ↓          ↑                  ↑
  └──────────┴──── skip × w ────┴──────────┘                  │
                                └─────── skip × w ────────────┘
```

**Implementacion**:
```python
# Encoder: guardar outputs
encoder_outputs = []
for i in range(n_layer // 2):
    x = layer_i(x)
    encoder_outputs.append(x)

# Decoder: usar skip connections
for i in range(n_layer // 2, n_layer):
    j = n_layer - 1 - i  # indice espejo
    x = x + skip_weights[i - encoder_layers] * encoder_outputs[j]
    x = layer_i(x)
```

- `skip_weights` inicializados a 1.0, aprendidos con LR=0.001
- Cada skip connection tiene su propio peso escalar

**Transferibilidad a Parameter Golf**: **ALTA**. Con 9-10 capas:
- Encoder: capas 0-4, Decoder: capas 5-9
- 5 skip weights escalares = ~20 bytes (costo nulo)
- El SOTA de PG ya usa residual mixing, pero NO U-net skip completo
- **Combinar U-net con el skip 3→6 del speedrun podria ser potente**

---

### 4. Per-Head Attention Gating

**Que es**: Cada head de atencion tiene un gate aprendido que puede "apagar" su contribucion dependiendo del contexto.

**Como funciona**:
```python
# Cada head gate es una funcion del hidden state
gate = 2 * sigmoid(linear(x_norm))           # gate por head, forma (B, T, n_heads)
attn_output = attn_output * gate.unsqueeze(-1)  # Escalar cada head
```

- Inicializacion: linear.weight = 0, linear.bias = 0 (gate empieza en sigmoid(0) = 0.5)
- Cada head aprende a activarse/desactivarse por contexto
- LR separado: 0.01, betas (0.9, 0.99)

**Beneficio**: Permite al modelo "silenciar" heads irrelevantes para ciertos tokens, actuando como un soft-MoE a nivel de heads.

**Transferibilidad a Parameter Golf**: **ALTA**. Con 8 heads y dim=512:
- Gate linear: 512 → 8 = 4K parametros (negligible)
- Ya validado en ambos tracks del slowrun (limited y tiny)
- **Implementacion trivial, beneficio probado**

---

### 5. Layer Looping (Repeticion de Capas)

**Que es**: En las epocas finales del entrenamiento, ejecutar un subconjunto de capas multiples veces por forward pass. Aumenta la profundidad efectiva sin aumentar parametros.

**Implementacion**:
```python
# Ultimas 3 epocas: capas 15-20 se ejecutan 3 veces
if dupe_active:
    for repeat in range(dupe_loops):  # dupe_loops = 2 (total 3 pasadas)
        for layer in layers[15:21]:
            x = layer(x)
    # Luego continuar con capas 21-29
```

**Por que funciona**: En el regimen data-limited, el modelo "necesita" mas compute por token para extraer maxima informacion. Repetir capas del medio es mas barato que añadir capas (mismos parametros, mas compute).

**Unlimited track**: Mas agresivo — capas 15-25 repetidas 4 veces.

**Transferibilidad a Parameter Golf**: **MEDIA-ALTA**. Muy interesante porque:
- Parameter Golf esta limitado en parametros, no en compute (10 min)
- Repetir capas intermedias 2-3 veces **no aumenta el tamano del artifact** pero usa mas del presupuesto de 10 min
- Trade-off: menos pasos totales vs mas profundidad efectiva por paso
- **Con 9 capas, repetir capas 3-5 podria dar profundidad efectiva de ~15 capas sin costo en bytes**
- Necesita evaluarse si el costo en tiempo (~50% mas lento por paso) compensa

---

### 6. EMA de Pesos (Exponential Moving Average)

**Que es**: Mantener una copia promediada exponencialmente de los pesos durante entrenamiento. Al final, mezclar los pesos finales con el EMA.

**Implementacion**:
```python
# Cada 10 pasos (a partir del 90% del entrenamiento):
ema_shadow = decay * ema_shadow + (1 - decay) * current_weights

# Post-entrenamiento:
final_weights = 0.7 * trained_weights + 0.3 * ema_shadow
```

- Decay: 0.95
- Inicio: 90% del entrenamiento (no empieza antes para no contaminar con pesos tempranos)
- Blending optimo: 70/30 (mas peso a los finales, algo de EMA para suavizar)
- Update cada 10 pasos para minimizar overhead de copia CPU

**Transferibilidad a Parameter Golf**: **ALTA**. Cero costo en parametros del artifact (EMA se resuelve antes de export). El unico costo es memoria durante entrenamiento (~2× pesos). Con ~15M params, esto es ~30MB extra en GPU — negligible en H100 con 80GB.

**Consideracion critica**: EMA suaviza la distribucion de pesos, lo cual podria **mejorar cuantizacion** (menos outliers). Esto lo hace especialmente valioso para PG.

---

### 7. Half-Truncated RoPE

**Que es**: Solo aplicar rotaciones posicionales a la mitad de las dimensiones del head. La otra mitad queda como "dimensiones estacionarias" sin codificacion posicional.

**Implementacion**:
```python
half = head_dim // 4  # Solo 32 de 128 dims rotan
inv_freq = torch.arange(0, half*2, 2) / (half*2)  # Frecuencias para dims rotadas
# Pad con zeros para dims estacionarias
inv_freq = torch.cat([inv_freq, torch.zeros(head_dim//2 - half)])
```

**Por que funciona**: Las dimensiones estacionarias aprenden representaciones "position-agnostic" — patrones que son utiles independientemente de la posicion (tipos de token, categorias semanticas). Las dimensiones rotadas manejan relaciones posicionales.

**Transferibilidad a Parameter Golf**: **MEDIA**. Ya usa RoPE estandar (todas las dims rotan). Con head_dim = 64 (512/8 heads), truncar a la mitad daria 16 dims rotadas + 16 estacionarias. **Vale la pena experimentar, costo cero en parametros.**

---

### 8. Partial Key Offset

**Que es**: Desplazar las dimensiones estacionarias de las keys 1 posicion hacia atras. Combina con half-truncated RoPE para dar a las dims estacionarias una nocion de "token anterior".

```python
# Solo en capas con ventana larga o capa final:
k[:, 1:, :, head_dim//2:] = k[:, :-1, :, head_dim//2:].clone()
```

**Beneficio**: Implementa un "induction head" barato — las dimensiones estacionarias de la key codifican el token *anterior*, permitiendo al modelo detectar patrones de repeticion sin necesidad de atencion multi-capa.

**Transferibilidad a Parameter Golf**: **ALTA**. Cero parametros, implementacion de una linea. Similar al "smear" del speedrun pero aplicado a keys en vez de embeddings. **Directamente aplicable.**

---

### 9. Exclusive Self Attention (XSA)

**Que es**: Una variante de self-attention donde cada token atiende a tokens que son diferentes de si mismo, en vez de similares. Referenciado en [arXiv:2603.09078](https://arxiv.org/pdf/2603.09078).

**Transferibilidad a Parameter Golf**: **BAJA-MEDIA**. Mejora marginal en el slowrun (solo -0.002 val_loss) y requiere implementacion custom de atencion. No prioritario.

---

### 10. Ensemble + Distilacion en Cadena (Unlimited Track)

**Que es**: Entrenar N modelos secuencialmente, donde cada modelo aprende de:
1. El ground truth (cross-entropy loss)
2. Los logits del modelo anterior (KL divergence)

```python
loss = (1 - alpha) * CE(student_logits, targets) + alpha * T² * KL(student || teacher)
# alpha = 0.7 — el "conocimiento" del teacher domina
```

**Logit averaging para evaluacion**:
```python
final_logits = mean(model_1_logits, model_2_logits, ..., model_N_logits)
```

**Por que funciona**: Cada modelo en la cadena tiene diferentes seeds y por tanto diferentes minimos locales. La distilacion transfiere el "conocimiento oscuro" (distribuciones de probabilidad sobre tokens incorrectos) que el ground truth no contiene. El ensemble promedia los errores individuales.

**Transferibilidad a Parameter Golf**: **INDIRECTA pero importante**.
- Entrenar un ensemble y distilacion completa no cabe en 10 min
- PERO: **distilacion de un modelo grande a uno pequeño** es directamente relevante
  - Entrenar un modelo grande offline (sin limite de tiempo)
  - Destilar a un modelo de 16MB durante los 10 min
  - El "teacher" no cuenta contra el artifact si solo se usa durante entrenamiento
- **EMA como "self-ensemble"** ya esta capturado en la tecnica #6

---

### 11. Data Shuffling por Epoca

**Que es**: Reordenar los shards de datos al inicio de cada epoca con una seed determinista.

```python
# Cada epoca, nueva permutacion de shards:
gen = torch.Generator().manual_seed(epoch_seed)
indices = torch.randperm(len(shards), generator=gen)
```

**Por que importa**: Sin shuffling, el modelo ve los datos en el mismo orden cada epoca, lo que puede causar sesgos en el entrenamiento (el modelo "memoriza" la secuencia de datos, no el contenido).

**Transferibilidad a Parameter Golf**: **NO APLICA**. PG entrena por 1 epoca sobre 8B tokens. No hay re-uso de datos, asi que shuffling entre epocas es irrelevante. El shuffling intra-epoca ya esta implementado por la carga aleatoria de shards.

---

## Insights Fundamentales del Slowrun

### 1. Overparametrization + Regularizacion > Modelo Pequeño sin Regularizacion

El hallazgo central: un modelo 2.7B con WD=1.6 y dropout=0.1 supera a un modelo 1.4B sin regularizacion, incluso con solo 100M tokens. La regularizacion no "limita" al modelo — lo fuerza a encontrar soluciones mas simples y generalizables.

**Implicacion para PG**: El modelo de 15M params de PG es tan pequeño que probablemente no se beneficia de regularizacion pesada (no tiene capacidad para sobreajustar en 1 epoca). Pero la **forma** de la regularizacion importa: WD schedule y EMA pueden mejorar la distribucion de pesos para cuantizacion.

### 2. Layer Looping: Mas Compute sin Mas Parametros

La repeticion de capas intermedias es una de las tecnicas mas relevantes para PG:
- No aumenta el artifact (mismos pesos, ejecutados multiples veces)
- Aumenta la profundidad efectiva sin costo en bytes
- El trade-off es puramente temporal (ms/paso vs pasos totales)

**Calculo para PG**:
```
Sin looping: 9 capas × 13,500 pasos × 43ms = 10 min, profundidad=9
Con looping 2×: 9 capas → efectivas 15, ~7,700 pasos × 78ms = 10 min, profundidad=15
```

Si la profundidad efectiva importa mas que el numero de pasos, layer looping gana.

### 3. Value Projections > Value Embeddings (para modelos pequeños)

El cambio de tablas de value embedding a proyecciones lineales desde x0 (record #3) fue una de las mejoras mas grandes (-0.027). Para PG con vocab pequeño, esto es especialmente relevante:
- Proyeccion 512→64: 32KB (vs tabla 1024×512 = 512KB)
- Mejor generalizacion (combinaciones lineales vs lookup discreto)
- Cuantiza mejor (matriz densa vs embedding sparse)

### 4. EMA como Regularizador y Facilitador de Cuantizacion

EMA no solo mejora la loss — suaviza la distribucion de pesos. Esto es potencialmente mas valioso en PG que en el slowrun:
- Menos outliers = menor error de cuantizacion int8
- El blending 70/30 (final + EMA) podria ser el "warmdown" perfecto para cuantizacion

---

## Resumen de Transferibilidad a Parameter Golf

### Prioridad ALTA

| Tecnica | Costo | Dificultad | Impacto Estimado |
|---------|-------|------------|-----------------|
| EMA de pesos (blending pre-export) | 0 params, ~30MB RAM | Baja | Medio (mejor cuantizacion) |
| Per-head attention gating | ~4K params | Baja | Medio |
| Value projections desde x0 | ~32KB | Baja | Medio |
| U-Net skip connections | ~10 params | Baja | Medio |
| Partial key offset | 0 params | Trivial | Bajo-Medio |

### Prioridad MEDIA

| Tecnica | Costo | Dificultad | Impacto Estimado |
|---------|-------|------------|-----------------|
| Layer looping (capas intermedias) | 0 params, +tiempo/paso | Media | Alto potencial |
| Half-truncated RoPE | 0 params | Baja | Bajo-Medio |
| Weight decay schedule | 0 params | Baja | Bajo (PG no data-limited) |

### Prioridad BAJA / NO APLICA

| Tecnica | Razon |
|---------|-------|
| Dropout pesado | PG no sobreajusta (1 epoca) |
| Data shuffling | PG no re-usa datos |
| Ensemble + distilacion | No cabe en 10 min (pero distilacion offline es interesante) |
| XSA | Mejora marginal (-0.002), complejidad alta |

---

## Conexiones entre los Tres Challenges

### Tecnicas que funcionan en TODOS

| Tecnica | Speedrun | Slowrun | Aplicada en PG? |
|---------|----------|---------|-----------------|
| SwiGLU MLP | ✓ (ReLU²) | ✓ | ✓ (ya en baseline) |
| RoPE | ✓ | ✓ | ✓ (ya en baseline) |
| U-Net skip connections | ✓ | ✓ | Parcial (phase-transition residual) |
| Value embeddings/projections | ✓ | ✓ | No |
| Per-head attention gating | ✓ | ✓ | No |
| Logit softcap | ✓ | ✓ | ✓ (cap=30) |
| Muon optimizer | ✓ | ✓ | ✓ (version basica) |
| Learned residual lambdas | ✓ | ✓ | Parcial (SOTA tiene sigmoid mixing) |

### Tecnicas unicas de cada challenge

| Challenge | Tecnica Unica | Razon |
|-----------|---------------|-------|
| **Speedrun** | Triton kernels fusionados, FP8 matmul | Optimizar ms/paso |
| **Speedrun** | Bigram hash embedding | Vocab grande (50K), info local barata |
| **Slowrun** | Layer looping | Mas compute sin mas params |
| **Slowrun** | EMA weight blending | Suavizar sobre multiples epocas |
| **Slowrun** | Heavy WD + dropout | Regimen data-limited |
| **Slowrun** | Ensemble distilacion | Compute ilimitado |
| **PG** | int8 cuantizacion + zlib | Artifact ≤ 16MB |
| **PG** | FP16 embedding (no quantize) | Embedding sensible a cuantizacion |
| **PG** | TTT LoRA eval | Adaptar por documento en eval |
| **PG** | Sliding window eval | Maximizar contexto en eval |

---

## Plan de Accion: Tecnicas del Slowrun para Parameter Golf

### Iteracion 1: Sin riesgo, implementacion rapida
1. **EMA de pesos**: Activar al 90% del entrenamiento, blend 70/30 antes de cuantizacion
2. **Per-head attention gating**: Linear 512→8 con sigmoid, zero-init
3. **Partial key offset**: Desplazar dims superiores de K 1 posicion

### Iteracion 2: Arquitectura
1. **Value projections desde x0**: Proyeccion 512→64 en capas finales (6-9)
2. **U-Net skip completo**: Encoder capas 0-4, decoder capas 5-9, 5 skip weights

### Iteracion 3: Experimentacion
1. **Layer looping**: Repetir capas 3-5 dos veces, medir trade-off tiempo/calidad
2. **Half-truncated RoPE**: Solo rotar mitad de head_dim
3. **Weight decay schedule**: WD creciente en la fase de warmdown (mejor cuantizacion)

---

*Fuente: [qlabs-eng/slowrun](https://github.com/qlabs-eng/slowrun) — analisis basado en 27 records across 3 tracks (febrero-marzo 2026)*
*Ultima actualizacion: 2026-03-20*
