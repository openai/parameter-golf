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

Menor es mejor. El baseline logra ~1.224 val_bpb; el SOTA actual es ~1.175.

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
Pesos FP32 → Cuantizacion int8 → Compresion zlib → Artifact
```

La **degradacion post-cuantizacion** es un factor critico:
- Baseline: ~0.007 BPB de penalizacion (pre-quant 1.217 → post-quant 1.224)
- Mejor caso: ~0.001 BPB con tecnicas especializadas

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

## Espacio de Exploracion

### Direcciones Prometedoras (No Exploradas o Sub-exploradas)

1. **Depth Recurrence / Weight Tying entre Capas**: Compartir pesos entre capas permite un modelo "mas profundo" sin aumentar parametros. Similar a Universal Transformers.

2. **Quantization-Aware Training (QAT)**: Simular cuantizacion int8 durante entrenamiento con straight-through estimators. Podria reducir drasticamente la penalizacion post-quant.

3. **Vocabulario mas grande** (2048, 4096): Mayor vocabulario mejora bits-per-byte pero aumenta el tamano del embedding. El trade-off depende de la compresibilidad de los pesos.

4. **Mixture of Experts (MoE)**: Mas parametros con activacion sparse. El challenge es que todos los expertos deben caber en 16MB comprimidos.

5. **Knowledge Distillation**: Entrenar un modelo grande como teacher y destilar a un modelo pequeno que quepa en 16MB.

6. **BitNets / 1-bit models**: Modelos con pesos binarios/ternarios que comprimen extremadamente bien.

7. **Arquitecturas no-transformer**: RWKV, Mamba (SSM), o hibridos que podrian ser mas eficientes en parametros.

8. **Tokenizadores optimizados**: Un tokenizador que comprima mejor FineWeb mejora val_bpb directamente (pero los submissions con cambios de tokenizador se examinan con mas rigor).

9. **Aggressive NTK-RoPE extrapolacion**: Evaluar con contexto mas largo que el de entrenamiento. Los resultados muestran que 1.375× es optimo; 2× degrada.

10. **Mega-kernels / Fusiones CUDA**: Optimizar el throughput para completar mas pasos en 10 minutos sin cambiar la arquitectura.

---

## Referencias

- **Neural Scaling Laws** (Kaplan et al., 2020): Fundamento teorico de L(N) — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **NanoGPT Speedrunning**: Challenge hermano que optimiza L(T) — [GitHub](https://github.com/KellerJordan/modded-nanogpt)
- **NanoGPT Slowrun**: Challenge hermano que optimiza L(D) — [GitHub](https://github.com/qlabs-eng/slowrun)
- **Muon Optimizer**: Ortogonalizacion Newton-Schulz para preacondicionamiento — usado en modded-nanogpt
- **OpenAI Discord**: #parameter-golf-discussions, #parameter-golf-announcements
- **Compute Grants**: $1M en creditos de OpenAI disponibles via [formulario](https://openai.com/index/parameter-golf/#credit-form)

---

*Ultima actualizacion: 2026-03-20*
