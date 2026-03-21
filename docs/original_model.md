# Propuesta de Contribucion Original: Parameter Golf

## Estado del Arte (2026-03-21)

### Merged SOTA: val_bpb 1.1428
- 10L, 512d, int5-MLP/int6-attn, BigramHash(10240), SWA, WD=0.04, zstd-22

### PR Frontier: val_bpb 1.1248 (PR #315)
- 11L, 512d, Partial RoPE (16/64 dims), LN Scale, Late QAT, EMA(0.997), XSA4

### Stack Convergido (lo que todos usan)
Todos los top-20 comparten: 11L/512d/MLP3x/relu²/GQA(8H/4KV)/int6+zstd-22/SmearGate/BigramHash/OrthoInit/WD=0.04/EMA o SWA/sliding eval stride=64.

Las diferencias entre el #1 y #20 son solo ~0.03 BPB — los margenes son minimos.

---

## Analisis de Oportunidades

### Tecnicas saturadas (poco margen)
- SmearGate, BigramHash, SWA/EMA, OrthoInit, WD tuning, sliding eval
- Todos las usan, contribuciones marginales (<0.001 BPP cada una)

### Tecnicas con margen comprobado
| Tecnica | Mejor uso actual | Oportunidad |
|---------|-----------------|-------------|
| **XSA** | 4 ultimas capas (PR #315) | Extender a todas las capas con costo minimo |
| **Partial RoPE** | 16/64 dims (PR #315) | Solo 1 PR lo usa — sub-explorado |
| **LN Scale** | 1/sqrt(layer+1) (PR #315) | Solo 1 PR — facil de combinar |
| **Gradient-Guided Quant** | PR #332, 12 capas | Permite 12L con MLP ligeramente mas estrecho |
| **TTT SGD** (no LoRA) | PR #338 | Freeze primeras 2 capas, 3 epocas SGD |
| **Backout** | PR #339 (sobre limite) | λ·h_mid restado del output — genuinamente novel |

### Tecnicas NO exploradas por nadie
| Tecnica | Potencial | Fuente |
|---------|-----------|--------|
| **Depth recurrence con low-rank adapters** | Alto | RingFormer (arXiv:2502.13181) — ningún PR lo implementa bien |
| **Value projections desde x0** | Medio | Slowrun record #3 — nadie en PG |
| **Per-head attention gating** | Medio | Slowrun/Speedrun — nadie en PG |
| **NoPE-RoPE hybrid** (capas alternadas) | Medio | arXiv:2501.18795 — PR #315 usa Partial RoPE pero no NoPE layers |
| **Batch size schedule** | Bajo-Medio | Speedrun record #46 — nadie en PG |
| **Content-dependent gating en residual** | Medio | Modded-nanogpt sparse attn gate — adaptable |

---

## Propuesta: Modelo "RingGolf"

### Idea Central: Depth Recurrence + Gradient-Guided Mixed Quant

**Hipotesis**: Un modelo con profundidad efectiva de 18-20 capas usando solo 6-7 bloques unicos (compartidos via depth recurrence) con adaptadores low-rank por iteracion, deberia superar modelos de 11-12 capas con parametros unicos, porque:

1. Mas profundidad efectiva mejora la composicion de funciones
2. Los adaptadores low-rank cuestan ~2% de los params del bloque compartido
3. Gradient-guided quant (int5/int6/int7 adaptativo) comprime mejor que int6 uniforme
4. El espacio ahorrado en params se reinvierte en anchura (dim=576+) o mas iteraciones de loop

### Arquitectura Propuesta

```
Embeddings (FP16, tied):
  tok_emb: 1024 × 576 = 589,824 params (FP16 passthrough)
  SmearGate: 576 params

Prelude (2 bloques unicos, no compartidos):
  Block 0: Attn(576d, 8H/4KV) + MLP(576×1728) = ~4.2M params  [int8]
  Block 1: Attn(576d, 8H/4KV) + MLP(576×1728) = ~4.2M params  [int8]

Core (4 bloques compartidos × 3 loops = 12 capas efectivas):
  Block 2-5: 4 bloques × ~4.2M = ~16.8M params [int6]
  Per-loop adapters: 3 loops × 4 bloques × 2 matrices(576×32) = ~442K params [int8]
  Loop iteration embeddings: 3 × 576 = 1,728 params

Coda (2 bloques unicos, no compartidos):
  Block 6: Attn + MLP = ~4.2M params  [int7 via gradient-guided]
  Block 7: Attn + MLP = ~4.2M params  [int8]

XSA en ultimas 4 capas efectivas (coda + ultimo loop de core)
Backout: lambda * h_mid (del loop 2, bloque 3)
```

**Profundidad efectiva: 2 + 12 + 2 = 16 capas** (vs 11 en el SOTA)
**Parametros unicos: ~34M** (vs 26.8M en 11L)
**Parametros compartidos via loop: 16.8M** (reutilizados 3×)

### Presupuesto de Bytes (estimado)

| Componente | Params | Quant | Bytes estimados |
|------------|--------|-------|-----------------|
| Embeddings | 590K | FP16 | 1,180,000 |
| Prelude (2 bloques) | 8.4M | int8 | 8,400,000 |
| Core (4 bloques) | 16.8M | int6 step=4 | ~5,000,000 (zstd-22) |
| Adapters | 442K | int8 | 442,000 |
| Coda (2 bloques) | 8.4M | int7/int8 | ~6,000,000 |
| Scalars/gates | ~10K | FP32 | 40,000 |
| **Total** | **~34.6M** | mixed | **~15.1 MB** |
| Codigo | — | — | ~65,000 |
| **Total submission** | | | **~15.2 MB** |

### Tecnicas Adicionales Incluidas

1. **Partial RoPE (16/72 dims)** — con dim=576, head_dim=72, rotar solo 16 dims
2. **LN Scale (1/sqrt(layer+1))** — damping en capas profundas
3. **Late QAT (ultimo 4%)** — activar STE solo cuando lr_scale < 0.1
4. **EMA (decay=0.997)** — reemplaza SWA
5. **XSA en ultimas 4 capas efectivas** — debiasing ortogonal
6. **Backout connection** — restar lambda·h_mid del output final
7. **SmearGate + BigramHash(2048)** — contexto bigramico
8. **Muon WD=0.04, momentum=0.99** — warmup 0.92→0.99/1500 pasos
9. **Gradient-guided quant** — int5/int6/int7 adaptativo por tensor

### Lo que es Original

1. **Depth recurrence con per-loop low-rank adapters**: Nadie en PG ha implementado esto correctamente. PR #268 lo propuso pero sin resultados. RingFormer (arXiv:2502.13181) muestra que funciona a escala, pero nadie lo ha probado a 16MB.

2. **dim=576 (no 512)**: Todos usan 512. Con depth recurrence, los params ahorrados por sharing se reinvierten en anchura. El Depth Delusion paper sugiere que width importa mas que depth — pero nosotros tenemos ambos (16 capas efectivas + 576 dim).

3. **Combinacion Backout + Depth Recurrence**: El backout de PR #339 es novel. Combinarlo con recurrence (restar la representacion del loop intermedio) es inedito.

---

## Plan de Ejecucion

### Fase 1: Validar depth recurrence basico (1×H100, 30 min)
1. Implementar shared blocks con loop count=3
2. Sin adapters ni backout — solo verificar que depth recurrence no diverge
3. Comparar con 11L baseline a mismos pasos
4. **Criterio de exito**: val_bpb mejor que 11L a mismos pasos de training

### Fase 2: Agregar per-loop adapters + gradient-guided quant (1×H100, 30 min)
1. Low-rank adapters (r=32) por iteracion de loop
2. Loop iteration embeddings (sumados a x antes de cada bloque compartido)
3. Gradient-guided quantization (int5/int6/int7 adaptativo)
4. Verificar que cabe en 16MB
5. **Criterio de exito**: artifact < 16MB, val_bpb mejor que Fase 1

### Fase 3: Full stack + Backout (1×H100, 10 min)
1. Agregar todas las tecnicas: Partial RoPE, LN Scale, XSA, Backout, Late QAT, EMA
2. Verificar que compile con torch.compile y no rompa grad flow
3. **Criterio de exito**: val_bpb < 1.35 en 1×H100/10min (comparable con nuestro v3 de 1.35)

### Fase 4: Run oficial (8×H100, 10 min)
1. Correr 3 seeds
2. Verificar reproducibilidad (std < 0.002)
3. Generar submission artifacts
4. **Criterio de exito**: val_bpb < 1.13 (competitivo con frontera)

### Fase 5: Submission
1. Crear carpeta en records/track_10min_16mb/
2. README.md, submission.json, train_gpt.py, train.log × 3 seeds
3. PR al upstream

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigacion |
|--------|-------------|------------|
| Depth recurrence diverge con Muon | Media | Usar learning rate mas bajo para bloques compartidos |
| Artifact > 16MB | Media | Ajustar loop count (2 en vez de 3) o dim (544 en vez de 576) |
| torch.compile no soporta loops dinamicos | Alta | Desenrollar el loop en el forward pass (static) |
| Per-loop adapters no mejoran | Baja | Sin adapters, depth recurrence pura sigue siendo viable |
| Overhead de recurrence > ganancia de profundidad | Media | Si falla, caer a 11L + tecnicas del stack convergido |

## Fallback

Si depth recurrence falla, nuestra contribucion original seria:
- **dim=576** (nadie lo ha probado) + gradient-guided quant + Backout + Partial RoPE + LN Scale
- Esto combina 4 tecnicas que solo PR #315 y #339 tienen, ninguna submission merged las usa

---

*Documento vivo — actualizar despues de cada fase de ejecucion*
*Creado: 2026-03-21*
