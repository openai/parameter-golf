---
name: matrix
description: Manage IGLA Competition Matrix — Numbers × Models grid with Phase 1/2/3 deployment strategy
type: skill
---

IGLA Competition Matrix — управляет 12 числовыми форматами × 6 архитектурами моделей (72 ячейки минимум).

## Competition Grid

**Numbers (Форматы):**
- N1: GF4 (4-bit) — extract-only, placeholder
- N2: GF8 (8-bit) — extract-only
- N3: GF12 (12-bit) — extract-only
- N4: GF16 (16-bit) — ✅ production ready
- N5: GF20 (20-bit) — extract-only
- N6: GF24 (24-bit) — extract-only
- N7: GF32 (32-bit) — spec-only, placeholder
- N8: GF64 (64-bit) — placeholder
- B1: fp16 — industry standard ✅
- B2: bf16 — industry standard ✅
- B3: fp32 — industry standard ✅
- B4: fp8 — NVIDIA-specific, optional

**Models (Архитектуры):**
- M1: MLP-baseline — feedforward {384,512,768,828,1024,1280}
- M2: JEPA-T — Joint-Embedding Predictive {512,768,1024} 2-4 attn
- M3: NCA — Neural Cellular Automata {384,512} 0 attn
- M4: TF-1L — 1-layer Transformer {512,768,1024} 1 attn
- M5: TF-2L — 2-layer Transformer {828,1024} 2 attn (champion)
- M6: TF-3L+ — 3+ layer {1024,1280} 3-4 attn

## Usage

### Phase 1: Tier A Real Numbers (T+1h)
Запускает 4 real formats × 5 models × 5 seeds × step=4000 = 100 lane-runs.

```
/matrix phase1
```

Форматы: GF16, fp16, bf16, fp32
Модели: MLP, JEPA-T, NCA, TF-1L, TF-2L
Hidden: h=1024 для всех, h=1536 для TF-2L champion
Seeds: {1597, 2584, 4181, 6765, 10946}
LR: 0.0015 для всех
Priority: 92
Step: 4000

### Phase 2: Tier B Emulated Numbers (T+3h)
Добавляет GF8/12/20/24 via on-the-fly quantization.

```
/matrix phase2
```

### Phase 3: Targeted Long Runs (T+2h)
Когда лидеры определены — запускает step=27000-50000 на победителях.

```
/matrix promote <strategy> <format> <hidden>
```

### Monitor Fleet
```
/matrix status
```

Показывает:
- Worker heartbeat status (alive < 5m / < 60s / stale)
- Queue status by account
- Leaderboard по формату и модели
- Gate-2 candidates

### Redeploy Workers
Один запрос для redeploy всех 6 воркеров:

```
/matrix redeploy
```

Использует `tri-railway service redeploy` для каждого account.

## Implementation Notes

- Tier A (real): native trainer support
- Tier B (emulated): quantize-on-the-fly в trios-trainer-igla
- Tier C (placeholder): записывает в ledger "not_yet_implemented"

Все эксперименты проходят через R5-honest contract — no mock data in competition rows.
