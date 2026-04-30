# IGLA Race Manager Plugin

Plugin для управления IGLA training competition — стратегия воркеров, мониторинг, deployment в одном месте.

## Команды

### `/igla` — Главный менеджер
```
/igla                    — Показать статус (fleet, queue, leaderboard)
/igla redeploy [acc|all] — Redeploy worker(s)
/igla queue --wave=<name> — Добавить волну экспериментов
/igla status --detailed  — Детальный отчёт по matrix
/igla champion --fmt=... --arch=... — Найти champion ячейку
```

### `/matrix` — Competition Matrix
```
/matrix phase1      — Tier A: 4 formats × 5 models × 5 seeds = 100 exp
/matrix phase2      — Tier B: GF8/12/20/24 emulated
/matrix promote <s> <f> <h> — Long run на победителе
/matrix status      — Fleet + queue + leaderboard
/matrix redeploy    — Redeploy всех 6 workers
/matrix queue MEGA-GAP — Скрытые размеры {1280...3072}
/matrix queue MEGA-FMT  — Все 11 форматов
/matrix queue MEGA-ATTN — Attention layers {0,2,4,6,8,12}
/matrix queue MEGA-LR   — LR sweep
/matrix queue MEGA-CTX  — Context sweep
```

## Competition Grid

### Numbers (12 форматов)
| Lane | Format | Bits | Status |
|------|--------|------|--------|
| N1 | GF4 | 4 | ❌ placeholder |
| N2 | GF8 | 8 | ⚠️ emulated |
| N3 | GF12 | 12 | ⚠️ emulated |
| N4 | GF16 | 16 | ✅ production |
| N5 | GF20 | 20 | ⚠️ emulated |
| N6 | GF24 | 24 | ⚠️ emulated |
| N7 | GF32 | 32 | ❌ placeholder |
| N8 | GF64 | 64 | ❌ placeholder |
| B1 | fp16 | 16 | ✅ industry |
| B2 | bf16 | 16 | ✅ industry |
| B3 | fp32 | 32 | ✅ industry |
| B4 | fp8 | 8 | ⚠️ NVIDIA-only |

### Models (6 архитектур)
| Lane | Model | Type | Hidden | Attn |
|------|-------|------|--------|------|
| M1 | MLP-baseline | feedforward | 384-1280 | — |
| M2 | JEPA-T | predictive | 512-1024 | 2-4 |
| M3 | NCA | cellular | 384-512 | 0 |
| M4 | TF-1L | transformer | 512-1024 | 1 |
| M5 | TF-2L | transformer | 828-1024 | 2 ⭐ |
| M6 | TF-3L+ | transformer | 1024-1280 | 3-4 |

## Components

### Skills
- **igla.md** — Main command wrapper
- **matrix.md** — Competition matrix management

### Agents
- **igla-operator.md** — Executes DB queries, Railway commands
- **matrix-agent.md** — Matrix phase deployment logic

### Hooks
- **monitor-experiments.md** — Auto-status after experiment ops

## Sanctioned Seeds

```rust
const SANCTIONED_SEEDS: [u64; 5] = [1597, 2584, 4181, 6765, 10946];
```

## R5 Honest Contract

- ✅ Only sanctioned seeds in competition
- ✅ No mock data in competition rows
- ✅ Real BPB measurements only
- ✅ Tier C placeholders marked as "not_yet_implemented"

## Fleet Architecture

6 Railway accounts (acc0-acc5) × 1 worker each = 6 total workers

| Account | Service | Status |
|---------|---------|--------|
| acc0 | seed-agent-0 | heartbeat check |
| acc1 | seed-agent-1 | heartbeat check |
| acc2 | seed-agent-2 | heartbeat check |
| acc3 | seed-agent-3 | heartbeat check |
| acc4 | seed-agent-4 | heartbeat check |
| acc5 | seed-agent-5 | heartbeat check |

## Quick Start

```bash
# 1. Check current status
/igla

# 2. If workers are dead (stale > 5 min)
/igla redeploy all

# 3. Deploy Phase 1 competition
/matrix phase1

# 4. Monitor progress
/igla status --detailed

# 5. When leaders emerge, promote to long run
/matrix promote TF-2L FP32 1024
```

## Gate-2 Target

**BPB < 1.85 at step >= 4000**

Current best: 2.251 BPB (gap: +0.401)

Time remaining: T-19h
