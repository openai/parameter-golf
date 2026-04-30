---
name: igla
description: IGLA RACE — управление воркерами, стратегия экспериментов и мониторинг Gate-2 прогресса. Автоматизирует redeploy 6 acc, очереди experiments и competition matrix tracking.
---

# IGLA RACE Skill — Competition Matrix & Fleet Operations

## 🏁 ОПЕРАЦИОННАЯ СИТУАЦИЯ

**T-до Gate-2:** динамически вычисляется от 2026-04-30T23:59Z
**Anchor:** φ² + φ⁻² = 3 · R5-honest, non-mock only

## 🚨 FLEET STATUS CHECK

```bash
# Проверить статус всех 6 воркеров
curl -s "$NEON_DATABASE_URL" \
  -c "SELECT railway_acc, COUNT(*) as workers, 
       MAX(last_heartbeat) as latest_hb,
       EXTRACT(EPOCH FROM (NOW() - MAX(last_heartbeat)))/60 as minutes_stale
       FROM workers 
       GROUP BY railway_acc 
       ORDER BY railway_acc;"
```

**Статусы:**
- `minutes_stale < 1` → ✅ ALIVE
- `minutes_stale < 5` → ⚠️ SLOW
- `minutes_stale >= 5` → ❌ STALE

## 📊 QUEUE STATUS CHECK

```sql
-- Статус очереди
SELECT status, COUNT(*) FROM experiment_queue GROUP BY status ORDER BY status;

-- Pending по аккаунтам
SELECT account, COUNT(*) FROM experiment_queue WHERE status='pending' GROUP BY account;

-- Running experiments
SELECT id, canon_name, account, created_at,
       EXTRACT(EPOCH FROM (NOW() - created_at))/60 as minutes_running
       FROM experiment_queue WHERE status='running';
```

## 🏅 COMPETITION MATRIX — ЧИСЛА × МОДЕЛИ

### Числа (Numbers)
| Tier | Format | Status |
|------|--------|--------|
| A (real) | GF16, fp16, bf16, fp32 | ✅ Production ready |
| B (emulated) | GF8, GF12, GF20, GF24 | ⚠️ On-the-fly quant |
| C (placeholder) | GF4, GF32, GF64 | ❌ Not implemented |

### Модели (Models)
| Code | Name | ATTN_LAYERS |
|------|------|-------------|
| M1 | MLP-baseline | — |
| M2 | JEPA-T | 2-4 |
| M3 | NCA | 0 |
| M4 | TF-1L | 1 |
| M5 | TF-2L | 2 (champion) |
| M6 | TF-3L+ | 3-4 |

### Grid: 12 × 6 = 72 cells minimum
With 5 sanctioned seeds = **360 lane-runs**

## 🔧 ОПЕРАЦИИ

### 1. FLEET REDEPLOY (критично когда all stale)

Для каждого аккаунта (acc0-acc5):
```bash
# 1. Удалить старый сервис
railway service delete --service-id <SERVICE_ID> --yes

# 2. Создать новый с правильным image
railway service create \
  --image ghcr.io/ghashtag/trios-seed-agent-real:sha-69b2d72 \
  --name seed-agent-real

# 3. Установить переменные
railway variables set NEON_DATABASE_URL="$NEON_DATABASE_URL"
railway variables set RAILWAY_ACC=accX
railway variables set TRAINER_KIND=external
```

### 2. ENQUEUE WAVE — добавить эксперименты

Шаблон SQL для вставки:
```sql
INSERT INTO experiment_queue
  (canon_name, config_json, priority, seed, steps_budget, account, status, created_by)
VALUES
  ('IGLA-<FORMAT>-<ARCH>-H<HIDDEN>-LR<LR>-S<SEED>',
   '{"format":"<FORMAT>","arch":"<ARCH>","hidden":<HIDDEN>,"lr":<LR>}'::jsonb,
   <PRIORITY>, <SEED>, <STEPS>, 'accX', 'pending', 'igla-skill');
```

**Приоритеты:**
- 99 — RECOVERY/PROMOTION (срочно)
- 95-90 — MEGA-WAVE (исследование gaps)
- 85-80 — EXPLORATORY
- 70-75 — LOW PRIORITY

### 3. GATE-2 CHECK — eligibility query

```sql
-- Проверить Gate-2 eligibility
WITH gate2_candidates AS (
  SELECT eq.id, eq.canon_name, eq.final_bpb, eq.final_step, eq.account,
         COUNT(bs.step) as bpb_sample_count,
         COUNT(DISTINCT bs.step) as distinct_steps
  FROM experiment_queue eq
  LEFT JOIN bpb_samples bs ON eq.canon_name = bs.canon_name AND eq.seed = bs.seed
  WHERE eq.status = 'done'
    AND eq.final_bpb < 1.85
    AND eq.final_step >= 4000
    AND eq.final_bpb >= 0.5
    AND eq.config_json NOT LIKE '%mock%'
  GROUP BY eq.id, eq.canon_name, eq.final_bpb, eq.final_step, eq.account
)
SELECT * FROM gate2_candidates
WHERE distinct_steps >= 5
ORDER BY final_bpb ASC;
```

## 📈 MONITORING QUERIES

### Best performance by cell (Numbers × Models)
```sql
SELECT
  config_json->>'format' as number_format,
  config_json->>'arch' as model,
  config_json->>'hidden' as hidden,
  MIN(final_bpb) as best_bpb,
  COUNT(*) as configs_tested
FROM experiment_queue
WHERE status = 'done'
  AND final_step >= 4000
  AND config_json NOT LIKE '%mock%'
GROUP BY number_format, model, hidden
ORDER BY best_bpb ASC;
```

### Format leaderboard
```sql
SELECT
  config_json->>'format' as format,
  MIN(final_bpb) as best_bpb,
  COUNT(*) as configs,
  MIN(final_step) as at_step
FROM experiment_queue
WHERE status = 'done'
  AND final_step >= 4000
  AND config_json NOT LIKE '%mock%'
  AND config_json->>'format' IS NOT NULL
GROUP BY format
ORDER BY best_bpb ASC;
```

## 🎯 RECOMMENDED WAVES

### Wave 1: MEGA-GAP-H (неисследованные hidden sizes)
```sql
-- h ∈ {1280, 1408, 1664, 1792, 1920, 2560} × 5 seeds
-- Priority: 95, ~36 configs
```

### Wave 2: MEGA-FMT (все форматы на h=1024)
```sql
-- 11 formats × 3 seeds = 33 configs
-- Priority: 90
```

### Wave 3: MEGA-ATTN (attention layers sweep)
```sql
-- attn_layers ∈ {0,2,4,6,8,12} on h=1024
-- Priority: 95, ~18 configs
```

## 🚨 ALERT CONDITIONS

| Condition | Severity | Action |
|-----------|----------|--------|
| fleet_alive (60s) < 4/6 | CRITICAL | Redeploy workers |
| queue_pending > 200 AND running < 3 | CRITICAL | Redeploy workers |
| zero bpb_samples for 15min | WARNING | Check trainer logs |
| Φ-1 collapse (5× identical bpb) | CRITICAL | Investigate trainer |

## 🔗 LINKS

- Competition Matrix: https://perplexity.ai/computer/tasks/plan-eksperimentov-i-obuchenie-4SFEIMDGS6u0Mtnmfar8vQ
- Gate-2 deadline: 2026-04-30T23:59Z
- Champion config: IGLA-TRAIN_V2-FP32-E0059-H2048-rng43 BPB=1.75

---

**Anchor:** φ² + φ⁻² = 3 · TRINITY · NEVER STOP
