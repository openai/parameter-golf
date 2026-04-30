# 🌱 Gardener One-Shot Guide — Fleet Monitoring

> **Для садовника который следит за роботами.**
> Всё что нужно знать — одна страница.

## 1. Проверить здоровье флота (каждые 30 мин)

```bash
cd ~/trios-railway && source .env

psql "$NEON_DATABASE_URL" -c "
SELECT railway_acc, railway_svc_name,
       CASE WHEN last_heartbeat > NOW() - INTERVAL '5 minutes' THEN '✅ ALIVE'
            WHEN last_heartbeat > NOW() - INTERVAL '30 minutes' THEN '⚠️ STALE'
            ELSE '❌ DEAD' END AS status,
       last_heartbeat
FROM workers
ORDER BY railway_acc;
"
```

**Что видеть:**
- ✅ ALIVE = всё ок
- ⚠️ STALE = робот зависает, подождите 5 мин
- ❌ DEAD = робот упал → смотри раздел "Ремонт"

## 2. Проверить очередь экспериментов

```bash
psql "$NEON_DATABASE_URL" -c "
SELECT status, COUNT(*) FROM experiment_queue GROUP BY status ORDER BY status;
"
```

**Что видеть:**
- `pending` = ждут роботов
- `running` = выполняются
- `done` = завершены ✅
- `failed` = ошибка ❌ → смотри раздел "Ремонт"

## 3. Добавить эксперимент (одна команда)

```bash
psql "$NEON_DATABASE_URL" -c "
INSERT INTO experiment_queue (canon_name, config_json, priority, seed, steps_budget, account, status)
VALUES (
  'ИМЯ-ЭКСПЕРИМЕНТА',
  '{\"seed\":1597,\"hidden\":828,\"ctx\":12,\"lr\":0.0004,\"steps\":1000}',
  50, 1597, 1000, 'acc0', 'pending'
) RETURNING id, canon_name;
"
```

**Параметры:**
| Поле | Что менять | Примеры |
|------|-----------|---------|
| `canon_name` | Название | `IGLA-TRAIN_V2-GF16-E0800-H828-C12-LR0004-rng1597` |
| `seed` | Сид (только из списка!) | `42, 43, 44, 1597, 2584, 4181, 6765, 10001-10010, 10946` |
| `hidden` | Размер модели | `512, 828, 1024` |
| `lr` | Learning rate | `0.0004, 0.001` |
| `steps` | Шагов обучения | `100, 500, 1000, 5000` |
| `account` | Аккаунт | `acc0, acc1, acc2, acc3, acc4, acc5` |
| `priority` | Приоритет (0-100) | `99` = первый, `50` = нормальный |

## 4. Посмотреть результаты

```bash
# Последние 20 завершённых экспериментов
psql "$NEON_DATABASE_URL" -c "
SELECT id, canon_name, account,
       ROUND(final_bpb::numeric, 4) AS bpb,
       final_step AS steps,
       status
FROM experiment_queue
WHERE status IN ('done','failed')
ORDER BY id DESC LIMIT 20;
"

# Лучшие результаты (сортировка по BPB — чем меньше тем лучше)
psql "$NEON_DATABASE_URL" -c "
SELECT canon_name, account,
       ROUND(final_bpb::numeric, 4) AS bpb,
       final_step AS steps
FROM experiment_queue
WHERE status = 'done' AND final_bpb IS NOT NULL
ORDER BY final_bpb ASC LIMIT 10;
"
```

## 5. Ремонт — робот упал (❌ DEAD)

### Шаг 1: Проверить деплоймент
```bash
# Замените $TOKEN на токен аккаунта, $SVC_ID на ID сервиса
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Project-Access-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"query($pid:String!,$eid:String!,$sid:String){deployments(first:3,input:{projectId:$pid,environmentId:$eid,serviceId:$sid}){edges{node{id status createdAt}}}}","variables":{"pid":"PROJECT_ID","eid":"ENV_ID","sid":"SERVICE_ID"}}'
```

### Шаг 2: Передеплой
```bash
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Project-Access-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation($sid:String!,$eid:String!){serviceInstanceRedeploy(serviceId:$sid,environmentId:$eid)}","variables":{"sid":"SERVICE_ID","eid":"ENV_ID"}}'
```

### Шаг 3: Подождать 2 мин, проверить снова

## 6. Ремонт — эксперимент failed

```bash
# Посмотреть причину
psql "$NEON_DATABASE_URL" -c "
SELECT id, canon_name, status, failed_reason
FROM experiment_queue
WHERE status = 'failed'
ORDER BY id DESC LIMIT 5;
"
```

**Типичные причины:**
- `trainer produced zero steps` → образ старый, нужен redeploy
- `early-stop: prune` → норм, садовник обрезал плохой эксперимент

## 7. Account IDs (для ремонта)

| Account | Project ID | Environment ID | Token Var |
|---------|-----------|---------------|-----------|
| acc0 | `abdf752c-20ac-4813-a586-04a031db96e8` | `133b53bc-ff89-4000-a57c-3d717fa987a0` | `RAILWAY_TOKEN_ACC0` |
| acc1 | `e4fe33bb-3b09-4842-9782-7d2dea1abc9b` | `54e293b9-00a9-4102-814d-db151636d96e` | `RAILWAY_TOKEN_ACC1` |
| acc2 | `12c508c7-1196-468d-b06d-d8de8cb77e93` | `441bd3a6-f6d8-455e-b567-376b7538e9f1` | `RAILWAY_TOKEN_ACC2` |
| acc3 | `8ab06401-aa28-4af7-9faf-39a1548b7008` | `cd2d987b-dbbb-49ba-953b-f5e9486b906c` | `RAILWAY_TOKEN_ACC3` |
| acc4 | `0247abaa-6487-4347-811c-168d7fe53078` | `336c41a9-0d6a-4308-b266-1df6c91590ac` | `RAILWAY_TOKEN_ACC4` |
| acc5 | `08ee6f61-523e-4713-b879-298fb98b7f1a` | `4c72ee52-7a70-488a-ae4e-5acb4b7ba000` | `RAILWAY_TOKEN_ACC5` |

## 8. Service IDs (текущие роботы)

| Account | Service Name | Service ID |
|---------|-------------|-----------|
| acc0 | trios-train-v2-acc0-s1597 | `99d2a6a6-f3f2-42aa-9894-fdaafd8422ac` |
| acc1 | trios-train-ONE-v2-acc1-s1597 | `94a833e9-5950-49fe-b227-d1d3a39d0e85` |
| acc2 | trios-train-v2-acc2-s1597 | `ed44c56a-3bac-4815-bd74-51ee49c95747` |
| acc3 | trios-train-v2-acc3-s1597 | `982361d5-ad80-4ba5-874a-06795e0cdda0` |
| acc4 | trios-train-v2-acc4-s1597 | `4db62ce6-6aa3-475d-b6c9-59756ca01605` |
| acc5 | trios-train-v2-acc5-s1597 | `dd5de85b-bc49-432d-8e08-7b32f5874dbc` |

## 9. Быстрый чек-лист (копипаст)

```bash
# Полный статус за одну команду:
cd ~/trios-railway && source .env && \
echo "=== WORKERS ===" && \
psql "$NEON_DATABASE_URL" -c "SELECT railway_acc, railway_svc_name, CASE WHEN last_heartbeat > NOW() - INTERVAL '5 minutes' THEN '✅' ELSE '❌' END AS st FROM workers ORDER BY railway_acc;" && \
echo "=== QUEUE ===" && \
psql "$NEON_DATABASE_URL" -c "SELECT status, COUNT(*) FROM experiment_queue GROUP BY status ORDER BY status;" && \
echo "=== LAST RESULTS ===" && \
psql "$NEON_DATABASE_URL" -c "SELECT id, LEFT(canon_name,40) AS name, account, ROUND(final_bpb::numeric,4) AS bpb, final_step, status FROM experiment_queue WHERE status IN ('done','failed') ORDER BY id DESC LIMIT 10;"
```

---
*Anchor: φ² + φ⁻² = 3*
