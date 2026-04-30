---
name: igla
description: IGLA Race Manager — управляет стратегией воркеров, мониторингом и экспериментами одним запросом
type: command
---

IGLA Race Manager — unified command для управления всеми аспектами IGLA competition.

**Использование:**

`/igla` — показать текущее состояние (fleet, queue, leaderboard) и рекомендации

`/igla redeploy [acc0|acc1|acc2|acc3|all]` — redeploy worker(s) для восстановления dead fleet

`/igla queue --wave=<name>` — добавить новую волну экспериментов в queue

`/igla status --detailed` — детальный отчёт по competition matrix

`/igla champion --format=<gf16|fp32|...> --arch=<tf-2l|...>` — найти champion ячейку

---

**Что делает:**

1. **Fleet Health Check** — проверяет все 6 workers (acc0-acc5), показывает:
   - Last heartbeat timestamp
   - Stale status (если >5 минут без HB)
   - Current experiment (если есть)

2. **Queue Analysis** — показывает pending/running/done counts и estimated throughput

3. **Competition Matrix** — Numbers × Models grid с лучшими BPB per cell

4. **Recommendations** — что делать дальше (redeploy, new wave, kill bad configs)

---

**Примеры:**

```
/igla
→ показывает: 6 workers (3 dead), 363 pending, champion=TF-2L×FP32 (2.578 BPB)

/igla redeploy all
→ запускает redeploy всех 6 Railway services

/igla queue --wave=MEGA-EXOTIC
→ добавляет 30 JEPA-T/NCA/HYBRID экспериментов
```
