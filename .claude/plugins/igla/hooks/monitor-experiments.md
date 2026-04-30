---
name: monitor-experiments
description: Auto-trigger IGLA status check after experiment-related operations
events:
  - PostToolUse
---

After any tool use that could affect the IGLA experiment pipeline, trigger a quick status check.

**Trigger Conditions:**

The hook fires when any of these tools are used:
- `Bash` with commands containing: `railway`, `neon`, `psql`, `experiment`, `worker`, `seed-agent`
- `Write` to files in: `bin/seed-agent/`, `crates/igla-*/`, `.claude/plugins/igla/`
- `Skill` called with name containing: `igla`

**Action:**

When triggered, run a lightweight status check:

1. **Fleet Quick Check** — Query `workers` table for stale detection
2. **Queue Count** — Get pending/running/done counts from `experiment_queue`
3. **Notify user** if any issues detected (dead workers, stalled queue)

**Environment Variables Required:**

- `NEON_DATABASE_URL` — PostgreSQL connection string
- `RAILWAY_TOKEN` — Railway API token (for service status)

**SQL Queries:**

```sql
-- Quick fleet health
SELECT railway_acc,
       CASE
         WHEN EXTRACT(EPOCH FROM (now() - last_heartbeat))/60 > 10 THEN 'DEAD'
         WHEN EXTRACT(EPOCH FROM (now() - last_heartbeat))/60 > 5 THEN 'STALE'
         ELSE 'OK'
       END as health,
       EXTRACT(EPOCH FROM (now() - last_heartbeat))/60 as minutes_stale
FROM workers
ORDER BY railway_acc;

-- Queue summary
SELECT status, COUNT(*)
FROM experiment_queue
WHERE status IN ('pending', 'running', 'done', 'pruned', 'failed')
GROUP BY status;
```

**Output:**

If everything is healthy, do nothing (silent success).

If issues detected, show:

```
⚠️ IGLA Fleet Alert
Dead workers: acc0, acc2 (no heartbeat >10 min)
Pending queue: 363 experiments
Recommend: /igla redeploy all
```

**Never:**

- Block the original tool execution (run asynchronously)
- Flood the user with status on every small edit (only on significant operations)
- Connect to unauthorized services
