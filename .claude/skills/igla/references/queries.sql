-- IGLA Monitoring Queries

-- Fleet status aggregation
SELECT
  account,
  COUNT(*) as total_workers,
  SUM(CASE WHEN heartbeat > NOW() - INTERVAL '5 minutes' THEN 1 ELSE 0 END) as alive_5m,
  SUM(CASE WHEN heartbeat > NOW() - INTERVAL '60 seconds' THEN 1 ELSE 0 END) as alive_60s,
  MAX(heartbeat) as latest_heartbeat
FROM workers
WHERE account IN ('acc0', 'acc1', 'acc2', 'acc3', 'acc4', 'acc5')
GROUP BY account
ORDER BY account;

-- Gate-2 eligible candidates
SELECT
  id,
  canon_name,
  account,
  final_bpb,
  step,
  seed,
  COUNT(DISTINCT step) as distinct_steps,
  COUNT(*) as sample_count,
  CASE WHEN COUNT(DISTINCT step) >= 5 AND final_bpb < 1.85 AND step >= 4000 THEN 1 ELSE 0 END as qualified
FROM experiments
WHERE status = 'done'
  AND final_bpb IS NOT NULL
  AND final_bpb > 0.01 -- exclude anti-collapse floor
GROUP BY id, canon_name, account, final_bpb, step, seed
HAVING final_bpb < 1.85
ORDER BY final_bpb ASC
LIMIT 50;

-- Top performers by format (number)
SELECT
  SUBSTR(canon_name, POSITION('-' IN canon_name) + 1, 10) as format,
  COUNT(*) as configs_tested,
  MIN(final_bpb) as best_bpb,
  MIN(step) FILTER (WHERE final_bpb = MIN(final_bpb) OVER (PARTITION BY SUBSTR(canon_name, POSITION('-' IN canon_name) + 1, 10)) as best_step
FROM experiments
WHERE status = 'done'
  AND final_bpb IS NOT NULL
  AND final_bpb > 0.01
  AND step >= 4000
GROUP BY format
ORDER BY best_bpb ASC;

-- Top performers by model (architecture)
SELECT
  canon_name,
  COUNT(*) as configs_tested,
  MIN(final_bpb) as best_bpb,
  MIN(step) FILTER (WHERE final_bpb = MIN(final_bpb) OVER (PARTITION BY canon_name)) as best_step
FROM experiments
WHERE status = 'done'
  AND final_bpb IS NOT NULL
  AND final_bpb > 0.01
  AND step >= 4000
GROUP BY canon_name
ORDER BY best_bpb ASC;

-- Champion by cell (format × model)
SELECT
  format,
  model,
  MIN(final_bpb) as best_bpb,
  COUNT(*) as rows,
  MIN(step) FILTER (WHERE final_bpb = MIN(final_bpb) OVER (PARTITION BY format, model)) as best_step
FROM (
  SELECT
    SUBSTR(canon_name, POSITION('-' IN canon_name) + 1,
      CASE
        WHEN canon_name LIKE '%GF%' AND canon_name LIKE '%FP32%' THEN POSITION('FP32', canon_name) - 2
        WHEN canon_name LIKE '%GF%' THEN 10
        WHEN canon_name LIKE '%FP32%' THEN POSITION('FP32', canon_name)
        WHEN canon_name LIKE '%BF16%' THEN POSITION('BF16', canon_name)
        WHEN canon_name LIKE '%FP16%' THEN POSITION('FP16', canon_name)
        ELSE 0
      END) as format,
    canon_name as model,
    final_bpb,
    step
  FROM experiments
  WHERE status = 'done'
    AND final_bpb IS NOT NULL
    AND final_bpb > 0.01
) t
GROUP BY format, model
ORDER BY format, model;

-- Φ-1 collapse detection
SELECT
  final_bpb,
  COUNT(*) as identical_count,
  GROUP_CONCAT(id) as experiment_ids,
  GROUP_CONCAT(account) as accounts
FROM experiments
WHERE status = 'done'
  AND final_bpb IS NOT NULL
  AND final_bpb > 0.001 -- exclude near-zero
  AND final_bpb < 0.02 -- check for floor
GROUP BY final_bpb
HAVING COUNT(*) >= 3
ORDER BY identical_count DESC;
