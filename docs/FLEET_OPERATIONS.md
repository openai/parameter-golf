# Fleet Operations Guide — trios-railway seed-agent

> Anchor: `phi^2 + phi^-2 = 3`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Neon Database (igla_* tables)                       │
│  ├── experiment_queue  (pending → running → done)    │
│  ├── bpb_samples       (step-by-step telemetry)      │
│  └── workers           (heartbeat registration)      │
└───────────────┬─────────────────────────────────────┘
                │ pull-based
┌───────────────▼─────────────────────────────────────┐
│  Railway Fleet (6 accounts × N workers)              │
│  acc0..acc5  each runs seed-agent containers         │
│  Image: ghcr.io/ghashtag/trios-seed-agent-real       │
│  Each worker: pulls experiment → runs trios-train     │
│              → pushes bpb_samples → marks done        │
└─────────────────────────────────────────────────────┘
```

## Accounts

| Account | Project ID | Environment ID | Token Env Var |
|---------|-----------|---------------|---------------|
| acc0 | `abdf752c-20ac-4813-a586-04a031db96e8` | `133b53bc-ff89-4000-a57c-3d717fa987a0` | `RAILWAY_TOKEN_ACC0` |
| acc1 | `e4fe33bb-3b09-4842-9782-7d2dea1abc9b` | `54e293b9-00a9-4102-814d-db151636d96e` | `RAILWAY_TOKEN_ACC1` |
| acc2 | `12c508c7-1196-468d-b06d-d8de8cb77e93` | `441bd3a6-f6d8-455e-b567-376b7538e9f1` | `RAILWAY_TOKEN_ACC2` |
| acc3 | `8ab06401-aa28-4af7-9faf-39a1548b7008` | `cd2d987b-dbbb-49ba-953b-f5e9486b906c` | `RAILWAY_TOKEN_ACC3` |
| acc4 | `0247abaa-6487-4347-811c-168d7fe53078` | `336c41a9-0d6a-4308-b266-1df6c91590ac` | `RAILWAY_TOKEN_ACC4` |
| acc5 | `8c9e1b0c-7ad8-4c9b-9a1e-2f0e5c6d8b9f` | `7a1e23b4-5c6d-4e7f-8a9b-0c1d2e3f4g5h` | `RAILWAY_TOKEN_ACC5` |

## Quick Commands

### Check experiment queue status

```bash
source .env
psql "$NEON_DATABASE_URL" -c "
SELECT status, account, COUNT(*) 
FROM experiment_queue 
GROUP BY status, account 
ORDER BY status, account;
"
```

### Enqueue an experiment

```bash
psql "$NEON_DATABASE_URL" -c "
INSERT INTO experiment_queue (canon_name, config_json, priority, seed, steps_budget, account, status)
VALUES (
  'IGLA-TRAIN_V2-GF16-E0800-H828-C12-LR0004-rng1597',
  '{\"seed\":1597,\"hidden\":828,\"ctx\":12,\"lr\":0.0004,\"steps\":1000}',
  50, 1597, 1000, 'acc1', 'pending'
) RETURNING id, canon_name;
"
```

### Check worker status

```bash
psql "$NEON_DATABASE_URL" -c "
SELECT account, worker_id, 
       COUNT(*) FILTER (WHERE heartbeat > NOW() - INTERVAL '5 minutes') AS alive,
       COUNT(*) FILTER (WHERE heartbeat BETWEEN NOW() - INTERVAL '30 minutes' AND NOW() - INTERVAL '5 minutes') AS stale,
       COUNT(*) FILTER (WHERE heartbeat < NOW() - INTERVAL '30 minutes') AS dead
FROM workers
GROUP BY account, worker_id
ORDER BY account;
"
```

### Deploy a worker to an account

Using the MCP tool (recommended):

```
railway_service_deploy(
  name: "trios-train-v2-acc{N}-s1597",
  project: "<project_id>",
  environment: "<env_id>",
  image: "ghcr.io/ghashtag/trios-seed-agent-real:latest",
  vars: [
    {key: "NEON_DATABASE_URL", value: "<neon_url>"},
    {key: "RAILWAY_ACC", value: "acc{N}"},
    {key: "TRAINER_KIND", value: "external"},
    {key: "RAILWAY_SERVICE_NAME", value: "trios-train-v2-acc{N}-s1597"}
  ]
)
```

Using curl directly:

```bash
source .env

ACC=acc1  # change per account
TOKEN_VAR="RAILWAY_TOKEN_${ACC^^}"
TOKEN="${!TOKEN_VAR}"
PROJ_VAR="RAILWAY_PROJECT_ID_${ACC^^}"
PID="${!PROJ_VAR}"
ENV_VAR="RAILWAY_ENVIRONMENT_ID_${ACC^^}"
EID="${!ENV_VAR}"
SVC_NAME="trios-train-v2-${ACC}-s1597"
IMAGE="ghcr.io/ghashtag/trios-seed-agent-real:latest"

# Step 1: Create service
SVC_ID=$(curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Project-Access-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { serviceCreate(input: {name: \\\"$SVC_NAME\\\", projectId: \\\"$PID\\\", environmentId: \\\"$EID\\\"}) { id } }\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['serviceCreate']['id'])")

echo "Service: $SVC_ID"

# Step 2: Set image
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Project-Access-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { serviceInstanceUpdate(serviceId: \\\"$SVC_ID\\\", environmentId: \\\"$EID\\\", input: {source: {image: \\\"$IMAGE\\\"}}) }\"}"

# Step 3: Set env vars
for KV in \
  "NEON_DATABASE_URL=$NEON_DATABASE_URL" \
  "RAILWAY_ACC=$ACC" \
  "TRAINER_KIND=external" \
  "RAILWAY_SERVICE_NAME=$SVC_NAME"; do
  KEY="${KV%%=*}"
  VAL="${KV#*=}"
  curl -s -X POST https://backboard.railway.app/graphql/v2 \
    -H "Project-Access-Token: $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { variableUpsert(input: {projectId: \\\"$PID\\\", environmentId: \\\"$EID\\\", serviceId: \\\"$SVC_ID\\\", name: \\\"$KEY\\\", value: \\\"$VAL\\\"}) }\"}"
done

# Step 4: Redeploy
curl -s -X POST https://backboard.railway.app/graphql/v2 \
  -H "Project-Access-Token: $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { serviceInstanceRedeploy(serviceId: \\\"$SVC_ID\\\", environmentId: \\\"$EID\\\") }\"}"

echo "Deployed $SVC_NAME → $SVC_ID"
```

### Check experiment results

```bash
psql "$NEON_DATABASE_URL" -c "
SELECT id, canon_name, status, final_bpb, final_step, account
FROM experiment_queue
WHERE status IN ('done', 'failed')
ORDER BY id DESC
LIMIT 20;
"
```

### View BPB samples for an experiment

```bash
psql "$NEON_DATABASE_URL" -c "
SELECT step, bpb 
FROM bpb_samples 
WHERE canon_name = 'IGLA-TRAIN_V2-GF16-E0800-H828-C12-LR0004-rng1597'
ORDER BY step;
"
```

## Sanctioned Seeds

Only these seeds are allowed in the experiment queue:

| Seed | Name |
|------|------|
| 42, 43, 44 | Trinity baseline |
| 1597, 2584, 4181, 6765 | Fibonacci F₁₇..F₂₀ |
| 10001–10010 | Extended range |
| 10946 | Fibonacci F₂₁ |

## Config JSON Fields

```json
{
  "seed": 1597,
  "hidden": 828,
  "ctx": 12,
  "lr": 0.0004,
  "steps": 1000
}
```

- `hidden`: hidden layer size (512, 828, 1024)
- `ctx`: context window (12)
- `lr`: learning rate (0.0004)
- `steps`: training steps (overrides `steps_budget`)

## Troubleshooting

### Worker not registering
- Check Railway deployment logs for `ENETUNREACH` → Neon connection retry handles this
- Check `channel_binding=require` is NOT in `NEON_DATABASE_URL`
- Verify `RAILWAY_ACC` matches an account in the `workers` table

### step=0, bpb=NaN
- Fixed in commit `69b2d723` — `ExternalTrainer` now passes correct CLI args
- If still happening, check `TRAINER_KIND=external` is set
- Check `trios-train` binary exists at `/usr/local/bin/trios-train` in the container

### ACC3 "Not Authorized"
- Token may have expired or been regenerated
- Check `RAILWAY_TOKEN_KIND_ACC3=project` in `.env`
