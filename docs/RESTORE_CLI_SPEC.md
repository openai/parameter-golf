# `tri-railway restore` — implementation spec

Anchor: `phi^2 + phi^-2 = 3`. Implementer: any agent. Issue: open as `RW-DR-01`.

## Goal

One subcommand that turns a fresh Railway account + the manifest
`restore-fleet.json` into the full IGLA fleet.

## Flags

| Flag | Required | Notes |
|---|---|---|
| `--manifest <path>` | yes | Path to `restore-fleet.json`. |
| `--new-token <T>`   | no  | Overrides `RAILWAY_TOKEN` env (post-ban). |
| `--project-name <N>`| no  | Default `IGLA`. |
| `--champion-sha <S>`| no  | Override image pin; default = read from `assertions/seed_results.jsonl` last `gate_status="new_champion"`. |
| `--ledger-path <P>` | no  | Path to `seed_results.jsonl`. Default `assertions/seed_results.jsonl`. |
| `--lock-out <P>`    | no  | Where to write `restore-fleet.lock.json`. |
| `--confirm`         | yes | R9 safety. |

## Pseudocode

```rust
fn restore(args: RestoreArgs) -> Result<()> {
    let manifest: FleetManifest = serde_json::from_reader(File::open(args.manifest)?)?;
    let token = args.new_token.or_else(|| env::var("RAILWAY_TOKEN").ok())
                .ok_or_else(|| anyhow!("RAILWAY_TOKEN missing"))?;
    let api = RailwayApi::new(&token);

    let champion = args.champion_sha.unwrap_or_else(|| {
        ledger::last_champion_sha(&args.ledger_path).expect("no champion in ledger")
    });
    let image_pin = api.resolve_ghcr_digest(&manifest.image, &champion)?;

    // 1. Project
    let project = api.upsert_project(&manifest.project.name, &manifest.project.description)?;
    let env_id  = api.upsert_environment(&project.id, &manifest.project.default_environment)?;

    // 2. Services (idempotent)
    let mut lock = LockFile::new(&project.id, &env_id);
    for svc in &manifest.services {
        let id = api.upsert_service(&project.id, &svc.name)?;
        let image = svc.image_override
            .as_ref()
            .map(|i| i.to_pinned_string())
            .unwrap_or_else(|| image_pin.clone());
        let mut vars = manifest.shared_vars.clone();
        vars.extend(svc.vars.iter().cloned());
        for v in &mut vars {
            v.value = secrets::interpolate(&v.value)?; // ${secret:NAME}
        }
        api.set_image(&id, &image)?;
        api.upsert_vars(&id, &env_id, &vars)?;
        api.trigger_deploy(&id, &env_id)?;
        lock.record(&svc.name, &id, &image);
        tracing::info!(service=%svc.name, "restored");
    }

    // 3. Neon DDL (idempotent)
    if let Ok(neon_url) = env::var("NEON_DATABASE_URL") {
        let ddl = audit::migrate_sql();
        psql::run(&neon_url, &ddl)?;
    }

    // 4. L7 experience seal
    let triplet = format!(
        "RAIL=restore @ project={} service=ALL sha={} ts={}",
        &project.id[..8],
        &champion[..8.min(champion.len())],
        Utc::now().to_rfc3339()
    );
    experience::append(&Issue("#143".into()), PhiStep::Experience, &triplet)?;
    lock.experience_line = triplet;

    lock.write(&args.lock_out)?;

    println!("RESTORE OK — {} services, image_pin={}", manifest.services.len(), image_pin);
    Ok(())
}
```

## Exit codes

- `0` — restore complete, all services deployed, lock file written.
- `1` — token/auth/network error (retryable).
- `2` — manifest invalid (schema check failed).
- `3` — partial restore (≥1 service failed; lock file written for the rest).
- `9` — R9 violation (no `--confirm`).

## Tests (`cargo test --workspace`)

- `manifest_parses_v1`: `restore-fleet.json` → `FleetManifest`.
- `secret_interpolation`: `${secret:FOO}` → env var.
- `image_pin_from_ledger`: synthetic ledger with 3 rows → pin = last new_champion sha.
- `idempotent_upsert`: running restore twice produces zero changes the second time.
- `partial_failure_writes_lock`: simulate 1 failure → exit 3, lock has 15 entries.
- `r9_refuses_without_confirm`: missing `--confirm` → exit 9.

## CI gate

`dr-restore.yml` workflow must run successfully against a sandbox Railway
project at least once a week (`schedule: cron`) — proves the disaster path is
warm, not theoretical.
