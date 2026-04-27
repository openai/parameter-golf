# AGENTS.md — trios-railway

Constitution: same as `gHashTag/trios` (`SOUL.md`, `CLAUDE.md`,
`AGENTS.md`, `LAWS.md`, `NOW.json`, eternal issue
[gHashTag/trios#143](https://github.com/gHashTag/trios/issues/143)).
Anchor: `phi^2 + phi^-2 = 3`.

## Scope of this repo

`trios-railway` is the **operator surface** for Railway. It does:

- typed Railway GraphQL queries and mutations
- online audit between Railway reality, the Neon `igla_*` ledger,
  and `.trinity/experience/`
- single-binary CLI `tri-railway` (one verb = one subcommand, L20)

It does **NOT**:

- touch trainer/JEPA/INV-* code (that lives in `trios-trainer-igla`)
- modify `.t27`/`.tri` specs (CANON_DE_ZIGFICATION)
- close eternal issue #143 (L10)

## Standing rules (binding)

- **R1** Rust-only.
- **R5** Honest exit codes; CLI never claims success on upstream failure.
- **R7** Every mutation seals an audit triplet:
  `RAIL=<verb> @ project=<8c> service=<8c> sha=<8c> ts=<rfc3339>`
- **R9** Mutations are gated by `igla check <sha>`.
- **L1** No `.sh` files (CI self-checks).
- **L2** Every PR `Closes #N`.
- **L3** Clippy zero warnings.
- **L4** Tests pass; new code carries new tests.
- **L7** Append a line to `.trinity/experience/` for every significant task.
- **L8** Push first.
- **L11** Pick a soul-name (humorous English) before mutation.
- **L21** `.trinity/experience/` is append-only; never truncate.

## Ring layout

```
crates/
├── trios-railway-core/         RW-00 identity types · RW-01 transport
├── trios-railway-audit/        AU-00 DDL · AU-01 drift detector
└── trios-railway-experience/   EX-00 append-only writer
bin/
└── tri-railway/                BR-CLI entry point
```

Each ring carries `README.md` + `TASK.md` + `AGENTS.md` describing its
local invariants (issue
[#11](https://github.com/gHashTag/trios-railway/issues/11)).

## Commit etiquette

```
feat(rw-01): typed list-services query

Closes #4
Agent: GENERAL
```

## Do not

- Touch `crates/trios-trainer-igla/*` — different repo entirely.
- Open browsers (`R7` of `NOW.json`); use `gh` CLI and the Neon connector.
- Hand-edit generated GraphQL response JSON; treat it as opaque bytes.
