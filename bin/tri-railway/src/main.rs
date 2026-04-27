//! `tri railway` — single-binary entry point.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.
//!
//! v0.0.1 ships the wiring for these subcommands:
//!
//!   tri-railway version
//!   tri-railway audit migrate-sql        # prints DDL to stdout
//!   tri-railway experience append ...    # writes a single L7 line
//!
//! Read/mutation verbs (`list`, `create`, `deploy`, `delete`, `logs`,
//! `audit run`) land under issues #4..#9; the structure here is set up
//! so each verb plugs in as one new `match` arm.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use trios_railway_audit::migrations;
use trios_railway_core::{
    mutations as M, queries as Q, Client, EnvironmentId, ProjectId, RailwayHash, ServiceId,
};
use trios_railway_experience::{append_line, ExperienceLine};

const IGLA_PROJECT_ID: &str = "e4fe33bb-3b09-4842-9782-7d2dea1abc9b";
const IGLA_PROD_ENV_ID: &str = "54e293b9-00a9-4102-814d-db151636d96e";
const DEFAULT_TRAINER_IMAGE: &str = "ghcr.io/ghashtag/trios-trainer-igla:latest";

#[derive(Parser, Debug)]
#[command(
    name = "tri-railway",
    version,
    about = "Manage Railway services for the IGLA project + online audit.",
    long_about = "tri railway: companion CLI to trios-trainer-igla and trios-mcp.\n\
                  R-rules: R1 (Rust-only), R5 (honest exit codes),\n\
                  R7 (every mutation seals an audit triplet),\n\
                  R9 (igla check before any mutation),\n\
                  L7 (experience log), L21 (context immutability)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)]
enum Cmd {
    /// Print the version and exit.
    Version,

    /// Audit operations.
    Audit {
        #[command(subcommand)]
        sub: AuditCmd,
    },

    /// Local experience log helpers (L7).
    Experience {
        #[command(subcommand)]
        sub: ExperienceCmd,
    },

    /// Service operations against Railway (RW-02 + RW-03).
    ///
    /// Requires `RAILWAY_TOKEN` in the environment. UUID-shaped tokens
    /// are auto-detected as project-access tokens.
    Service {
        #[command(subcommand)]
        sub: ServiceCmd,
    },
}

#[derive(Subcommand, Debug)]
enum ServiceCmd {
    /// Print all services in the configured project.
    List {
        #[arg(long, env = "TRIOS_RAILWAY_PROJECT", default_value = IGLA_PROJECT_ID)]
        project: String,
    },
    /// Create a new image-backed service named `--name` with `--image`,
    /// upsert the variables, and trigger one redeploy. R7 audit triplet
    /// is appended to the local experience log.
    Deploy {
        #[arg(long, env = "TRIOS_RAILWAY_PROJECT", default_value = IGLA_PROJECT_ID)]
        project: String,
        #[arg(long, env = "TRIOS_RAILWAY_ENV", default_value = IGLA_PROD_ENV_ID)]
        environment: String,
        /// Service name (e.g. `trios-train-seed-43`).
        #[arg(long)]
        name: String,
        /// Docker image; defaults to the IGLA trainer image.
        #[arg(long, default_value = DEFAULT_TRAINER_IMAGE)]
        image: String,
        /// `KEY=VALUE` env pairs to upsert. Repeatable.
        #[arg(long = "var", value_name = "KEY=VALUE")]
        vars: Vec<String>,
        /// Reuse this existing service id instead of creating a new one.
        #[arg(long)]
        existing: Option<String>,
        /// If set, only print what would happen.
        #[arg(long)]
        dry_run: bool,
        /// Repo root for the experience log.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Trigger a redeploy of an existing service.
    Redeploy {
        #[arg(long, env = "TRIOS_RAILWAY_ENV", default_value = IGLA_PROD_ENV_ID)]
        environment: String,
        #[arg(long)]
        service: String,
    },
    /// Permanently delete a service.
    Delete {
        #[arg(long)]
        service: String,
        /// Confirm the destruction with `--yes`.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand, Debug)]
enum AuditCmd {
    /// Print idempotent DDL for the Neon schema (issue #6).
    MigrateSql,
}

#[derive(Subcommand, Debug)]
enum ExperienceCmd {
    /// Append one line to the daily `.trinity/experience/<YYYYMMDD>.trinity` file.
    Append {
        /// Repository root (defaults to current directory).
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Issue ref like `#1`.
        #[arg(long)]
        issue: String,
        /// PHI LOOP step.
        #[arg(long)]
        phi_step: String,
        /// Free-form task summary.
        #[arg(long)]
        task: String,
        /// Status string.
        #[arg(long, default_value = "OK")]
        status: String,
        /// Soul-name (humorous English, L11).
        #[arg(long, default_value = "RailRangerOne")]
        soul_name: String,
        /// Agent codename.
        #[arg(long, default_value = "GENERAL")]
        agent: String,
        /// Verb being recorded (used for the audit triplet).
        #[arg(long, default_value = "experience")]
        verb: String,
        /// Project id (defaults to the IGLA project).
        #[arg(
            long,
            env = "TRIOS_RAILWAY_PROJECT",
            default_value = "e4fe33bb-3b09-4842-9782-7d2dea1abc9b"
        )]
        project: String,
        /// Optional service id for the triplet.
        #[arg(long)]
        service: Option<String>,
        /// Token fingerprint (never the token itself).
        #[arg(long, default_value = "no-token")]
        token_fp: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .compact()
        .init();

    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Version => {
            println!("tri-railway {}", env!("CARGO_PKG_VERSION"));
        }
        Cmd::Audit {
            sub: AuditCmd::MigrateSql,
        } => {
            for stmt in migrations::ddl_statements() {
                println!("{stmt};");
            }
        }
        Cmd::Service { sub } => run_service(sub).await?,
        Cmd::Experience { sub } => match sub {
            ExperienceCmd::Append {
                root,
                issue,
                phi_step,
                task,
                status,
                soul_name,
                agent,
                verb,
                project,
                service,
                token_fp,
            } => {
                let project_id = ProjectId::from(project);
                let service_id = service.map(ServiceId::from);
                let hash = RailwayHash::seal(&verb, &project_id, service_id.as_ref(), &token_fp);
                let line = ExperienceLine::from_hash(
                    &agent, &soul_name, &issue, &task, &status, &phi_step, &hash,
                )?;
                let path = append_line(&root.join(".trinity"), &line).await?;
                println!("appended: {}", path.display());
            }
        },
    }

    Ok(())
}

fn parse_var(s: &str) -> Result<(String, String)> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("variable `{s}` is not in KEY=VALUE form"))?;
    if k.is_empty() {
        anyhow::bail!("empty variable name in `{s}`");
    }
    Ok((k.to_string(), v.to_string()))
}

async fn run_service(cmd: ServiceCmd) -> Result<()> {
    let client =
        Client::from_env().map_err(|e| anyhow::anyhow!("RAILWAY_TOKEN not set or invalid: {e}"))?;
    let token_fp = client.token_fingerprint();

    match cmd {
        ServiceCmd::List { project } => {
            let pid = ProjectId::from(project);
            let pv = Q::project_view(&client, &pid).await?;
            println!("project {} ({})", pv.name, pv.id);
            for s in pv.services() {
                println!("  {}  {}  {}", s.id, s.name, s.created_at);
            }
        }
        ServiceCmd::Deploy {
            project,
            environment,
            name,
            image,
            vars,
            existing,
            dry_run,
            root,
        } => {
            let pid = ProjectId::from(project);
            let eid = EnvironmentId::from(environment);
            let mut parsed = Vec::with_capacity(vars.len());
            for v in &vars {
                parsed.push(parse_var(v)?);
            }

            if dry_run {
                println!("DRY RUN: would deploy {name} from {image}");
                println!("  project   = {}", pid.as_str());
                println!("  env       = {}", eid.as_str());
                if let Some(eid) = &existing {
                    println!("  reuse svc = {eid}");
                }
                for (k, v) in &parsed {
                    println!("  var       = {k}={v}");
                }
                return Ok(());
            }

            let service_id: ServiceId = if let Some(eid) = existing {
                ServiceId::from(eid)
            } else {
                let created = M::service_create(&client, &pid, &name).await?;
                println!("created service {} ({})", created.name, created.id);
                ServiceId::from(created.id)
            };

            M::service_instance_set_image(&client, &service_id, &eid, &image).await?;
            println!("set image: {image}");

            for (k, v) in &parsed {
                M::variable_upsert(&client, &pid, &eid, &service_id, k, v).await?;
                println!("  var: {k}=<{}>", v.len());
            }

            let deploy_id = M::service_redeploy(&client, &service_id, &eid).await?;
            println!("redeploy triggered: {deploy_id}");

            // R7 triplet to local experience log.
            let hash = RailwayHash::seal("deploy", &pid, Some(&service_id), &token_fp);
            let line = ExperienceLine::from_hash(
                "GENERAL",
                "RailRangerOne",
                "#5",
                &format!("deploy {name} image={image}"),
                "OK",
                "PUSH",
                &hash,
            )?;
            let path = append_line(&root.join(".trinity"), &line).await?;
            println!("experience: {}", path.display());
        }
        ServiceCmd::Redeploy {
            environment,
            service,
        } => {
            let eid = EnvironmentId::from(environment);
            let sid = ServiceId::from(service);
            let deploy_id = M::service_redeploy(&client, &sid, &eid).await?;
            println!("redeploy triggered: {deploy_id}");
        }
        ServiceCmd::Delete { service, yes } => {
            if !yes {
                anyhow::bail!("refusing to delete service `{service}` without --yes");
            }
            let sid = ServiceId::from(service);
            M::service_delete(&client, &sid).await?;
            println!("deleted: {sid}");
        }
    }
    Ok(())
}
