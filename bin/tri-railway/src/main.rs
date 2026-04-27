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
use trios_railway_core::{ProjectId, RailwayHash, ServiceId};
use trios_railway_experience::{append_line, ExperienceLine};

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
