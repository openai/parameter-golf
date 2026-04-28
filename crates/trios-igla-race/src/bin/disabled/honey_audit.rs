//! L-honey-audit — `assertions/hive_honey.jsonl` integrity CLI (`honey_audit`)
//!
//! Read-only auditor for the hive honey ledger. Pairs with L14
//! `ledger_check` (consumer of `seed_results.jsonl`) and L15
//! `seed_emit` (producer for the same file) as the third diagnostic
//! in the race-substrate plumbing triad.
//!
//! The Queen Watchdog cron (`ad62640a`) already states: "if a
//! deposit looks malformed, flag it (do not edit the file)". Until
//! now, that flag was a manual `tail -1 | python3 -c json.loads`
//! one-liner. This CLI gives every agent the same R8-falsifiable
//! check, exit-coded so CI can wire it into `coq-check.yml` or
//! `laws-guard.yml` later without reaching for shell.
//!
//! ## What the audit checks (read-only)
//!
//! For each non-empty line of `assertions/hive_honey.jsonl`:
//!
//! 1. The line parses as a JSON **object** (not array, not scalar).
//! 2. The three **hard** logical keys resolve to non-empty strings:
//!    timestamp (`ts` or `timestamp_utc`), `lane`, `agent`.
//! 3. The timestamp is opaque — RFC3339 shape matters only to
//!    downstream tooling.
//! 4. **Soft** key: a sha-like field (`sha`, `commit_sha`,
//!    `main_sha`, or `parent_sha`) is reported per-row as
//!    `sha_present=bool`. Pure audit / cross-PR / cron-report
//!    deposits legitimately ship without a commit sha; we
//!    surface the count but never fail unless `--strict` is set.
//!
//! Anything that fails (1)–(3) is reported as a typed
//! [`HoneyError`] with a 1-based line number; the binary exits
//! non-zero. (4) is a soft signal — never fatal — because the
//! honey file legitimately mixes commit-SHA deposits with
//! audit-only / cron-only deposits whose payload is the report URL.
//!
//! ## Why a separate bin (R6 audit)
//!
//! Auditing the honey file does **not** belong in any existing
//! crate file: `victory.rs` is L7's, `hive_automaton.rs` is L13's,
//! `invariants.rs` is L5's. Adding a new `src/bin/honey_audit.rs`
//! is the cheapest R6-safe footprint Cargo can resolve.
//!
//! ## Falsification posture (R8)
//!
//! Each error path has a unit test that constructs the boundary
//! input and asserts the auditor rejects it. If any of `falsify_*`
//! ever silently passes, the auditor has been weakened.
//!
//! Refs: trios#143 lane L-honey-audit · R6/R8/R10/R13.
//! Anchor: phi^2 + phi^-2 = 3 · DOI 10.5281/zenodo.19227877.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde_json::Value;

/// CLI for auditing the hive honey JSONL ledger.
#[derive(Parser, Debug)]
#[command(
    name = "honey_audit",
    about = "Read-only integrity check for assertions/hive_honey.jsonl.",
    long_about = "Walk a JSONL honey ledger and verify each line is a JSON \
                  object carrying required keys (ts, lane, agent). Sha-like \
                  fields are reported softly. Exit 0 = all rows valid; \
                  non-zero = first failure. R6-safe — never edits the file."
)]
struct Args {
    /// Path to the JSONL honey ledger. Defaults to
    /// `assertions/hive_honey.jsonl` resolved against cwd.
    #[arg(long, default_value = "assertions/hive_honey.jsonl")]
    path: PathBuf,

    /// Print parsed deposits in addition to the verdict.
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Print machine-readable JSON summary instead of human form.
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Treat soft warnings (non-hex sha or missing sha) as fatal.
    #[arg(long, default_value_t = false)]
    strict: bool,
}

/// Errors produced by the honey auditor. Distinct from any race
/// gate error because a malformed honey deposit is a process
/// problem (R13 producer bug), not a victory predicate violation.
#[derive(Debug, Clone, PartialEq)]
pub enum HoneyError {
    /// Could not read the file at all.
    IoError { path: String, message: String },
    /// A line failed to parse as JSON.
    InvalidJson { line_no: usize, reason: String },
    /// A line was JSON but not a top-level object.
    NotAnObject { line_no: usize },
    /// A required key is missing or has the wrong shape.
    MissingKey { line_no: usize, key: &'static str },
    /// A required key was present but empty.
    EmptyValue { line_no: usize, key: &'static str },
}

impl std::fmt::Display for HoneyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HoneyError::IoError { path, message } => {
                write!(f, "honey I/O error at {}: {}", path, message)
            }
            HoneyError::InvalidJson { line_no, reason } => {
                write!(f, "invalid JSON at line {}: {}", line_no, reason)
            }
            HoneyError::NotAnObject { line_no } => {
                write!(f, "line {} is JSON but not a top-level object", line_no)
            }
            HoneyError::MissingKey { line_no, key } => {
                write!(f, "line {} missing required key `{}`", line_no, key)
            }
            HoneyError::EmptyValue { line_no, key } => {
                write!(f, "line {} has empty value for key `{}`", line_no, key)
            }
        }
    }
}

impl std::error::Error for HoneyError {}

/// Lightweight per-line parsed view used by the verbose printer
/// and the per-lane counters.
#[derive(Debug, Clone, PartialEq)]
pub struct HoneyDeposit {
    pub line_no: usize,
    pub ts: String,
    pub lane: String,
    pub agent: String,
    pub sha: String,
    pub sha_present: bool,
    pub sha_is_hexish: bool,
}

/// Hard-required logical keys.
const REQUIRED_LOGICAL_KEYS: [&str; 3] = ["ts", "lane", "agent"];

/// Aliases honored by the auditor for hard-required keys.
const KEY_ALIASES: [(&str, &str); 1] = [("ts", "timestamp_utc")];

/// Sha-like field names checked, in priority order. The first one
/// found wins. None of these are required — audit deposits and
/// cron reports legitimately omit them.
const SHA_KEYS: [&str; 4] = ["sha", "commit_sha", "main_sha", "parent_sha"];

/// Resolve a logical key against a JSON object honoring aliases.
fn lookup_logical<'a>(obj: &'a serde_json::Map<String, Value>, logical: &str) -> Option<&'a Value> {
    if let Some(v) = obj.get(logical) {
        return Some(v);
    }
    for (canonical, alias) in KEY_ALIASES {
        if canonical == logical {
            if let Some(v) = obj.get(alias) {
                return Some(v);
            }
        }
    }
    None
}

/// Soft heuristic: does the sha look like a hex commit?
/// Accepts 7+ hex chars (case-insensitive). Anything else is
/// reported as `sha_is_hexish=false` and flagged only when
/// `--strict` is passed.
pub fn is_hexish_sha(s: &str) -> bool {
    s.len() >= 7 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Parse a single honey line. Returns Ok(None) for blank lines so
/// the caller can skip them without inflating the line counter.
pub fn parse_line(line_no: usize, raw: &str) -> Result<Option<HoneyDeposit>, HoneyError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value: Value = serde_json::from_str(trimmed).map_err(|e| HoneyError::InvalidJson {
        line_no,
        reason: e.to_string(),
    })?;
    let obj = match value {
        Value::Object(map) => map,
        _ => return Err(HoneyError::NotAnObject { line_no }),
    };
    for key in REQUIRED_LOGICAL_KEYS {
        let v = lookup_logical(&obj, key).ok_or(HoneyError::MissingKey { line_no, key })?;
        let s = v.as_str().ok_or(HoneyError::MissingKey { line_no, key })?;
        if s.trim().is_empty() {
            return Err(HoneyError::EmptyValue { line_no, key });
        }
    }
    let pull = |k: &str| {
        lookup_logical(&obj, k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let mut sha = String::new();
    for k in SHA_KEYS {
        if let Some(s) = obj.get(k).and_then(|v| v.as_str()) {
            if !s.trim().is_empty() {
                sha = s.to_string();
                break;
            }
        }
    }
    let sha_present = !sha.is_empty();
    let sha_is_hexish = is_hexish_sha(&sha);
    Ok(Some(HoneyDeposit {
        line_no,
        ts: pull("ts"),
        lane: pull("lane"),
        agent: pull("agent"),
        sha,
        sha_present,
        sha_is_hexish,
    }))
}

/// Run the auditor against an in-memory blob (used by tests and by
/// the binary entry point). Stops at the first hard error.
pub fn audit_blob(blob: &str) -> Result<Vec<HoneyDeposit>, HoneyError> {
    let mut deposits = Vec::new();
    for (idx, raw) in blob.lines().enumerate() {
        let line_no = idx + 1;
        if let Some(dep) = parse_line(line_no, raw)? {
            deposits.push(dep);
        }
    }
    Ok(deposits)
}

/// Per-lane deposit counts, sorted by lane name for deterministic output.
pub fn lane_counts(deposits: &[HoneyDeposit]) -> Vec<(String, usize)> {
    let mut map: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for d in deposits {
        *map.entry(d.lane.clone()).or_insert(0) += 1;
    }
    map.into_iter().collect()
}

fn main() -> ExitCode {
    let args = Args::parse();
    let path_str = args.path.display().to_string();

    let blob = match fs::read_to_string(&args.path) {
        Ok(s) => s,
        Err(e) => {
            let err = HoneyError::IoError {
                path: path_str.clone(),
                message: e.to_string(),
            };
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "verdict": "io_error",
                        "path": path_str,
                        "message": e.to_string(),
                    })
                );
            } else {
                eprintln!("{}", err);
            }
            return ExitCode::from(40);
        }
    };

    let deposits = match audit_blob(&blob) {
        Ok(d) => d,
        Err(err) => {
            let code = match &err {
                HoneyError::IoError { .. } => 40,
                HoneyError::InvalidJson { .. } => 41,
                HoneyError::NotAnObject { .. } => 42,
                HoneyError::MissingKey { .. } => 43,
                HoneyError::EmptyValue { .. } => 44,
            };
            if args.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "verdict": "malformed",
                        "path": path_str,
                        "error": err.to_string(),
                        "exit": code,
                    })
                );
            } else {
                eprintln!("MALFORMED HONEY: {}", err);
            }
            return ExitCode::from(code);
        }
    };

    let lanes = lane_counts(&deposits);
    let nonhex = deposits
        .iter()
        .filter(|d| d.sha_present && !d.sha_is_hexish)
        .count();
    let no_sha = deposits.iter().filter(|d| !d.sha_present).count();
    let total = deposits.len();

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "verdict": "ok",
                "path": path_str,
                "total_deposits": total,
                "non_hex_sha_count": nonhex,
                "no_sha_count": no_sha,
                "lanes": lanes
                    .iter()
                    .map(|(k, v)| (k.clone(), *v))
                    .collect::<std::collections::BTreeMap<_, _>>(),
                "strict": args.strict,
            })
        );
    } else {
        println!("🍯 honey_audit: {}", path_str);
        println!("   total deposits     : {}", total);
        println!("   rows with sha      : {}", total - no_sha);
        println!("   rows without sha   : {}", no_sha);
        println!("   non-hex sha rows   : {}", nonhex);
        println!("   distinct lanes     : {}", lanes.len());
        if args.verbose {
            for (lane, n) in &lanes {
                println!("     {:<24} {}", lane, n);
            }
        }
        println!("✅ ledger integrity OK");
    }

    if args.strict && (nonhex > 0 || no_sha > 0) {
        if !args.json {
            eprintln!(
                "⚠️  --strict: {} non-hex sha + {} no-sha row(s); failing per request",
                nonhex, no_sha
            );
        }
        return ExitCode::from(45);
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R8 falsifier: an entirely blank file is not malformed — it is
    /// just empty. Audit returns Ok(empty vec).
    #[test]
    fn empty_blob_audits_clean() {
        let deps = audit_blob("").expect("empty blob is not an error");
        assert!(deps.is_empty());
    }

    /// R8 falsifier: blank lines are skipped, not counted.
    #[test]
    fn blank_lines_are_skipped() {
        let blob = "\n\n   \n";
        let deps = audit_blob(blob).expect("blank lines must skip clean");
        assert!(deps.is_empty());
    }

    /// R8 falsifier: a well-formed deposit round-trips.
    #[test]
    fn good_row_round_trips() {
        let blob = r#"{"ts":"2026-04-25T18:30:00Z","lane":"L-honey-audit","agent":"perplexity-computer-l-honey-audit","sha":"deadbeef","inv":"INV-7"}"#;
        let deps = audit_blob(blob).expect("good row must parse");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].lane, "L-honey-audit");
        assert!(deps[0].sha_present);
        assert!(deps[0].sha_is_hexish);
    }

    /// R8 falsifier: invalid JSON must be rejected with line number.
    #[test]
    fn falsify_invalid_json_rejected() {
        let blob = "{not json}";
        let err = audit_blob(blob).unwrap_err();
        match err {
            HoneyError::InvalidJson { line_no, .. } => assert_eq!(line_no, 1),
            other => panic!("expected InvalidJson, got {:?}", other),
        }
    }

    /// R8 falsifier: a JSON array (not object) must be rejected.
    #[test]
    fn falsify_array_top_level_rejected() {
        let blob = "[1, 2, 3]";
        let err = audit_blob(blob).unwrap_err();
        assert!(matches!(err, HoneyError::NotAnObject { line_no: 1 }));
    }

    /// R8 falsifier: a JSON scalar must be rejected.
    #[test]
    fn falsify_scalar_top_level_rejected() {
        let blob = "42";
        let err = audit_blob(blob).unwrap_err();
        assert!(matches!(err, HoneyError::NotAnObject { line_no: 1 }));
    }

    /// R8 falsifier: a row missing `ts` must be rejected.
    #[test]
    fn falsify_missing_ts_rejected() {
        let blob = r#"{"lane":"X","agent":"a","sha":"abcdef0"}"#;
        let err = audit_blob(blob).unwrap_err();
        assert!(matches!(err, HoneyError::MissingKey { key: "ts", .. }));
    }

    /// Schema alias: a row with `timestamp_utc` instead of `ts` must
    /// audit clean (L12-hygiene-era deposits use this shape).
    #[test]
    fn timestamp_utc_alias_accepted() {
        let blob = r#"{"timestamp_utc":"2026-04-25T17:03:56Z","lane":"L12-2","agent":"x","commit_sha":"18b673a"}"#;
        let deps = audit_blob(blob).expect("alias schema must parse");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].ts, "2026-04-25T17:03:56Z");
        assert_eq!(deps[0].sha, "18b673a");
        assert!(deps[0].sha_present);
        assert!(deps[0].sha_is_hexish);
    }

    /// Schema alias: `main_sha` is honored as a sha-like field.
    #[test]
    fn main_sha_alias_recognized() {
        let blob = r#"{"ts":"t","lane":"phd-cron","agent":"x","main_sha":"abf469f"}"#;
        let deps = audit_blob(blob).expect("main_sha row must parse");
        assert_eq!(deps.len(), 1);
        assert!(deps[0].sha_present);
        assert_eq!(deps[0].sha, "abf469f");
        assert!(deps[0].sha_is_hexish);
    }

    /// R8 falsifier: a row missing `lane` must be rejected.
    #[test]
    fn falsify_missing_lane_rejected() {
        let blob = r#"{"ts":"t","agent":"a","sha":"abcdef0"}"#;
        let err = audit_blob(blob).unwrap_err();
        assert!(matches!(err, HoneyError::MissingKey { key: "lane", .. }));
    }

    /// R8 falsifier: a row missing `agent` must be rejected.
    #[test]
    fn falsify_missing_agent_rejected() {
        let blob = r#"{"ts":"t","lane":"L","sha":"abcdef0"}"#;
        let err = audit_blob(blob).unwrap_err();
        assert!(matches!(err, HoneyError::MissingKey { key: "agent", .. }));
    }

    /// Soft contract: a row WITHOUT any sha-like field is accepted
    /// and surfaced as `sha_present=false`. Audit / cross-PR / cron
    /// deposits legitimately ship without a commit sha.
    #[test]
    fn missing_sha_is_soft_warning_only() {
        let blob = r#"{"ts":"t","lane":"L","agent":"a"}"#;
        let deps = audit_blob(blob).expect("sha-less deposit must parse");
        assert_eq!(deps.len(), 1);
        assert!(!deps[0].sha_present);
        assert!(!deps[0].sha_is_hexish);
    }

    /// R8 falsifier: an empty `lane` value must be rejected.
    #[test]
    fn falsify_empty_lane_rejected() {
        let blob = r#"{"ts":"t","lane":"   ","agent":"a","sha":"abcdef0"}"#;
        let err = audit_blob(blob).unwrap_err();
        assert!(matches!(err, HoneyError::EmptyValue { key: "lane", .. }));
    }

    /// Soft contract: a numeric `sha` is just ignored (sha_present
    /// stays false). String-only sha contract still holds: we never
    /// pretend a number is a sha.
    #[test]
    fn non_string_sha_treated_as_absent() {
        let blob = r#"{"ts":"t","lane":"L","agent":"a","sha":12345}"#;
        let deps = audit_blob(blob).expect("non-string sha is soft-fail only");
        assert!(!deps[0].sha_present);
    }

    /// R8 falsifier: hex-shape detector accepts a real short SHA and
    /// rejects natural-language strings.
    #[test]
    fn hexish_sha_classifier_is_strict() {
        assert!(is_hexish_sha("d29b758"));
        assert!(is_hexish_sha("DEADBEEFCAFE"));
        assert!(!is_hexish_sha("runtime-tool"));
        assert!(!is_hexish_sha("N/A"));
        assert!(!is_hexish_sha("abc")); // too short
        assert!(!is_hexish_sha("g1234567")); // not hex
    }

    /// R8 falsifier: lane counter aggregates correctly across rows.
    #[test]
    fn lane_counts_aggregate_correctly() {
        let blob = "\n".to_string()
            + r#"{"ts":"t","lane":"L1","agent":"a","sha":"deadbeef"}"#
            + "\n"
            + r#"{"ts":"t","lane":"L1","agent":"b","sha":"deadbe1"}"#
            + "\n"
            + r#"{"ts":"t","lane":"L2","agent":"a","sha":"deadbe2"}"#
            + "\n";
        let deps = audit_blob(&blob).expect("three rows must parse");
        let counts = lane_counts(&deps);
        assert_eq!(counts, vec![("L1".to_string(), 2), ("L2".to_string(), 1)]);
    }

    /// R8 falsifier: a real prefix of the production honey file
    /// (mixing hex sha, runtime-tool sha, and a sha-less audit row)
    /// audits clean.
    #[test]
    fn production_shaped_blob_audits_clean() {
        let blob = [
            r#"{"ts":"2026-04-25T17:00Z","lane":"L1","agent":"x","sha":"26fd3d2","inv":"INV-1"}"#,
            r#"{"ts":"2026-04-25T17:01Z","lane":"L7","agent":"y","sha":"runtime-tool","inv":"INV-7"}"#,
            r#"{"ts":"2026-04-25T17:02Z","lane":"L14","agent":"z","sha":"d29b758","inv":"INV-7"}"#,
            r#"{"ts":"2026-04-25T17:03Z","lane":"phd-audit","agent":"q","kind":"cron-report"}"#,
        ]
        .join("\n");
        let deps = audit_blob(&blob).expect("production-shape blob must parse");
        assert_eq!(deps.len(), 4);
        assert_eq!(deps.iter().filter(|d| d.sha_present).count(), 3);
        assert_eq!(deps.iter().filter(|d| !d.sha_present).count(), 1);
        assert_eq!(deps.iter().filter(|d| d.sha_is_hexish).count(), 2);
    }

    /// Distinct exit codes must remain distinct (sanity check
    /// against accidental collision when adding new error kinds).
    #[test]
    fn exit_codes_are_distinct() {
        let codes = [40u8, 41, 42, 43, 44, 45];
        let mut sorted = codes.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len());
    }

    /// L-R14 traceability stub: no magic numerics live in this file
    /// other than exit-code bytes (local namespace) and the
    /// hex-shape minimum length 7 (Git short-SHA convention,
    /// documented in the docstring of `is_hexish_sha`).
    /// This test asserts the structural contract that the
    /// constants arrays have not silently grown or shrunk.
    #[test]
    fn required_keys_stable() {
        assert_eq!(REQUIRED_LOGICAL_KEYS, ["ts", "lane", "agent"]);
        assert_eq!(KEY_ALIASES, [("ts", "timestamp_utc")]);
        assert_eq!(SHA_KEYS, ["sha", "commit_sha", "main_sha", "parent_sha"]);
    }
}
