//! Multi-account Railway routing — `RailwayMultiClient`.
//!
//! Closes the long-standing footgun where one MCP instance held a single
//! `RAILWAY_TOKEN` and silently mutated the wrong fleet whenever a tool
//! call landed on a project belonging to a different account. The
//! pattern instead is:
//!
//! - register each operator account by `AccountId` enum (`Acc0..Acc3`),
//!   pinning its credentials in a `secrecy::SecretString`,
//! - route every mutation through `RailwayMultiClient::get(id)` which
//!   returns the per-account `Client` (or a typed
//!   `RailwayError::NotAuthorized { account }` if the slot is empty),
//! - `Scope::{One(AccountId), All}` is the single argument that callers
//!   pass to fan-out helpers (e.g. `tri-gardener` fleet probes).
//!
//! Tokens never reach `Debug` / `Display` output. `secrecy` enforces it
//! by default and we add a regression test
//! (`debug_format_does_not_leak_token`) that fails loudly the moment
//! someone derives `Debug` over an `AccountCreds`.
//!
//! Anchor: `phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP`.

use secrecy::{ExposeSecret, SecretString};
use std::collections::BTreeMap;
use thiserror::Error;

use crate::ids::{EnvironmentId, ProjectId};
use crate::transport::{AuthMode, Client, ClientError};

/// Stable, ledger-friendly account label. Order is fixed so
/// `Scope::All` iterates in `acc0..acc3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AccountId {
    Acc0,
    Acc1,
    Acc2,
    Acc3,
}

impl AccountId {
    pub fn as_str(&self) -> &'static str {
        match self {
            AccountId::Acc0 => "acc0",
            AccountId::Acc1 => "acc1",
            AccountId::Acc2 => "acc2",
            AccountId::Acc3 => "acc3",
        }
    }

    pub fn all() -> [AccountId; 4] {
        [AccountId::Acc0, AccountId::Acc1, AccountId::Acc2, AccountId::Acc3]
    }

    /// Parse `"acc0"`/`"ACC1"`/`"acc-2"` etc. into the enum.
    pub fn from_alias(s: &str) -> Option<AccountId> {
        let normalized: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        match normalized.as_str() {
            "acc0" | "0" => Some(AccountId::Acc0),
            "acc1" | "1" => Some(AccountId::Acc1),
            "acc2" | "2" => Some(AccountId::Acc2),
            "acc3" | "3" => Some(AccountId::Acc3),
            _ => None,
        }
    }
}

/// Per-account credentials. `token` is held in `SecretString` and never
/// renders via `Debug` / `Display`.
#[derive(Clone)]
pub struct AccountCreds {
    pub token: SecretString,
    pub project: ProjectId,
    pub env: EnvironmentId,
    pub auth: AuthMode,
}

impl std::fmt::Debug for AccountCreds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccountCreds")
            .field("project", &self.project)
            .field("env", &self.env)
            .field("auth", &self.auth)
            // Token deliberately redacted. R5: never silent — show the
            // length only, so the operator can confirm a non-empty
            // secret without leaking it.
            .field("token", &format!("<redacted len={}>", self.token.expose_secret().len()))
            .finish()
    }
}

/// Routing scope for a single mutation call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Single-account: caller specifies which account to touch.
    One(AccountId),
    /// All registered accounts (used by fleet probes).
    All,
}

impl Scope {
    pub fn iter(&self) -> Vec<AccountId> {
        match self {
            Scope::One(a) => vec![*a],
            Scope::All => AccountId::all().to_vec(),
        }
    }
}

#[derive(Debug, Error)]
pub enum RailwayError {
    #[error("account not authorized: {account:?} (no creds registered)")]
    NotAuthorized { account: AccountId },
    #[error("client error: {0}")]
    Client(#[from] ClientError),
}

impl PartialEq for RailwayError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                RailwayError::NotAuthorized { account: a },
                RailwayError::NotAuthorized { account: b },
            ) => a == b,
            (RailwayError::Client(_), RailwayError::Client(_)) => true,
            _ => false,
        }
    }
}

/// Multi-tenant Railway client.
///
/// Holds one `Client` per registered `AccountId`. Lookup by `AccountId`
/// (`get`) returns a borrowed `&Client` ready to make GraphQL calls.
/// Lookup of the credentials triple by `AccountId` (`creds`) is also
/// available for tools that need to know which `(project, env)` they
/// are about to mutate.
#[derive(Default)]
pub struct RailwayMultiClient {
    clients: BTreeMap<AccountId, (AccountCreds, Client)>,
}

impl std::fmt::Debug for RailwayMultiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RailwayMultiClient")
            .field("registered", &self.registered())
            .finish()
    }
}

impl RailwayMultiClient {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an account. Returns `Err` if the credentials cannot be
    /// converted into a `Client` (e.g. empty token, malformed UUID).
    pub fn register(
        &mut self,
        id: AccountId,
        creds: AccountCreds,
    ) -> Result<(), RailwayError> {
        let token = creds.token.expose_secret().to_string();
        let client = Client::with_token_and_mode(token, creds.auth)?;
        self.clients.insert(id, (creds, client));
        Ok(())
    }

    pub fn get(&self, id: AccountId) -> Result<&Client, RailwayError> {
        self.clients
            .get(&id)
            .map(|(_, c)| c)
            .ok_or(RailwayError::NotAuthorized { account: id })
    }

    pub fn creds(&self, id: AccountId) -> Result<&AccountCreds, RailwayError> {
        self.clients
            .get(&id)
            .map(|(c, _)| c)
            .ok_or(RailwayError::NotAuthorized { account: id })
    }

    pub fn registered(&self) -> Vec<AccountId> {
        self.clients.keys().copied().collect()
    }

    /// Read all four account slots from `RAILWAY_TOKEN_ACC{0..3}` plus
    /// the matching `_PROJECT_ID_ACC{N}` / `_ENVIRONMENT_ID_ACC{N}` /
    /// `_TOKEN_KIND_ACC{N}` triple. Slots without a token are skipped
    /// silently — operator may register fewer than 4 accounts.
    ///
    /// `_TOKEN_KIND_ACC{N}`:
    /// - `"team"` / `"personal"` / `"bearer"` → `AuthMode::Team`
    /// - `"project"` → `AuthMode::Project`
    /// - missing → defaults to `Team` if the token does not look UUID-like, else `Project`
    pub fn from_env() -> Result<Self, RailwayError> {
        let mut mc = Self::default();
        for id in AccountId::all() {
            let suffix = match id {
                AccountId::Acc0 => "ACC0",
                AccountId::Acc1 => "ACC1",
                AccountId::Acc2 => "ACC2",
                AccountId::Acc3 => "ACC3",
            };
            let token = match std::env::var(format!("RAILWAY_TOKEN_{suffix}")) {
                Ok(t) if !t.is_empty() => t,
                _ => continue, // slot not configured
            };
            let project = std::env::var(format!("RAILWAY_PROJECT_ID_{suffix}"))
                .unwrap_or_default();
            let env = std::env::var(format!("RAILWAY_ENVIRONMENT_ID_{suffix}"))
                .unwrap_or_default();
            let kind = std::env::var(format!("RAILWAY_TOKEN_KIND_{suffix}"))
                .unwrap_or_default();
            let auth = parse_auth_mode(&token, &kind);
            let creds = AccountCreds {
                token: SecretString::from(token),
                project: ProjectId::from(project),
                env: EnvironmentId::from(env),
                auth,
            };
            mc.register(id, creds)?;
        }
        Ok(mc)
    }
}

/// Tripwire #107 — closed set of project UUIDs the gateway is allowed
/// to mutate. Any deploy / redeploy / delete / cleanup against a
/// non-listed project is rejected at parse-time. Adding a new project
/// requires a code change (R5: never silent).
pub const ALLOWED_PROJECT_IDS: &[&str] = &[
    "da1fb0c7-199f-42b0-9f08-a84d122feb5b", // primary control plane
    "49a92e6d-1722-4f0b-8361-64b5b8577e37", // secondary control plane
    "e4fe33bb-3b09-4842-9782-7d2dea1abc9b", // Acc1 IGLA (current race)
    "265301ce-0bf2-4187-a36f-348b0eb9942f", // Acc0 trios-trainer
    "39d833c1-4cb6-4af9-b61b-c204b6733a98", // Acc2 thriving-eagerness
];

#[derive(Debug, thiserror::Error)]
#[error("INV-12 #107: project {project:?} not in ALLOWED_PROJECT_IDS")]
pub struct ProjectWhitelistError {
    pub project: String,
}

/// Tripwire #107 enforcement helper.
pub fn assert_project_allowed(project: &str) -> Result<(), ProjectWhitelistError> {
    if ALLOWED_PROJECT_IDS.contains(&project) {
        Ok(())
    } else {
        Err(ProjectWhitelistError {
            project: project.to_string(),
        })
    }
}

fn parse_auth_mode(token: &str, kind_hint: &str) -> AuthMode {
    let normalized = kind_hint.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "team" | "personal" | "bearer" => AuthMode::Team,
        "project" => AuthMode::Project,
        _ if is_uuid_like(token) => AuthMode::Project,
        _ => AuthMode::Team,
    }
}

fn is_uuid_like(s: &str) -> bool {
    let t = s.trim();
    t.len() == 36 && t.chars().filter(|c| *c == '-').count() == 4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_creds(token: &str, project: &str, env: &str, auth: AuthMode) -> AccountCreds {
        AccountCreds {
            token: SecretString::from(token.to_string()),
            project: ProjectId::from(project.to_string()),
            env: EnvironmentId::from(env.to_string()),
            auth,
        }
    }

    #[test]
    fn account_id_string_round_trip() {
        for id in AccountId::all() {
            assert_eq!(AccountId::from_alias(id.as_str()), Some(id));
        }
        assert_eq!(AccountId::from_alias("ACC1"), Some(AccountId::Acc1));
        assert_eq!(AccountId::from_alias("acc-2"), Some(AccountId::Acc2));
        assert_eq!(AccountId::from_alias("3"), Some(AccountId::Acc3));
        assert_eq!(AccountId::from_alias("acc4"), None);
    }

    #[test]
    fn register_then_get_returns_client() {
        let mut mc = RailwayMultiClient::new();
        let creds = fake_creds("personal-token-xyz", "p1", "e1", AuthMode::Team);
        mc.register(AccountId::Acc0, creds).unwrap();
        let _client = mc.get(AccountId::Acc0).unwrap();
        let cd = mc.creds(AccountId::Acc0).unwrap();
        assert_eq!(cd.project.as_str(), "p1");
        assert_eq!(cd.env.as_str(), "e1");
    }

    #[test]
    fn get_unknown_account_returns_not_authorized() {
        let mc = RailwayMultiClient::new();
        let err = mc.get(AccountId::Acc1).unwrap_err();
        assert!(matches!(
            err,
            RailwayError::NotAuthorized {
                account: AccountId::Acc1
            }
        ));
    }

    #[test]
    fn debug_format_does_not_leak_token() {
        let creds = fake_creds("super-secret-token-abc-123", "p", "e", AuthMode::Team);
        let dbg = format!("{:?}", creds);
        assert!(
            !dbg.contains("super-secret-token"),
            "Debug must not leak token, got: {dbg}"
        );
        // But length must be observable for honest auditing.
        assert!(dbg.contains("len="));
    }

    #[test]
    fn registered_lists_accounts_in_order() {
        let mut mc = RailwayMultiClient::new();
        // Register in mixed order — registered() must come back sorted.
        mc.register(AccountId::Acc2, fake_creds("t2", "p2", "e2", AuthMode::Team))
            .unwrap();
        mc.register(AccountId::Acc0, fake_creds("t0", "p0", "e0", AuthMode::Team))
            .unwrap();
        mc.register(AccountId::Acc1, fake_creds("t1", "p1", "e1", AuthMode::Team))
            .unwrap();
        assert_eq!(
            mc.registered(),
            vec![AccountId::Acc0, AccountId::Acc1, AccountId::Acc2]
        );
    }

    #[test]
    fn scope_all_iterates_in_acc0_acc1_acc2_acc3_order() {
        let s = Scope::All;
        let v = s.iter();
        assert_eq!(
            v,
            vec![
                AccountId::Acc0,
                AccountId::Acc1,
                AccountId::Acc2,
                AccountId::Acc3
            ]
        );
    }

    #[test]
    fn scope_one_yields_single_account() {
        let s = Scope::One(AccountId::Acc2);
        assert_eq!(s.iter(), vec![AccountId::Acc2]);
    }

    #[test]
    fn parse_auth_mode_recognises_explicit_kind() {
        assert_eq!(parse_auth_mode("anything", "team"), AuthMode::Team);
        assert_eq!(parse_auth_mode("anything", "project"), AuthMode::Project);
        assert_eq!(parse_auth_mode("anything", "personal"), AuthMode::Team);
        assert_eq!(parse_auth_mode("anything", "bearer"), AuthMode::Team);
    }

    #[test]
    fn parse_auth_mode_falls_back_to_uuid_heuristic() {
        let uuid = "082755d9-dd95-450b-b6f8-79ffd47f834a";
        assert_eq!(parse_auth_mode(uuid, ""), AuthMode::Project);
        assert_eq!(parse_auth_mode("personal-pat-not-uuid", ""), AuthMode::Team);
    }

    #[test]
    fn from_env_skips_empty_slots() {
        // Ensure no env vars from other tests leak in. We set Acc0
        // only, leaving Acc1..Acc3 unset.
        let prev: Vec<(String, Option<String>)> = [
            "RAILWAY_TOKEN_ACC0",
            "RAILWAY_TOKEN_ACC1",
            "RAILWAY_TOKEN_ACC2",
            "RAILWAY_TOKEN_ACC3",
        ]
        .iter()
        .map(|k| ((*k).to_string(), std::env::var(*k).ok()))
        .collect();
        for (k, _) in &prev {
            std::env::remove_var(k);
        }
        std::env::set_var("RAILWAY_TOKEN_ACC0", "tok-acc0-personal");
        std::env::set_var("RAILWAY_PROJECT_ID_ACC0", "proj0");
        std::env::set_var("RAILWAY_ENVIRONMENT_ID_ACC0", "env0");
        std::env::set_var("RAILWAY_TOKEN_KIND_ACC0", "team");

        let mc = RailwayMultiClient::from_env().unwrap();
        assert_eq!(mc.registered(), vec![AccountId::Acc0]);
        assert_eq!(mc.creds(AccountId::Acc0).unwrap().project.as_str(), "proj0");

        // Cleanup → restore prior env.
        std::env::remove_var("RAILWAY_TOKEN_ACC0");
        std::env::remove_var("RAILWAY_PROJECT_ID_ACC0");
        std::env::remove_var("RAILWAY_ENVIRONMENT_ID_ACC0");
        std::env::remove_var("RAILWAY_TOKEN_KIND_ACC0");
        for (k, v) in prev {
            if let Some(val) = v {
                std::env::set_var(&k, val);
            }
        }
    }

    #[test]
    fn from_env_picks_up_four_accounts_when_all_set() {
        for (i, suffix) in ["ACC0", "ACC1", "ACC2", "ACC3"].iter().enumerate() {
            std::env::set_var(format!("RAILWAY_TOKEN_{suffix}"), format!("tok-{i}"));
            std::env::set_var(format!("RAILWAY_PROJECT_ID_{suffix}"), format!("p{i}"));
            std::env::set_var(format!("RAILWAY_ENVIRONMENT_ID_{suffix}"), format!("e{i}"));
            std::env::set_var(format!("RAILWAY_TOKEN_KIND_{suffix}"), "team");
        }
        let mc = RailwayMultiClient::from_env().unwrap();
        assert_eq!(mc.registered(), AccountId::all().to_vec());
        for id in AccountId::all() {
            let cd = mc.creds(id).unwrap();
            assert!(!cd.project.as_str().is_empty());
            assert!(!cd.env.as_str().is_empty());
        }
        for suffix in ["ACC0", "ACC1", "ACC2", "ACC3"] {
            std::env::remove_var(format!("RAILWAY_TOKEN_{suffix}"));
            std::env::remove_var(format!("RAILWAY_PROJECT_ID_{suffix}"));
            std::env::remove_var(format!("RAILWAY_ENVIRONMENT_ID_{suffix}"));
            std::env::remove_var(format!("RAILWAY_TOKEN_KIND_{suffix}"));
        }
    }
}
