//! Minimal Railway GraphQL client.
//!
//! Endpoint: `https://backboard.railway.com/graphql/v2`.
//! Authentication: bearer `RAILWAY_TOKEN`.
//!
//! Bigger query/mutation typed wrappers will land under issues #4 and
//! #5; this module ships only the raw transport so the rest of the
//! workspace can build, test, and ship.

use serde::{Deserialize, Serialize};
use thiserror::Error;

const ENDPOINT: &str = "https://backboard.railway.com/graphql/v2";

/// True if `s` matches the canonical 8-4-4-4-12 hex UUID shape.
fn is_uuid_like(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (i, &b) in bytes.iter().enumerate() {
        let is_dash = matches!(i, 8 | 13 | 18 | 23);
        let ok = if is_dash {
            b == b'-'
        } else {
            b.is_ascii_hexdigit()
        };
        if !ok {
            return false;
        }
    }
    true
}

/// Authorization style for the Railway GraphQL endpoint.
///
/// `team` (default) sends `Authorization: Bearer <token>` and works for
/// account/team tokens. `project` sends `Project-Access-Token: <token>`
/// and is required for project-scoped tokens (which is what the IGLA
/// workspace token currently is).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AuthMode {
    #[default]
    Team,
    Project,
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("RAILWAY_TOKEN not set")]
    MissingToken,
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("graphql errors: {0}")]
    GraphQl(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    endpoint: String,
    token: String,
    auth: AuthMode,
}

#[derive(Debug, Serialize)]
struct Request<'a, V: Serialize> {
    query: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    variables: Option<V>,
}

#[derive(Debug, Deserialize)]
struct Response<T> {
    data: Option<T>,
    #[serde(default)]
    errors: Vec<GraphQlError>,
}

#[derive(Debug, Deserialize)]
struct GraphQlError {
    message: String,
}

impl Client {
    pub fn from_env() -> Result<Self, ClientError> {
        let token = std::env::var("RAILWAY_TOKEN").map_err(|_| ClientError::MissingToken)?;
        // Heuristic: a UUID-shaped token (8-4-4-4-12, length 36) is
        // overwhelmingly a project-access token in Railway. Account/team
        // tokens are longer, opaque, and start with `rwt_` or similar.
        let auth = if is_uuid_like(&token) {
            AuthMode::Project
        } else {
            AuthMode::Team
        };
        Self::with_token_and_mode(token, auth)
    }

    pub fn with_token(token: impl Into<String>) -> Result<Self, ClientError> {
        Self::with_token_and_mode(token, AuthMode::Team)
    }

    pub fn with_token_and_mode(
        token: impl Into<String>,
        auth: AuthMode,
    ) -> Result<Self, ClientError> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("trios-railway/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            endpoint: ENDPOINT.to_string(),
            token: token.into(),
            auth,
        })
    }

    #[must_use]
    pub fn auth_mode(&self) -> AuthMode {
        self.auth
    }

    /// Override the endpoint; used by integration tests.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Token fingerprint suitable for audit lines (never the token).
    #[must_use]
    pub fn token_fingerprint(&self) -> String {
        crate::hash::token_fingerprint(&self.token)
    }

    /// Issue a raw GraphQL request.
    pub async fn query<V, T>(&self, query: &str, variables: Option<V>) -> Result<T, ClientError>
    where
        V: Serialize,
        T: for<'de> Deserialize<'de>,
    {
        let body = Request { query, variables };
        let req = self.http.post(&self.endpoint);
        let req = match self.auth {
            AuthMode::Team => req.bearer_auth(&self.token),
            AuthMode::Project => req.header("Project-Access-Token", &self.token),
        };
        let resp = req
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<Response<T>>()
            .await?;

        if !resp.errors.is_empty() {
            let msg = resp
                .errors
                .into_iter()
                .map(|e| e.message)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(ClientError::GraphQl(msg));
        }
        resp.data
            .ok_or_else(|| ClientError::GraphQl("empty data".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_token_yields_error() {
        // Make sure our error mapping is correct without touching env.
        let err = ClientError::MissingToken;
        assert_eq!(err.to_string(), "RAILWAY_TOKEN not set");
    }

    #[test]
    fn with_token_constructs() {
        let c = Client::with_token("abc").unwrap();
        assert_eq!(c.token_fingerprint().len(), 8);
        assert_eq!(c.auth_mode(), AuthMode::Team);
    }

    #[test]
    fn project_token_detected_as_uuid() {
        assert!(is_uuid_like("447e97bf-8c32-42c9-a585-c7e359f7458f"));
        assert!(!is_uuid_like("rwt_abcd_not_a_uuid"));
        assert!(!is_uuid_like("too-short"));
    }
}
