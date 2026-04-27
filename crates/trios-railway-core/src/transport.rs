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
        Self::with_token(token)
    }

    pub fn with_token(token: impl Into<String>) -> Result<Self, ClientError> {
        let http = reqwest::Client::builder()
            .user_agent(concat!("trios-railway/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self {
            http,
            endpoint: ENDPOINT.to_string(),
            token: token.into(),
        })
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
        let resp = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.token)
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
    }
}
