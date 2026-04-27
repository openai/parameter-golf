//! RW-02: typed read queries against the Railway GraphQL API.
//!
//! Each public function takes a `Client` and the strongly-typed
//! identifiers from `crate::ids` and returns a serde-defined view.
//!
//! Networked behaviour is exercised by `bin/tri-railway` integration
//! commands and by gating CI; unit tests here only cover deserialization
//! shapes and query string stability.

use serde::Deserialize;
use serde_json::json;

use crate::ids::{DeployId, EnvironmentId, ProjectId, ServiceId};
use crate::transport::{Client, ClientError};

/// A single Railway service as exposed by `project.services.edges`.
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceSummary {
    pub id: String,
    pub name: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProjectView {
    pub id: String,
    pub name: String,
    #[serde(rename = "services")]
    services: ServiceConnection,
}

#[derive(Debug, Clone, Deserialize)]
struct ServiceConnection {
    edges: Vec<ServiceEdge>,
}

#[derive(Debug, Clone, Deserialize)]
struct ServiceEdge {
    node: ServiceSummary,
}

impl ProjectView {
    #[must_use]
    pub fn services(&self) -> Vec<ServiceSummary> {
        self.services.edges.iter().map(|e| e.node.clone()).collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeployView {
    pub id: String,
    pub status: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    pub service: DeployService,
    pub meta: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeployService {
    pub name: String,
}

/// Stable GraphQL query string for `project(id)`.
pub const QUERY_PROJECT: &str = "query Q($id: String!) {
  project(id: $id) {
    id
    name
    services {
      edges { node { id name createdAt } }
    }
  }
}";

/// Stable GraphQL query string for `deployments`.
pub const QUERY_DEPLOYMENTS: &str = "query Q($pid: String!, $eid: String!, $sid: String) {
  deployments(first: 5, input: {projectId: $pid, environmentId: $eid, serviceId: $sid}) {
    edges {
      node {
        id status createdAt
        service { name }
        meta
      }
    }
  }
}";

/// Stable GraphQL query string for `variables`.
pub const QUERY_VARIABLES: &str = "query Q($pid: String!, $eid: String!, $sid: String!) {
  variables(projectId: $pid, environmentId: $eid, serviceId: $sid)
}";

/// Fetch a project by id, including its services list.
pub async fn project_view(
    client: &Client,
    project: &ProjectId,
) -> Result<ProjectView, ClientError> {
    #[derive(Debug, Deserialize)]
    struct R {
        project: ProjectView,
    }
    let vars = json!({ "id": project.as_str() });
    let r: R = client.query(QUERY_PROJECT, Some(vars)).await?;
    Ok(r.project)
}

/// Latest deployments in a service (or whole project if `service` is None).
pub async fn recent_deployments(
    client: &Client,
    project: &ProjectId,
    env: &EnvironmentId,
    service: Option<&ServiceId>,
) -> Result<Vec<DeployView>, ClientError> {
    #[derive(Debug, Deserialize)]
    struct DConn {
        edges: Vec<DEdge>,
    }
    #[derive(Debug, Deserialize)]
    struct DEdge {
        node: DeployView,
    }
    #[derive(Debug, Deserialize)]
    struct R {
        deployments: DConn,
    }
    let vars = json!({
        "pid": project.as_str(),
        "eid": env.as_str(),
        "sid": service.map(super::ids::ServiceId::as_str),
    });
    let r: R = client.query(QUERY_DEPLOYMENTS, Some(vars)).await?;
    Ok(r.deployments.edges.into_iter().map(|e| e.node).collect())
}

/// Variables for a service (returns the raw JSON object so callers can
/// pick keys without coupling to a fixed schema).
pub async fn service_variables(
    client: &Client,
    project: &ProjectId,
    env: &EnvironmentId,
    service: &ServiceId,
) -> Result<serde_json::Value, ClientError> {
    #[derive(Debug, Deserialize)]
    struct R {
        variables: serde_json::Value,
    }
    let vars = json!({
        "pid": project.as_str(),
        "eid": env.as_str(),
        "sid": service.as_str(),
    });
    let r: R = client.query(QUERY_VARIABLES, Some(vars)).await?;
    Ok(r.variables)
}

/// Convenience: id-only of the most recent deployment for a service.
pub async fn latest_deploy_id(
    client: &Client,
    project: &ProjectId,
    env: &EnvironmentId,
    service: &ServiceId,
) -> Result<Option<DeployId>, ClientError> {
    let deploys = recent_deployments(client, project, env, Some(service)).await?;
    Ok(deploys.into_iter().next().map(|d| DeployId::new(d.id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_view_extracts_services() {
        let raw = serde_json::json!({
            "id": "p", "name": "IGLA",
            "services": { "edges": [
                { "node": { "id": "s1", "name": "trios-train-seed-43", "createdAt": "2026-04-26T14:03:56.760Z" } }
            ]}
        });
        let pv: ProjectView = serde_json::from_value(raw).unwrap();
        let s = pv.services();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].name, "trios-train-seed-43");
    }

    #[test]
    fn query_strings_are_stable() {
        assert!(QUERY_PROJECT.contains("services"));
        assert!(QUERY_DEPLOYMENTS.contains("imageDigest") || QUERY_DEPLOYMENTS.contains("meta"));
        assert!(QUERY_VARIABLES.contains("variables"));
    }

    #[test]
    fn deploy_view_parses_meta_image() {
        let raw = serde_json::json!({
            "id": "d1",
            "status": "SUCCESS",
            "createdAt": "2026-04-27T06:13:48.938Z",
            "service": { "name": "trios-train-seed-102" },
            "meta": { "image": "ghcr.io/ghashtag/trios-trainer-igla:latest", "imageDigest": "sha256:abc" }
        });
        let d: DeployView = serde_json::from_value(raw).unwrap();
        assert_eq!(d.status, "SUCCESS");
        assert_eq!(d.meta["imageDigest"], "sha256:abc");
    }
}
