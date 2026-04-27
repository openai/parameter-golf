//! RW-03: typed mutations against the Railway GraphQL API.
//!
//! Mutations supported:
//! - `serviceCreate` (image-based, project-scoped)
//! - `variableUpsert` (one variable at a time, scoped to project + env + service)
//! - `serviceInstanceDeployV2` (trigger a redeploy on the latest source)
//! - `serviceInstanceUpdate` (set source.image so the next deploy pulls a new image)
//! - `serviceDelete`
//!
//! All mutations log a `R7` triplet via `RailwayHash::seal` at the call site
//! (callers must do that — this module returns plain ids).

use serde::Deserialize;
use serde_json::json;

use crate::ids::{DeployId, EnvironmentId, ProjectId, ServiceId};
use crate::transport::{Client, ClientError};

pub const M_SERVICE_CREATE: &str = "mutation M($input: ServiceCreateInput!) {
  serviceCreate(input: $input) { id name projectId }
}";

pub const M_VARIABLE_UPSERT: &str = "mutation M($input: VariableUpsertInput!) {
  variableUpsert(input: $input)
}";

pub const M_DEPLOY_REDEPLOY: &str = "mutation M($serviceId: String!, $environmentId: String!) {
  serviceInstanceRedeploy(serviceId: $serviceId, environmentId: $environmentId)
}";

pub const M_SERVICE_INSTANCE_UPDATE: &str =
    "mutation M($serviceId: String!, $environmentId: String!, $input: ServiceInstanceUpdateInput!) {
  serviceInstanceUpdate(serviceId: $serviceId, environmentId: $environmentId, input: $input)
}";

pub const M_SERVICE_DELETE: &str = "mutation M($id: String!) {
  serviceDelete(id: $id)
}";

#[derive(Debug, Clone, Deserialize)]
pub struct CreatedService {
    pub id: String,
    pub name: String,
    #[serde(rename = "projectId")]
    pub project_id: String,
}

/// Create a new service in a project. The image is set on the service
/// instance via a follow-up `serviceInstanceUpdate` call (Railway splits
/// service vs service-instance config).
pub async fn service_create(
    client: &Client,
    project: &ProjectId,
    name: &str,
) -> Result<CreatedService, ClientError> {
    #[derive(Deserialize)]
    struct R {
        #[serde(rename = "serviceCreate")]
        service_create: CreatedService,
    }
    let vars = json!({
        "input": {
            "projectId": project.as_str(),
            "name": name,
        }
    });
    let r: R = client.query(M_SERVICE_CREATE, Some(vars)).await?;
    Ok(r.service_create)
}

/// Pin the image source on a service instance. The next redeploy will use it.
pub async fn service_instance_set_image(
    client: &Client,
    service: &ServiceId,
    env: &EnvironmentId,
    image: &str,
) -> Result<(), ClientError> {
    let vars = json!({
        "serviceId": service.as_str(),
        "environmentId": env.as_str(),
        "input": {
            "source": { "image": image }
        }
    });
    let _: serde_json::Value = client.query(M_SERVICE_INSTANCE_UPDATE, Some(vars)).await?;
    Ok(())
}

/// Upsert a single environment variable for a service.
pub async fn variable_upsert(
    client: &Client,
    project: &ProjectId,
    env: &EnvironmentId,
    service: &ServiceId,
    name: &str,
    value: &str,
) -> Result<(), ClientError> {
    let vars = json!({
        "input": {
            "projectId": project.as_str(),
            "environmentId": env.as_str(),
            "serviceId": service.as_str(),
            "name": name,
            "value": value,
        }
    });
    let _: serde_json::Value = client.query(M_VARIABLE_UPSERT, Some(vars)).await?;
    Ok(())
}

/// Redeploy a service in an environment using the most recent source.
/// Returns the new deployment id.
pub async fn service_redeploy(
    client: &Client,
    service: &ServiceId,
    env: &EnvironmentId,
) -> Result<DeployId, ClientError> {
    #[derive(Deserialize)]
    struct R {
        #[serde(rename = "serviceInstanceRedeploy")]
        service_instance_redeploy: serde_json::Value,
    }
    let vars = json!({
        "serviceId": service.as_str(),
        "environmentId": env.as_str(),
    });
    let r: R = client.query(M_DEPLOY_REDEPLOY, Some(vars)).await?;
    let id = match r.service_instance_redeploy {
        serde_json::Value::String(s) => s,
        v => v.to_string(),
    };
    Ok(DeployId::new(id))
}

/// Permanently delete a service (and all its deployments).
pub async fn service_delete(client: &Client, service: &ServiceId) -> Result<(), ClientError> {
    let vars = json!({ "id": service.as_str() });
    let _: serde_json::Value = client.query(M_SERVICE_DELETE, Some(vars)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_strings_present() {
        for m in [
            M_SERVICE_CREATE,
            M_VARIABLE_UPSERT,
            M_DEPLOY_REDEPLOY,
            M_SERVICE_INSTANCE_UPDATE,
            M_SERVICE_DELETE,
        ] {
            assert!(m.contains("mutation M("));
        }
    }

    #[test]
    fn created_service_parses() {
        let raw = serde_json::json!({"id":"s1","name":"trios-train-seed-43","projectId":"p"});
        let cs: CreatedService = serde_json::from_value(raw).unwrap();
        assert_eq!(cs.name, "trios-train-seed-43");
    }
}
