# trios-railway-core

Identity types + Railway GraphQL transport.

- `ids` — typed `ProjectId`, `EnvironmentId`, `ServiceId`, `DeployId`.
- `hash` — R7 `RailwayHash::seal(...)` for audit triplets.
- `transport` — `Client` over `https://backboard.railway.com/graphql/v2`.
