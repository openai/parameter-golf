// Tool registry for trios-railway-mcp.
// JSON Schema (draft-07) input descriptors, used by MCP `tools/list`.
//
// Constitutional notes:
//   R1 — TS on stdio, no Python.
//   R5 — exit codes forwarded verbatim, never overclaim DONE.
//   R7 — every emit (where applicable) carries a triplet line.
//   R9 — never expose a tool that bypasses embargo.

export interface ToolDef {
  name: string;
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties?: boolean;
  };
}

export const TOOLS: ToolDef[] = [
  {
    name: "railway_service_list",
    description:
      "List Railway services in the configured IGLA project. Returns service id, name, latest deployment status and image digest.",
    inputSchema: {
      type: "object",
      properties: {
        project_id: {
          type: "string",
          description:
            "Optional override for project id (defaults to RAILWAY_PROJECT_ID).",
        },
      },
      additionalProperties: false,
    },
  },
  {
    name: "railway_service_deploy",
    description:
      "Deploy a new Railway service from an OCI image with environment variables. Writes an L7 experience triplet on success (R7).",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Service name (lower-kebab)." },
        image: {
          type: "string",
          description:
            "OCI image (default: ghcr.io/ghashtag/trios-trainer-igla:latest).",
        },
        env: {
          type: "object",
          description: "Environment variables to set on the service.",
          additionalProperties: { type: "string" },
        },
        existing: {
          type: "boolean",
          description: "If true, update an existing service instead of create.",
          default: false,
        },
        dry_run: { type: "boolean", default: false },
      },
      required: ["name"],
      additionalProperties: false,
    },
  },
  {
    name: "railway_service_redeploy",
    description: "Trigger a redeploy of an existing Railway service by id.",
    inputSchema: {
      type: "object",
      properties: {
        service_id: { type: "string" },
      },
      required: ["service_id"],
      additionalProperties: false,
    },
  },
  {
    name: "railway_service_delete",
    description:
      "Delete a Railway service. Requires explicit confirm=true to guard against accidents.",
    inputSchema: {
      type: "object",
      properties: {
        service_id: { type: "string" },
        confirm: {
          type: "boolean",
          description: "Must be exactly true to proceed.",
        },
      },
      required: ["service_id", "confirm"],
      additionalProperties: false,
    },
  },
  {
    name: "railway_audit_run",
    description:
      "Run the D1..D7 drift audit across all services in the project. Returns drift events as JSON.",
    inputSchema: {
      type: "object",
      properties: {
        project_id: { type: "string" },
      },
      additionalProperties: false,
    },
  },
  {
    name: "neon_hive_status",
    description:
      "Report freshness of Hive tables (igla_race_trials, igla_agents_heartbeat, igla_race_experience) — last write timestamp and row count. Read-only; uses TRIOS_NEON_DSN.",
    inputSchema: {
      type: "object",
      properties: {
        tables: {
          type: "array",
          items: { type: "string" },
          description: "Subset of tables to check; defaults to all three.",
        },
      },
      additionalProperties: false,
    },
  },
  {
    name: "railway_experience_append",
    description:
      "Append an L7 row to .trinity/experience/<YYYYMMDD>.trinity (constitution: every emit carries the R7 triplet).",
    inputSchema: {
      type: "object",
      properties: {
        seed: { type: "integer" },
        bpb: { type: "number" },
        step: { type: "integer" },
        sha: { type: "string", description: "7-char commit SHA." },
        jsonl_row: { type: "integer" },
        gate_status: {
          type: "string",
          enum: ["passed", "failed", "in_progress"],
        },
      },
      required: ["seed", "bpb", "step", "sha", "jsonl_row", "gate_status"],
      additionalProperties: false,
    },
  },
];
