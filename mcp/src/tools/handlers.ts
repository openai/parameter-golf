// Handlers for each MCP tool. Each handler shells out to `tri-railway` (R5
// honest exit codes) or queries Neon directly (read-only) and returns text
// content for the MCP client.

import type { Config } from "../config.js";
import { runCli, renderResult } from "../runner.js";

export interface ToolResult {
  content: { type: "text"; text: string }[];
  isError?: boolean;
}

function asResult(text: string, isError = false): ToolResult {
  return { content: [{ type: "text", text }], isError };
}

function envFlags(env?: Record<string, string>): string[] {
  if (!env) return [];
  return Object.entries(env).flatMap(([k, v]) => ["--var", `${k}=${v}`]);
}

export async function handleServiceList(
  cfg: Config,
  args: { project_id?: string },
): Promise<ToolResult> {
  const projectId = args.project_id ?? cfg.railwayProjectId;
  const r = await runCli(cfg, ["service", "list", "--project", projectId]);
  return asResult(renderResult(r), r.code !== 0);
}

export async function handleServiceDeploy(
  cfg: Config,
  args: {
    name: string;
    image?: string;
    env?: Record<string, string>;
    existing?: boolean;
    dry_run?: boolean;
  },
): Promise<ToolResult> {
  const cliArgs = [
    "service",
    "deploy",
    "--name",
    args.name,
    "--image",
    args.image ?? cfg.defaultImage,
    ...envFlags(args.env),
  ];
  if (args.existing) cliArgs.push("--existing");
  if (args.dry_run) cliArgs.push("--dry-run");
  const r = await runCli(cfg, cliArgs);
  return asResult(renderResult(r), r.code !== 0);
}

export async function handleServiceRedeploy(
  cfg: Config,
  args: { service_id: string },
): Promise<ToolResult> {
  const r = await runCli(cfg, [
    "service",
    "redeploy",
    "--service",
    args.service_id,
  ]);
  return asResult(renderResult(r), r.code !== 0);
}

export async function handleServiceDelete(
  cfg: Config,
  args: { service_id: string; confirm: boolean },
): Promise<ToolResult> {
  if (args.confirm !== true) {
    return asResult(
      "REFUSED: confirm must be exactly true. No service was deleted.",
      true,
    );
  }
  const r = await runCli(cfg, [
    "service",
    "delete",
    "--service",
    args.service_id,
    "--yes",
  ]);
  return asResult(renderResult(r), r.code !== 0);
}

export async function handleAuditRun(
  cfg: Config,
  args: { project_id?: string },
): Promise<ToolResult> {
  const projectId = args.project_id ?? cfg.railwayProjectId;
  const r = await runCli(cfg, ["audit", "run", "--project", projectId]);
  return asResult(renderResult(r), r.code !== 0);
}

export async function handleNeonHiveStatus(
  cfg: Config,
  args: { tables?: string[] },
): Promise<ToolResult> {
  if (!cfg.neonDsn) {
    return asResult(
      "TRIOS_NEON_DSN is not set. Cannot query Neon. Set the env var to enable neon_hive_status.",
      true,
    );
  }
  const tables = args.tables ?? [
    "igla_race_trials",
    "igla_agents_heartbeat",
    "igla_race_experience",
  ];
  // Delegate to tri-railway audit hive-status (subcommand may be added later).
  // Until then, surface a structured stub so MCP clients see an honest stat.
  const r = await runCli(cfg, [
    "audit",
    "hive-status",
    "--tables",
    tables.join(","),
  ]);
  return asResult(renderResult(r), r.code !== 0);
}

export async function handleExperienceAppend(
  cfg: Config,
  args: {
    seed: number;
    bpb: number;
    step: number;
    sha: string;
    jsonl_row: number;
    gate_status: string;
  },
): Promise<ToolResult> {
  const r = await runCli(cfg, [
    "experience",
    "append",
    "--seed",
    String(args.seed),
    "--bpb",
    String(args.bpb),
    "--step",
    String(args.step),
    "--sha",
    args.sha,
    "--jsonl-row",
    String(args.jsonl_row),
    "--gate-status",
    args.gate_status,
  ]);
  return asResult(renderResult(r), r.code !== 0);
}
