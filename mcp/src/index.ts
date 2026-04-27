#!/usr/bin/env node
// trios-railway-mcp — Model Context Protocol stdio server wrapping tri-railway.
//
// Anchor: phi^2 + phi^-2 = 3.
// Issue: https://github.com/gHashTag/trios-railway/issues/18
//
// Constitutional notes:
//   R1 — TS on stdio.
//   R5 — honest exit codes; isError=true on non-zero CLI.
//   R7 — tri-railway emits R7 triplet to .trinity/experience; we forward stdout.
//   R9 — never expose a tool that bypasses ledger::is_embargoed.

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { loadConfig } from "./config.js";
import { TOOLS } from "./tools/registry.js";
import {
  handleAuditRun,
  handleExperienceAppend,
  handleNeonHiveStatus,
  handleServiceDelete,
  handleServiceDeploy,
  handleServiceList,
  handleServiceRedeploy,
} from "./tools/handlers.js";

async function main(): Promise<void> {
  const cfg = loadConfig();

  const server = new Server(
    { name: "trios-railway-mcp", version: "0.0.1" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS,
  }));

  server.setRequestHandler(CallToolRequestSchema, async (req): Promise<any> => {
    const { name, arguments: rawArgs } = req.params;
    const args = (rawArgs ?? {}) as Record<string, unknown>;
    try {
      switch (name) {
        case "railway_service_list":
          return await handleServiceList(cfg, args as { project_id?: string });
        case "railway_service_deploy":
          return await handleServiceDeploy(
            cfg,
            args as Parameters<typeof handleServiceDeploy>[1],
          );
        case "railway_service_redeploy":
          return await handleServiceRedeploy(
            cfg,
            args as { service_id: string },
          );
        case "railway_service_delete":
          return await handleServiceDelete(
            cfg,
            args as { service_id: string; confirm: boolean },
          );
        case "railway_audit_run":
          return await handleAuditRun(cfg, args as { project_id?: string });
        case "neon_hive_status":
          return await handleNeonHiveStatus(
            cfg,
            args as { tables?: string[] },
          );
        case "railway_experience_append":
          return await handleExperienceAppend(
            cfg,
            args as Parameters<typeof handleExperienceAppend>[1],
          );
        default:
          return {
            content: [{ type: "text", text: `Unknown tool: ${name}` }],
            isError: true,
          };
      }
    } catch (err) {
      const msg = err instanceof Error ? err.stack ?? err.message : String(err);
      return {
        content: [{ type: "text", text: `Tool error:\n${msg}` }],
        isError: true,
      };
    }
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
  // eslint-disable-next-line no-console -- stderr is allowed in MCP stdio
  console.error(
    `[trios-railway-mcp] up. bin=${cfg.triRailwayBin} project=${cfg.railwayProjectId}`,
  );
}

main().catch((e) => {
  // eslint-disable-next-line no-console
  console.error("[trios-railway-mcp] fatal:", e);
  process.exit(1);
});
