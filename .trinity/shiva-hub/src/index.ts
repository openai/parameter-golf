/**
 * SHIVA NATRAJA — Main Entry Point
 *
 * φ² + φ⁻² = 3 · TRINITY · SHIVA · DANCE
 *
 * IGLA RACE MISSION
 * Deadline: 2026-04-30
 * Target: BPB < 1.50
 */

import { ShivaNatrajaHub } from "./hub.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// Configuration from environment
const CONFIG = {
  neonDatabaseUrl: process.env.NEON_DATABASE_URL || process.env.POSTGRES_URL,
  railwayToken: process.env.RAILWAY_TOKEN,
  vibeeScript: process.env.VIBEE_SCRIPT || "/Users/playra/vibee/gleam/run_mcp.sh",
  triMcpHost: process.env.TRI_MCP_HOST || "127.0.0.1",
  triMcpPort: parseInt(process.env.TRI_MCP_PORT || "3026"),
  triMcpUsername: process.env.TRI_MCP_USERNAME || "perplexity",
  triMcpPassword: process.env.TRI_MCP_PASSWORD || "test123",
};

// Global hub instance
let hub: ShivaNatrajaHub | null = null;

/**
 * Create and start the MCP server
 */
async function main() {
  console.log("[SHIVA] 🕉️ SHIVA NATRAJA — Four-Armed A2A Hub 🕉️");
  console.log("[SHIVA] φ² + φ⁻² = 3 · TRINITY · DANCE");
  console.log("[SHIVA] IGLA RACE MISSION — Target: BPB < 1.50");

  // Initialize hub
  hub = new ShivaNatrajaHub(CONFIG);

  // Set up event listeners
  hub.on("ready", (status) => {
    console.log(`[SHIVA] Hub ready with ${status.connectedArms}/4 arms`);
  });

  hub.on("arm:connected", (arm) => {
    console.log(`[SHIVA] Arm connected: ${arm}`);
  });

  hub.on("arm:disconnected", (info) => {
    console.log(`[SHIVA] Arm disconnected:`, info);
  });

  hub.on("arm:error", (error) => {
    console.error(`[SHIVA] Arm error:`, error);
  });

  hub.on("bpb:update", (stats) => {
    console.log(`[SHIVA] BPB update: ${stats.average.toFixed(4)} (sample: ${stats.sample.bpb.toFixed(4)})`);
  });

  hub.on("target:reached", (data) => {
    console.log(`[SHIVA] 🎯🎯🎯 IGLA RACE TARGET REACHED! 🎯🎯🎯`);
    console.log(`[SHIVA] BPB: ${data.bpb.toFixed(3)} at step ${data.step}`);
  });

  hub.on("health:check", (health) => {
    if (!health.healthy) {
      console.warn(`[SHIVA] Health check: ${health.healthy ? "OK" : "UNHEALTHY"}`);
    }
  });

  // Connect all arms
  await hub.initialize();

  // Create MCP server that exposes hub tools
  const server = new McpServer(
    {
      name: "shiva-natraja-a2a-hub",
      version: "1.0.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: [
        {
          name: "shiva_health",
          description: "Get health status of all four Shiva arms (MCP servers)",
          inputSchema: {
            type: "object",
            properties: {},
          },
        },
        {
          name: "shiva_bpb_stats",
          description: "Get BPB statistics for IGLA RACE mission",
          inputSchema: {
            type: "object",
            properties: {},
          },
        },
        {
          name: "shiva_route_event",
          description: "Manually route an A2A event between arms",
          inputSchema: {
            type: "object",
            properties: {
              from: {
                type: "string",
                enum: ["neon", "railway", "vibee", "tri-mcp-browser"],
                description: "Source arm",
              },
              to: {
                type: "string",
                enum: ["neon", "railway", "vibee", "tri-mcp-browser"],
                description: "Destination arm",
              },
              type: {
                type: "string",
                description: "Event type",
              },
              payload: {
                type: "object",
                description: "Event payload (JSON object)",
              },
            },
            required: ["from", "to", "type"],
          },
        },
        {
          name: "shiva_call_tool",
          description: "Call a tool on a specific Shiva arm",
          inputSchema: {
            type: "object",
            properties: {
              arm: {
                type: "string",
                enum: ["neon", "railway", "vibee", "tri-mcp-browser"],
                description: "Target arm",
              },
              tool: {
                type: "string",
                description: "Tool name to call",
              },
              args: {
                type: "object",
                description: "Tool arguments",
              },
            },
            required: ["arm", "tool"],
          },
        },
        {
          name: "shiva_arm_status",
          description: "Get detailed status of a specific arm",
          inputSchema: {
            type: "object",
            properties: {
              arm: {
                type: "string",
                enum: ["neon", "railway", "vibee", "tri-mcp-browser"],
                description: "Arm to query",
              },
            },
            required: ["arm"],
          },
        },
      ],
    };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      switch (name) {
        case "shiva_health": {
          const health = hub!.getHealth();
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(health, null, 2),
              },
            ],
          };
        }

        case "shiva_bpb_stats": {
          const stats = hub!.getBpbStats();
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(stats, null, 2),
              },
            ],
          };
        }

        case "shiva_route_event": {
          await hub!.routeEvent(
            args.from,
            args.to,
            args.type,
            args.payload || {}
          );
          return {
            content: [
              {
                type: "text",
                text: `Event routed from ${args.from} to ${args.to}: ${args.type}`,
              },
            ],
          };
        }

        case "shiva_call_tool": {
          const result = await hub!.callTool(args.arm, args.tool, args.args || {});
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify(result, null, 2),
              },
            ],
          };
        }

        case "shiva_arm_status": {
          const status = hub!.getHealth().arms[args.arm];
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ arm: args.arm, connected: status }, null, 2),
              },
            ],
          };
        }

        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error: ${(error as Error).message}`,
          },
        ],
        isError: true,
      };
    }
  });

  // Start stdio transport
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.log("[SHIVA] MCP server running on stdio");
  console.log("[SHIVA] 🕉️ The cosmic dance begins... 🕉️");

  // Graceful shutdown
  process.on("SIGINT", async () => {
    console.log("[SHIVA] Received SIGINT, shutting down...");
    await hub?.shutdown();
    process.exit(0);
  });
}

// Start the dance
main().catch((error) => {
  console.error("[SHIVA] Fatal error:", error);
  process.exit(1);
});
