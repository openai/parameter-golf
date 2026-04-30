/**
 * SHIVA NATRAJA — MCP Client for Each Arm
 *
 * Each arm (MCP server) is connected via stdio or HTTP.
 * This module manages the connection and tool invocation.
 */

import { spawn, ChildProcess } from "child_process";
import { EventEmitter } from "events";
import type { ArmStatus, McpArm } from "./types.js";

export interface McpTool {
  name: string;
  description: string;
}

export class McpClient extends EventEmitter {
  private arm: McpArm;
  private process: ChildProcess | null = null;
  private tools: Map<string, McpTool> = new Map();
  private connected = false;
  private lastHeartbeat = 0;
  private latency = 0;
  private pendingRequests = new Map<number, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    startTime: number;
  }>();
  private requestId = 0;

  constructor(arm: McpArm) {
    super();
    this.arm = arm;
  }

  /**
   * Connect to MCP arm based on configuration
   */
  async connect(config: any): Promise<void> {
    console.log(`[SHIVA] Connecting to ${this.arm} arm...`);

    try {
      switch (this.arm) {
        case "neon":
          await this.connectNeon(config);
          break;
        case "railway":
          await this.connectRailway(config);
          break;
        case "vibee":
          await this.connectVibee(config);
          break;
        case "tri-mcp-browser":
          await this.connectTriMcpBrowser(config);
          break;
      }

      this.connected = true;
      this.lastHeartbeat = Date.now();
      this.emit("connected", this.arm);
      console.log(`[SHIVA] ${this.arm} arm CONNECTED`);
    } catch (error) {
      console.error(`[SHIVA] Failed to connect ${this.arm}:`, error);
      this.emit("error", { arm: this.arm, error });
      throw error;
    }
  }

  private async connectNeon(config: any): Promise<void> {
    // NEON MCP: Use stdio with npx
    // Note: @neondatabase/mcp-server-neon is deprecated
    // Should use mcp.neon.tech instead
    this.process = spawn("npx", ["-y", "@neondatabase/mcp-server-neon"], {
      env: { ...process.env, NEON_DATABASE_URL: config.neonDatabaseUrl },
    });

    this.setupStdioHandlers();
  }

  private async connectRailway(config: any): Promise<void> {
    // Railway MCP: Use stdio with npx
    this.process = spawn("npx", ["-y", "@railway/mcp-server"], {
      env: { ...process.env, RAILWAY_TOKEN: config.railwayToken },
    });

    this.setupStdioHandlers();
  }

  private async connectVibee(config: any): Promise<void> {
    // VIBEE MCP: Use custom script
    this.process = spawn(config.vibeeScript || "/Users/playra/vibee/gleam/run_mcp.sh", [], {
      env: process.env,
    });

    this.setupStdioHandlers();
  }

  private async connectTriMcpBrowser(config: any): Promise<void> {
    // TRI-MCP-BROWSER: Use HTTP instead of stdio
    // Connect via HTTP/REST API
    const auth = Buffer.from(`${config.username}:${config.password}`).toString("base64");

    // Health check
    const response = await fetch(`http://${config.host}:${config.port}/health`, {
      headers: {
        "Authorization": `Basic ${auth}`,
      },
    });

    if (!response.ok) {
      throw new Error(`TRI-MCP-BROWSER health check failed: ${response.status}`);
    }

    // Get available tools via /tools endpoint
    const toolsResponse = await fetch(`http://${config.host}:${config.port}/tools`, {
      headers: {
        "Authorization": `Basic ${auth}`,
      },
    });

    if (toolsResponse.ok) {
      const tools = await toolsResponse.json();
      tools.forEach((tool: McpTool) => {
        this.tools.set(tool.name, tool);
      });
    }

    // Set up heartbeat
    this.setupHttpHeartbeat(config);
  }

  private setupStdioHandlers(): void {
    if (!this.process) return;

    this.process.stdout?.on("data", (data: Buffer) => {
      try {
        const lines = data.toString().split("\n").filter(Boolean);
        for (const line of lines) {
          const message = JSON.parse(line);
          this.handleMessage(message);
        }
      } catch (error) {
        console.error(`[SHIVA] Failed to parse message from ${this.arm}:`, error);
      }
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      console.error(`[SHIVA] ${this.arm} stderr:`, data.toString());
    });

    this.process.on("exit", (code) => {
      this.connected = false;
      this.emit("disconnected", { arm: this.arm, code });
      console.log(`[SHIVA] ${this.arm} process exited with code ${code}`);
    });

    // Initialize MCP protocol
    this.sendRequest({ jsonrpc: "2.0", id: this.nextRequestId(), method: "initialize", params: {} });
    this.sendRequest({ jsonrpc: "2.0", id: this.nextRequestId(), method: "tools/list", params: {} });
  }

  private setupHttpHeartbeat(config: any): void {
    const auth = Buffer.from(`${config.username}:${config.password}`).toString("base64");
    const interval = setInterval(async () => {
      try {
        const start = Date.now();
        const response = await fetch(`http://${config.host}:${config.port}/health`, {
          headers: { "Authorization": `Basic ${auth}` },
        });
        this.latency = Date.now() - start;
        this.lastHeartbeat = Date.now();

        if (response.ok) {
          this.emit("heartbeat", { arm: this.arm, latency: this.latency });
        } else {
          this.emit("error", { arm: this.arm, error: `Health check failed: ${response.status}` });
        }
      } catch (error) {
        this.emit("error", { arm: this.arm, error });
      }
    }, 5000); // 5 second heartbeat

    this.process = { kill: () => clearInterval(interval) } as any;
  }

  private handleMessage(message: any): void {
    if (message.id !== undefined) {
      const pending = this.pendingRequests.get(message.id);
      if (pending) {
        this.latency = Date.now() - pending.startTime;
        this.pendingRequests.delete(message.id);

        if (message.error) {
          pending.reject(new Error(message.error.message));
        } else {
          pending.resolve(message.result);
        }
      }
    } else if (message.method === "tools/list") {
      const tools = message.result?.tools || [];
      tools.forEach((tool: McpTool) => {
        this.tools.set(tool.name, tool);
      });
      this.emit("tools", { arm: this.arm, tools });
    }
  }

  private sendRequest(message: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const id = message.id || this.nextRequestId();
      message.id = id;

      this.pendingRequests.set(id, {
        resolve,
        reject,
        startTime: Date.now(),
      });

      if (this.process?.stdin) {
        this.process.stdin.write(JSON.stringify(message) + "\n");
      }
    });
  }

  private nextRequestId(): number {
    return ++this.requestId;
  }

  /**
   * Call a tool on this MCP arm
   */
  async callTool(name: string, args: any = {}): Promise<any> {
    if (!this.connected) {
      throw new Error(`${this.arm} not connected`);
    }

    const tool = this.tools.get(name);
    if (!tool) {
      throw new Error(`Tool ${name} not found on ${this.arm}`);
    }

    const start = Date.now();
    try {
      const result = await this.sendRequest({
        jsonrpc: "2.0",
        id: this.nextRequestId(),
        method: "tools/call",
        params: { name, arguments: args },
      });
      this.latency = Date.now() - start;
      return result;
    } catch (error) {
      this.latency = Date.now() - start;
      throw error;
    }
  }

  /**
   * Get current status
   */
  getStatus(): ArmStatus {
    return {
      arm: this.arm,
      connected: this.connected,
      lastHeartbeat: this.lastHeartbeat,
      tools: Array.from(this.tools.keys()),
      latency: this.latency,
    };
  }

  /**
   * Disconnect from MCP arm
   */
  async disconnect(): Promise<void> {
    if (this.process) {
      this.process.kill();
      this.process = null;
    }
    this.connected = false;
    this.emit("disconnected", { arm: this.arm });
  }
}
