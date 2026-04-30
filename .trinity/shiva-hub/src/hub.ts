/**
 * SHIVA NATRAJA — Central Hub Coordinator
 *
 * The four-armed dancer that orchestrates all MCP arms.
 * φ² + φ⁻² = 3 · TRINITY · SHIVA · DANCE
 */

import { EventEmitter } from "events";
import { McpClient } from "./mcp-client.js";
import { EventRouter } from "./event-router.js";
import type {
  ArmStatus,
  McpArm,
  A2AEvent,
  EventType,
  HealthCheck,
  BpbSample,
  WorkerRegistration,
} from "./types.js";

export class ShivaNatrajaHub extends EventEmitter {
  private arms: Map<McpArm, McpClient> = new Map();
  private router: EventRouter;
  private config: any;
  private healthInterval: NodeJS.Timeout | null = null;
  private bpbSamples: BpbSample[] = [];
  private maxBpbSamples = 100;

  constructor(config: any) {
    super();
    this.config = config;
    this.router = new EventRouter();
    this.setupRouterListeners();
  }

  /**
   * Initialize all four arms
   */
  async initialize(): Promise<void> {
    console.log("[SHIVA] 🕉️ Shiva Natraja awakens... 🕉️");
    console.log("[SHIVA] Connecting four arms for the cosmic dance...");

    const armConfigs: { arm: McpArm; config: any }[] = [
      {
        arm: "neon",
        config: { neonDatabaseUrl: this.config.neonDatabaseUrl },
      },
      {
        arm: "railway",
        config: { railwayToken: this.config.railwayToken },
      },
      {
        arm: "vibee",
        config: { vibeeScript: this.config.vibeeScript },
      },
      {
        arm: "tri-mcp-browser",
        config: {
          host: this.config.triMcpHost || "127.0.0.1",
          port: this.config.triMcpPort || 3026,
          username: this.config.triMcpUsername || "perplexity",
          password: this.config.triMcpPassword || "test123",
        },
      },
    ];

    // Connect all arms in parallel
    const connections = armConfigs.map(({ arm, config }) => {
      const client = new McpClient(arm);
      this.arms.set(arm, client);

      // Forward arm events to hub
      client.on("connected", (armName) => this.emit("arm:connected", armName));
      client.on("disconnected", (info) => this.emit("arm:disconnected", info));
      client.on("heartbeat", (info) => this.emit("arm:heartbeat", info));
      client.on("error", (error) => this.emit("arm:error", error));

      return client.connect(config).catch((err) => {
        console.error(`[SHIVA] Failed to connect ${arm}:`, err.message);
        return null; // Don't fail entire hub if one arm is down
      });
    });

    await Promise.all(connections);

    // Start health monitoring
    this.startHealthCheck();

    // Log final state
    const connectedCount = Array.from(this.arms.values()).filter((a) => a.getStatus().connected).length;
    console.log(`[SHIVA] 🕉️ Dance begins with ${connectedCount}/4 arms connected! 🕉️`);

    // Emit hub ready event
    this.emit("ready", { connectedArms: connectedCount });
  }

  /**
   * Setup router listeners for A2A event handling
   */
  private setupRouterListeners(): void {
    // Listen for events to each arm
    const armTypes: McpArm[] = ["neon", "railway", "vibee", "tri-mcp-browser"];

    for (const arm of armTypes) {
      this.router.on(`to:${arm}`, async (event: A2AEvent) => {
        await this.handleEventForArm(arm, event);
      });
    }

    // Monitor routed events
    this.router.on("routed", (event: A2AEvent) => {
      console.log(`[SHIVA] Event routed: ${event.type} from ${event.from} to ${event.to}`);
    });
  }

  /**
   * Handle an incoming event for a specific arm
   */
  private async handleEventForArm(arm: McpArm, event: A2AEvent): Promise<void> {
    const client = this.arms.get(arm);
    if (!client?.getStatus().connected) {
      console.warn(`[SHIVA] Cannot route to ${arm}: not connected`);
      return;
    }

    try {
      switch (event.type) {
        case "BPB_SAMPLE":
          await this.handleBpbSample(event.payload as BpbSample);
          break;
        case "WORKER_REGISTER":
          await this.handleWorkerRegistration(event.payload as WorkerRegistration);
          break;
        case "TRAINING_COMPLETE":
          await this.handleTrainingComplete(event.payload);
          break;
        default:
          // Generic event handling
          console.log(`[SHIVA] ${arm} received ${event.type}`);
          break;
      }
    } catch (error) {
      console.error(`[SHIVA] Error handling ${event.type} for ${arm}:`, error);
      this.emit("error", { arm, event, error });
    }
  }

  /**
   * Handle BPB sample from training
   */
  private async handleBpbSample(sample: BpbSample): Promise<void> {
    // Store sample for IGLA RACE metrics
    this.bpbSamples.push(sample);

    // Enforce limit
    if (this.bpbSamples.length > this.maxBpbSamples) {
      this.bpbSamples.shift();
    }

    // Calculate current BPB average
    const avgBpb = this.bpbSamples.reduce((sum, s) => sum + s.bpb, 0) / this.bpbSamples.length;

    // Emit for monitoring
    this.emit("bpb:update", {
      sample,
      average: avgBpb,
      count: this.bpbSamples.length,
    });

    // If BPB < 1.50, emit target reached event
    if (avgBpb < 1.50) {
      this.emit("target:reached", { bpb: avgBpb, step: sample.step });
      console.log(`[SHIVA] 🎯 IGLA RACE TARGET REACHED! BPB: ${avgBpb.toFixed(3)} < 1.50 🎯`);
    }
  }

  /**
   * Handle worker registration
   */
  private async handleWorkerRegistration(reg: WorkerRegistration): Promise<void> {
    console.log(`[SHIVA] Worker registered: ${reg.workerId} (${reg.trainerKind})`);
    this.emit("worker:registered", reg);
  }

  /**
   * Handle training complete
   */
  private async handleTrainingComplete(payload: any): Promise<void> {
    console.log(`[SHIVA] Training complete: ${JSON.stringify(payload)}`);
    this.emit("training:complete", payload);

    // Route to VIBEE for code generation if needed
    if (payload.generateCode) {
      await this.router.route(this.router.createEvent(
        "neon",
        "vibee",
        "CODE_GENERATED",
        { experimentId: payload.experimentId }
      ));
    }
  }

  /**
   * Start periodic health checks
   */
  private startHealthCheck(): void {
    this.healthInterval = setInterval(async () => {
      const health = this.getHealth();
      this.emit("health:check", health);
    }, 10000); // Every 10 seconds
  }

  /**
   * Get current health status
   */
  getHealth(): HealthCheck {
    const arms: Record<McpArm, boolean> = {} as any;
    let totalLatency = 0;
    let connectedCount = 0;

    for (const [arm, client] of this.arms.entries()) {
      const status = client.getStatus();
      arms[arm] = status.connected;
      if (status.connected) {
        totalLatency += status.latency;
        connectedCount++;
      }
    }

    return {
      healthy: connectedCount >= 3, // At least 3 arms must be connected
      arms,
      totalLatency: connectedCount > 0 ? totalLatency / connectedCount : 0,
      eventQueueSize: this.router.getQueueSize(),
    };
  }

  /**
   * Get BPB statistics for IGLA RACE
   */
  getBpbStats(): {
    current: number;
    average: number;
    min: number;
    max: number;
    samples: number;
  } {
    if (this.bpbSamples.length === 0) {
      return { current: 0, average: 0, min: 0, max: 0, samples: 0 };
    }

    const bpbs = this.bpbSamples.map((s) => s.bpb);
    return {
      current: bpbs[bpbs.length - 1],
      average: bpbs.reduce((a, b) => a + b, 0) / bpbs.length,
      min: Math.min(...bpbs),
      max: Math.max(...bpbs),
      samples: this.bpbSamples.length,
    };
  }

  /**
   * Manually route an event
   */
  async routeEvent(from: McpArm, to: McpArm, type: EventType, payload: any): Promise<void> {
    const event = this.router.createEvent(from, to, type, payload);
    await this.router.route(event);
  }

  /**
   * Call a tool on a specific arm
   */
  async callTool(arm: McpArm, toolName: string, args: any = {}): Promise<any> {
    const client = this.arms.get(arm);
    if (!client) {
      throw new Error(`Arm ${arm} not initialized`);
    }

    return await client.callTool(toolName, args);
  }

  /**
   * Shutdown the hub gracefully
   */
  async shutdown(): Promise<void> {
    console.log("[SHIVA] 🕉️ Shiva Natraja rests... 🕉️");

    if (this.healthInterval) {
      clearInterval(this.healthInterval);
    }

    for (const [arm, client] of this.arms.entries()) {
      console.log(`[SHIVA] Disconnecting ${arm}...`);
      await client.disconnect();
    }

    console.log("[SHIVA] All arms disconnected. Dance complete.");
    this.emit("shutdown");
  }
}
