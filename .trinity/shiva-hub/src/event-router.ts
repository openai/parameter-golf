/**
 * SHIVA NATRAJA — Event Router
 *
 * The cosmic dance of events between arms.
 * Routes A2A events based on the routing matrix.
 */

import { EventEmitter } from "events";
import type { A2AEvent, EventType, McpArm, RoutingMatrix } from "./types.js";

export class EventRouter extends EventEmitter {
  private queue: A2AEvent[] = [];
  private routingMatrix: RoutingMatrix;
  private maxQueueSize = 1000;

  constructor() {
    super();
    this.routingMatrix = this.buildDefaultRoutingMatrix();
  }

  /**
   * Build default routing matrix based on IGLA RACE requirements
   */
  private buildDefaultRoutingMatrix(): RoutingMatrix {
    return {
      // Training events flow: NEON (BPB samples) → RAILWAY (fleet actions)
      [EventType.BPB_SAMPLE]: ["neon", "railway"],
      [EventType.TRAINING_START]: ["neon", "railway"],
      [EventType.TRAINING_COMPLETE]: ["neon", "railway", "vibee"], // Trigger code gen

      // Fleet events flow: RAILWAY → NEON (registration)
      [EventType.WORKER_REGISTER]: ["railway", "neon"],
      [EventType.WORKER_HEARTBEAT]: ["railway"],
      [EventType.WORKER_ERROR]: ["railway", "vibee"], // Error handling code

      // Code events flow: VIBEE → TRI-MCP-BROWSER (validation)
      [EventType.CODE_GENERATED]: ["vibee", "tri-mcp-browser"],
      [EventType.CODE_VALIDATED]: ["tri-mcp-browser", "vibee"],
      [EventType.COMPILE_SUCCESS]: ["vibee"],
      [EventType.COMPILE_FAILURE]: ["vibee"], // Retry logic

      // Browser events flow: TRI-MCP-BROWSER → RAILWAY (audit results)
      [EventType.PAGE_AUDIT_COMPLETE]: ["tri-mcp-browser", "railway"],
      [EventType.SCREENSHOT_CAPTURED]: ["tri-mcp-browser"],

      // Hub events: broadcast to all
      [EventType.HEALTH_CHECK]: ["neon", "railway", "vibee", "tri-mcp-browser"],
      [EventType.ROUTE_ERROR]: ["neon", "railway", "vibee", "tri-mcp-browser"],
    };
  }

  /**
   * Route an event to its destination arms
   */
  async route(event: A2AEvent): Promise<void> {
    // Add to queue for persistence
    this.queue.push(event);

    // Enforce queue limit
    if (this.queue.length > this.maxQueueSize) {
      this.queue.shift();
    }

    // Get destination arms
    const destinations = this.routingMatrix[event.type] || [];

    console.log(`[SHIVA ROUTE] ${event.from} → ${destinations.join(", ")} | ${event.type}`);

    // Emit to each destination
    for (const dest of destinations) {
      this.emit(`to:${dest}`, event);
    }

    // Emit globally for monitoring
    this.emit("routed", event);
  }

  /**
   * Create a new event
   */
  createEvent(
    from: McpArm,
    to: McpArm,
    type: EventType,
    payload: unknown,
    correlationId?: string
  ): A2AEvent {
    return {
      id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      from,
      to,
      type,
      timestamp: Date.now(),
      payload,
      correlationId,
    };
  }

  /**
   * Get current queue size
   */
  getQueueSize(): number {
    return this.queue.length;
  }

  /**
   * Get recent events for an arm
   */
  getRecentEvents(arm: McpArm, limit = 10): A2AEvent[] {
    return this.queue
      .filter((e) => e.from === arm || e.to === arm)
      .slice(-limit);
  }

  /**
   * Update routing matrix dynamically
   */
  updateRoutingMatrix(type: EventType, destinations: McpArm[]): void {
    this.routingMatrix[type] = destinations;
    console.log(`[SHIVA ROUTE] Updated ${type} → ${destinations.join(", ")}`);
  }
}
