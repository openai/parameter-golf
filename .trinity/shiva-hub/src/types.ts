/**
 * SHIVA NATRAJA — A2A Hub Types
 *
 * The four arms of Shiva dance together:
 * - NEON: Database BPB sampling (PostgreSQL)
 * - RAILWAY: Fleet deployment & monitoring
 * - VIBEE: Code generation & transformation
 * - TRI-MCP-BROWSER: Web automation & audits
 *
 * φ² + φ⁻² = 3 · TRINITY · SHIVA · DANCE
 */

// The four arms of Shiva
export type McpArm = "neon" | "railway" | "vibee" | "tri-mcp-browser";

// MCP connection status
export interface ArmStatus {
  arm: McpArm;
  connected: boolean;
  lastHeartbeat: number;
  tools: string[];
  latency: number;
  error?: string;
}

// A2A Event for cross-arm communication
export interface A2AEvent {
  id: string;
  from: McpArm;
  to: McpArm;
  type: EventType;
  timestamp: number;
  payload: unknown;
  correlationId?: string;
}

export enum EventType {
  // Training events
  BPB_SAMPLE = "BPB_SAMPLE",
  TRAINING_START = "TRAINING_START",
  TRAINING_COMPLETE = "TRAINING_COMPLETE",

  // Fleet events
  WORKER_REGISTER = "WORKER_REGISTER",
  WORKER_HEARTBEAT = "WORKER_HEARTBEAT",
  WORKER_ERROR = "WORKER_ERROR",

  // Code events
  CODE_GENERATED = "CODE_GENERATED",
  CODE_VALIDATED = "CODE_VALIDATED",
  COMPILE_SUCCESS = "COMPILE_SUCCESS",
  COMPILE_FAILURE = "COMPILE_FAILURE",

  // Browser events
  PAGE_AUDIT_COMPLETE = "PAGE_AUDIT_COMPLETE",
  SCREENSHOT_CAPTURED = "SCREENSHOT_CAPTURED",

  // Hub events
  HEALTH_CHECK = "HEALTH_CHECK",
  ROUTE_ERROR = "ROUTE_ERROR",
}

// BPB sample from training
export interface BpbSample {
  step: number;
  bpb: number;
  timestamp: number;
  workerId: string;
  experimentId: string;
}

// Worker registration
export interface WorkerRegistration {
  workerId: string;
  accountId: string;
  trainerKind: "external" | "mock";
  registeredAt: number;
}

// Code generation result
export interface CodeResult {
  language: string;
  code: string;
  validationPassed: boolean;
  metrics: {
    lines: number;
    complexity: number;
  };
}

// Browser audit result
export interface AuditResult {
  url: string;
  score: number;
  issues: string[];
  timestamp: number;
}

// Routing matrix - which arm handles which events
export interface RoutingMatrix {
  [event in EventType]: McpArm[];
}

// Health check result
export interface HealthCheck {
  healthy: boolean;
  arms: Record<McpArm, boolean>;
  totalLatency: number;
  eventQueueSize: number;
}
