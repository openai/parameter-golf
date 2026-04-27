// Spawn `tri-railway` as a subprocess and return stdout/stderr/exit code.
// R5 honesty: exit code is forwarded verbatim. We never swallow non-zero.

import { spawn } from "node:child_process";
import type { Config } from "./config.js";

export interface CliResult {
  stdout: string;
  stderr: string;
  code: number;
  signal: NodeJS.Signals | null;
  cmd: string;
}

export async function runCli(
  cfg: Config,
  args: string[],
  extraEnv: Record<string, string> = {},
): Promise<CliResult> {
  const env: NodeJS.ProcessEnv = { ...process.env, ...extraEnv };
  if (cfg.railwayToken) env.RAILWAY_TOKEN = cfg.railwayToken;
  if (cfg.railwayProjectId) env.RAILWAY_PROJECT_ID = cfg.railwayProjectId;
  if (cfg.railwayEnvironmentId)
    env.RAILWAY_ENVIRONMENT_ID = cfg.railwayEnvironmentId;

  const cmd = `${cfg.triRailwayBin} ${args.join(" ")}`;

  return new Promise<CliResult>((resolve, reject) => {
    const child = spawn(cfg.triRailwayBin, args, { env, stdio: "pipe" });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
    }, cfg.cliTimeoutMs);

    child.stdout.on("data", (b: Buffer) => out.push(b));
    child.stderr.on("data", (b: Buffer) => err.push(b));
    child.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({
        stdout: Buffer.concat(out).toString("utf8"),
        stderr: Buffer.concat(err).toString("utf8"),
        code: code ?? -1,
        signal,
        cmd,
      });
    });
  });
}

/** Render a CliResult as MCP text content with an honest status block. */
export function renderResult(r: CliResult): string {
  const status = r.code === 0 ? "OK" : `FAIL (exit ${r.code})`;
  const lines = [
    `# tri-railway result: ${status}`,
    `# cmd: ${r.cmd}`,
    "",
    "## stdout",
    r.stdout.trimEnd() || "(empty)",
  ];
  if (r.stderr.trim()) {
    lines.push("", "## stderr", r.stderr.trimEnd());
  }
  return lines.join("\n");
}
