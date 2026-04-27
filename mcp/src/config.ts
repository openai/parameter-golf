// Configuration for trios-railway-mcp.
// All settings are derived from environment variables so the same binary can be
// used in dev (local cargo build) and prod (npx-installed wrapper).
//
// Anchor: phi^2 + phi^-2 = 3.

export interface Config {
  /** Path to the `tri-railway` binary. */
  triRailwayBin: string;
  /** Railway API token (project-scoped UUID or team JWT). */
  railwayToken: string | undefined;
  /** Railway project ID for IGLA. */
  railwayProjectId: string;
  /** Railway environment ID (production). */
  railwayEnvironmentId: string;
  /** Default container image for new services. */
  defaultImage: string;
  /** Neon DSN for hive_status reads (read-only is fine). */
  neonDsn: string | undefined;
  /** Maximum CLI execution time in ms before timeout. */
  cliTimeoutMs: number;
}

export function loadConfig(): Config {
  return {
    triRailwayBin: process.env.TRI_RAILWAY_BIN ?? "tri-railway",
    railwayToken: process.env.RAILWAY_TOKEN,
    railwayProjectId:
      process.env.RAILWAY_PROJECT_ID ?? "e4fe33bb-3b09-4842-9782-7d2dea1abc9b",
    railwayEnvironmentId:
      process.env.RAILWAY_ENVIRONMENT_ID ??
      "54e293b9-00a9-4102-814d-db151636d96e",
    defaultImage:
      process.env.TRIOS_DEFAULT_IMAGE ??
      "ghcr.io/ghashtag/trios-trainer-igla:latest",
    neonDsn: process.env.TRIOS_NEON_DSN,
    cliTimeoutMs: Number(process.env.TRI_RAILWAY_TIMEOUT_MS ?? "120000"),
  };
}
