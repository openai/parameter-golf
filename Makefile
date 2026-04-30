.PHONY: help smoke smoke-ci smoke-verbose smoke-full clean test lint

# Default target
help:
	@echo "trios-railway Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  smoke          - Run fast smoke test (<60s, synthetic, CPU-only)"
	@echo "  smoke-ci       - Run smoke test with CI assertions (strict)"
	@echo "  smoke-verbose  - Run smoke test with verbose logging"
	@echo "  smoke-full     - Run smoke test with full validation (requires NEON_DATABASE_URL)"
	@echo "  test           - Run all tests"
	@echo "  lint           - Run clippy and rustfmt checks"
	@echo "  clean          - Clean build artifacts"

# Fast smoke test (<60s)
smoke:
	@echo "🚀 Running smoke test..."
	@cargo run --example smoke_in_memory

# Smoke test with CI assertions
smoke-ci:
	@echo "🚀 Running smoke test (CI mode)..."
	@cargo test -p trios-railway-smoke --features ci --lib

# Smoke test with verbose logging
smoke-verbose:
	@echo "🚀 Running smoke test (verbose)..."
	@RUST_LOG=debug cargo run --example smoke_in_memory

# Smoke test via IGLA race agent
smoke-agent:
	@echo "🚀 Running smoke agent (IGLA race)..."
	@cargo run -p trios-igla-race --features smoke --bin smoke_agent -- \
		--steps 1 --seed 42 --fail-on-error

# Full smoke test with database (requires NEON_DATABASE_URL)
smoke-full:
	@echo "🚀 Running full smoke test (requires NEON_DATABASE_URL)..."
	@test -n "$(NEON_DATABASE_URL)" || (echo "ERROR: NEON_DATABASE_URL not set" && exit 1)
	@cargo run -p trios-igla-race --bin seed_agent status

# Run all tests
test:
	@echo "🧪 Running all tests..."
	@cargo test --workspace

# Run linter
lint:
	@echo "🔍 Running clippy..."
	@cargo clippy --workspace -- -D warnings
	@echo "📝 Checking formatting..."
	@cargo fmt --workspace -- --check

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@cargo clean
