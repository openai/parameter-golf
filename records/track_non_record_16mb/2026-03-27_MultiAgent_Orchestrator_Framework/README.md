# Multi-Agent Orchestrator Framework for Parameter Golf

**Author:** genji0306 (Opensens Research)
**Date:** 2026-03-27
**Track:** Non-record / Tooling Submission

## Summary

A multi-agent orchestration framework that systematically coordinates solution analysis, hypothesis generation, experiment design, and automated testing for the Parameter Golf challenge. The framework is designed to accelerate iteration toward competitive submissions by structuring the research and experimentation workflow.

## Architecture

The orchestrator coordinates four specialized agents:

- **AL** (Solution Analyst): Normalizes and catalogs existing leaderboard submissions into structured, comparable artifacts.
- **RL** (Research Lead): Generates ranked hypotheses with testable verify paths and implementation cost estimates.
- **M** (Memento Designer): Designs a reusable skill schema with utility tracking and reflection-based learning across experiment runs.
- **CAR** (Experiment Runner): Executes experiments via Codex Autoresearch against generated experiment specifications.

## Components

- **Control plane CLI** for run management, artifact validation, and experiment synthesis
- **JSON schemas** for all inter-agent artifacts (hypotheses, experiments, reflections, skills)
- **Metric extraction and budget-check helpers** for Parameter Golf constraints (16MB artifact, 10-min wall clock)
- **Experiment materializer** that creates isolated working records from existing leaderboard entries
- **Agent Flow bridge** for real-time orchestration visualization

## Local Validation

Validated on Mac Mini M4 16GB using the MLX training path. The framework successfully managed multiple experiment configurations and tracked results across runs.

## Repository

Full orchestrator source: [github.com/genji0306/parameter-golf-orchestrator](https://github.com/genji0306/parameter-golf-orchestrator)

## Status

This is a tooling and methodology submission, not a model record. Active experimentation toward a competitive leaderboard submission is in progress.
