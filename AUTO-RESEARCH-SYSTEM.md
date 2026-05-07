# Auto-Research Feedback Loop System for OpenAI Parameter Golf

## Overview

This system designs three interconnected auto-research loops to optimize the training of language models under 16MB for the lowest bits-per-byte (bpb) score in the OpenAI Parameter Golf competition. The loops automate code mutation, research discovery, and strategy meta-optimization, leveraging free AI models, web search, and a RunPod GPU cluster.

## Loop 1: Code Mutation Loop (Model Optimization)

### 1. Exact Inputs and Outputs
- **Input**: 
  - Current best `train_gpt.py` file
  - Current bpb score from the latest training run
  - Hyperparameter configuration used in the last run
- **Output**: 
  - Ranked list of code mutations (each mutation includes: description, diff, predicted impact score, priority rank)

### 2. AI Models for Each Step
- **Mutation Generation**: Nemotron 3 Super (free) or DeepSeek V3.2 (free) - generates architectural changes and hyperparameter tweaks
- **Mutation Ranking**: Nemotron 3 Super - scores mutations based on predicted impact and novelty
- **Implementation**: Codex (free via ACP) - applies mutations to create new `train_gpt.py` variants

### 3. Connections to Other Loops
- Receives bottleneck signals from Loop 2 (e.g., "quantization is limiting") to focus mutations
- Feeds mutation results (bpb scores, training stability) to Loop 3 for strategy analysis
- Receives research findings from Loop 2 to implement new techniques (e.g., novel quantization methods)

### 4. Trigger Conditions
- After each training run completes on RunPod
- When Loop 2 identifies a specific technical bottleneck requiring code changes
- Every 6 hours as a fallback to explore random mutations

### 5. Convergence Detection
- Track bpb improvement per mutation batch; if median improvement < 0.005 bpb over 10 consecutive batches, reduce mutation aggressiveness
- Detect circular mutations by hashing mutation diffs; if same mutation reappears, increase exploration noise
- Stop generating mutations in a category if 20 variants show no improvement

### 6. Concrete Implementation
- Files:
  - `parameter-golf/loops/code_mutation.py`: Main loop controller
  - `parameter-golf/models/mutation_prompts.py`: Prompts for Nemotron/DeepSeek
  - `parameter-golf/scripts/apply_mutation.py`: Codex-invoked mutation applier
  - `parameter-golf/data/mutation_history.jsonl`: Log of all mutations tried
- Cron Job: `0 */6 * * * cd /home/ferrante42/.openclaw/workspace/parameter-golf && python3 loops/code_mutation.py --trigger scheduled`

### 7. Measuring Auto-Research Improvement
- Track mutation success rate (% of mutations that improve bpb)
- Measure average bpb improvement per accepted mutation over time
- Monitor mutation diversity (entropy of mutation types)

## Loop 2: Research Discovery Loop (Knowledge Optimization)

### 1. Exact Inputs and Outputs
- **Input**:
  - Current techniques being used (from mutation loop and strategy loop)
  - Latest bpb score and plateau detection signals from Loop 1
  - Bottleneck tags (e.g., "quantization", "attention efficiency", "optimizer")
- **Output**:
  - Curated list of research papers/repos with: title, URL, summary, relevance score, estimated bpb impact
  - Ranked by predicted impact on current bottleneck

### 2. AI Models for Each Step
- **Bottleneck Analysis**: DeepSeek V3.2 (free) - analyzes training logs to identify specific bottlenecks
- **Search Query Generation**: Nemotron 3 Super (free) - converts bottlenecks to precise search queries
- **Paper Relevance Scoring**: Gemini Pro (free via API) - ranks search results by relevance and novelty
- **Summary Generation**: DeepSeek V3.2 - creates concise summaries of papers/repos

### 3. Connections to Other Loops
- Feeds new techniques to Loop 1 for implementation as code mutations
- Provides research effectiveness data to Loop 3 for strategy optimization
- Receives failed technique reports from Loop 3 to avoid dead-end research

### 4. Trigger Conditions
- When Loop 1 reports bpb plateau (<0.001 improvement over 5 runs)
- When a specific bottleneck persists for 3 consecutive mutation batches
- Daily at 02:00 CDT to proactively explore new areas

### 5. Convergence Detection
- Track research-to-implementation conversion rate; if <10% over 20 papers, broaden search scope
- Detect exhausted research areas by tracking citation patterns; if all top papers from 2023-2024 tested with no improvement, shift to older/archive papers
- Prevent circular research by maintaining a hash of searched queries; avoid exact repeats for 30 days

### 6. Concrete Implementation
- Files:
  - `parameter-golf/loops/research_discovery.py`: Main loop controller
  - `parameter-golf/models/search_prompts.py`: Prompts for query generation and relevance scoring
  - `parameter-golf/scripts/web_search.py`: Brave API search wrapper
  - `parameter-golf/scripts/fetch_and_summarize.py`: Extracts and summarizes content
  - `parameter-golf/data/research_memory.jsonl`: Log of all researched papers/repos
- Cron Job: `0 2 * * * cd /home/ferrante42/.openclaw/workspace/parameter-golf && python3 loops/research_discovery.py --trigger scheduled`

### 7. Measuring Auto-Research Improvement
- Track research yield: bpb improvement per hour of research time
- Measure dead-end rate (% of researched techniques that fail to improve bpb when implemented)
- Monitor novelty of research sources (percentage from pre-2023 or obscure venues)

## Loop 3: Strategy Meta-Loop (Process Optimization)

### 1. Exact Inputs and Outputs
- **Input**:
  - Complete experiment history: bpb scores, mutation types, research sources, time spent, compute cost (RunPod hours)
  - Bottleneck evolution over time
  - Resource utilization metrics (GPU hours, model API costs)
- **Output**:
  - Updated priorities: which mutation types to emphasize, which research areas to explore
  - Resource allocation: RunPod hours allocation, model usage quotas
  - Agent assignment adjustments: which free models to use for which tasks

### 2. AI Models for Each Step
- **Pattern Analysis**: Gemini Pro (free) - identifies correlations between changes and bpb improvements
- **Priority Setting**: Nemotron 3 Super (free) - generates strategic recommendations
- **Resource Optimization**: DeepSeek V3.2 (free) - simulates allocation strategies

### 3. Connections to Other Loops
- Sets mutation aggressiveness and focus areas for Loop 1
- Determines when to trigger Loop 2 and what bottlenecks to investigate
- Adjusts which models are used in Loops 1 and 2 based on historical performance

### 4. Trigger Conditions
- After every 20 completed experiments (combined loops)
- Weekly deep analysis every Sunday at 03:00 CDT
- When compute efficiency drops below threshold (<0.01 bpb improvement per GPU hour)

### 5. Convergence Detection
- Detect strategic stagnation if priority recommendations repeat unchanged for 3 cycles
- Prevent over-optimization of single technique by enforcing exploration minimum (20% resources to novel approaches)
- Monitor for local maxima by simulating random restarts in strategy space

### 6. Concrete Implementation
- Files:
  - `parameter-golf/loops/strategy_meta.py`: Main loop controller
  - `parameter-golf/models/strategy_prompts.py`: Prompts for analysis and recommendation
  - `parameter-golf/scripts/experiment_logger.py`: Logs all experiment data
  - `parameter-golf/scripts/resource_allocator.py`: Computes optimal resource distribution
  - `parameter-golf/data/experiment_history.jsonl`: Complete log of all runs
  - `parameter-golf/data/strategy_decisions.jsonl`: Log of meta-loop decisions
- Cron Job: `0 3 * * 0 cd /home/ferrante42/.openclaw/workspace/parameter-golf && python3 loops/strategy_meta.py --trigger weekly`

### 7. Measuring Auto-Research Improvement
- Track strategy adaptation speed: time to shift focus when a technique plateaus
- Measure compute efficiency improvement (bpb per GPU hour) over time
- Monitor prediction accuracy of strategy loop (how often predicted high-impact areas deliver)

## Interconnections Between Loops

1. **Loop 1 → Loop 2**: When mutation loop plateaus, sends bottleneck analysis to trigger research discovery
2. **Loop 2 → Loop 1**: Research loop outputs new techniques as prioritized mutation targets
3. **Loop 1 & 2 → Loop 3**: All experiment results feed into strategy meta-loop for analysis
4. **Loop 3 → Loop 1 & 2**: Strategy loop sets priorities, resource allocation, and model assignments for both loops

## Research Quality Scoring System

Each research paper/repo is scored based on:
- **Direct Impact Score (40%)**: Measured bpb improvement when implemented (0-40 points)
- **Novelty Score (20%)**: How different from current techniques (0-20 points)
- **Implementation Feasibility (20%)**: Ease of integrating into current codebase (0-20 points)
- **Replicability (10%)**: Clarity of methodology and available code (0-10 points)
- **Source Credibility (10%)**: Venue reputation and author expertise (0-10 points)

Total score: 0-100 points. Only research scoring >60 is prioritized for immediate implementation.

## Research Memory

Implemented as `parameter-golf/data/research_memory.jsonl` with entries:
```jsonl
{
  "timestamp": "2026-03-24T00:27:00Z",
  "source_type": "paper|repo|blog",
  "title": "Title of work",
  "url": "https://...",
  "bottleneck_addressed": ["quantization", "attention"],
  "summary": "Brief summary",
  "relevance_score": 85,
  "novelty_score": 70,
  "implementation_attempted": true,
  "implementation_result": {"bpb_change": -0.023, "stable": true},
  "notes": "Worked well with our architecture"
}
```
- Tracks what's been tried, what worked, what failed
- Enables duplicate prevention and learning from past attempts
- Supports automatic gap detection by identifying unexplored bottleneck-technique combinations

## Automatic Gap Detection

Process:
1. Maintain a matrix of known bottlenecks (rows) vs. technique categories (columns)
2. When a technique is tried in a bottleneck cell, mark it as attempted
3. Weekly, identify cells with zero attempts but high theoretical potential (based on research memory and literature)
4. Generate gap report: "Top 5 unexplored bottleneck-technique combinations"
5. Feed gaps to Loop 2 as priority research areas

Gap detection algorithm:
- Score each unexplored cell by: (literature mention count) × (novelty of combination) / (years since last explored)
- Top-scoring gaps become research priorities

## Daily Research Report Generator

Generates `/home/ferrante42/.openclaw/workspace/parameter-golf/reports/DAILY_RESEARCH_REPORT.md` each morning at 06:00 CDT.

Report includes:
- **Summary of Previous Day**: bpb changes, mutations tried, research conducted
- **Loop 1 Performance**: Mutation success rate, best mutation, convergence status
- **Loop 2 Performance**: Papers researched, promising findings, dead ends encountered
- **Loop 3 Insights**: Strategic shifts, resource allocation changes
- **Gap Analysis**: Top 3 unexplored areas
- **Recommendations**: Specific actions for next day
- **Metrics Trends**: bpb over time, compute efficiency, research yield

Implementation:
- File: `parameter-golf/scripts/generate_daily_report.py`
- Cron: `0 6 * * * cd /home/ferrante42/.openclaw/workspace/parameter-golf && python3 scripts/generate_daily_report.py`

## Implementation Files Structure

```
parameter-golf/
├── loops/
│   ├── code_mutation.py
│   ├── research_discovery.py
│   └── strategy_meta.py
├── models/
│   ├── mutation_prompts.py
│   ├── search_prompts.py
│   └── strategy_prompts.py
├── scripts/
│   ├── apply_mutation.py
│   ├── web_search.py
│   ├── fetch_and_summarize.py
│   ├── experiment_logger.py
│   ├── resource_allocator.py
│   └── generate_daily_report.py
├── data/
│   ├── mutation_history.jsonl
│   ├── research_memory.jsonl
│   ├── experiment_history.jsonl
│   └── strategy_decisions.jsonl
├── reports/
└── AUTO-RESEARCH-SYSTEM.md
```

## Getting Started Tonight

1. **Initialize data files**: Create empty JSONL files in `data/` directory
2. **Set up API keys**: Ensure Brave Search, RunPod, and free model API keys are in environment
3. **Start initial run**: Execute current best `train_gpt.py` to establish baseline
4. **Activate loops**: 
   - Start code mutation loop: `python3 loops/code_mutation.py --trigger initial`
   - Schedule cron jobs using `crontab -e` with the provided schedules
5. **Monitor**: Check logs in `logs/` directory and daily reports

This system is designed to be operational within hours and will continuously improve the model's bpb score through automated research and mutation cycles.