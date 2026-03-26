# Parameter Golf — Tools & Resources

## Compute
| Tool | Purpose | Cost | Credentials |
|------|---------|------|-------------|
| **RunPod** | GPU training (1× and 8×H100) | $2.69-21.52/hr | `~/.openclaw/credentials/runpod-api-key` |
| **Kaggle** | Free GPU smoke tests (2×T4, 30 hrs/mo) | Free | `~/.openclaw/credentials/kaggle-api-key` |
| **Local (WSL2)** | CPU syntax checks, smoke tests | Free | N/A |

## Code & Repos
| Tool | Purpose | Location |
|------|---------|----------|
| **Our fork** | Competition submission repo | `github.com/nickferrantelive/parameter-golf` |
| **Upstream** | OpenAI's official repo | `github.com/openai/parameter-golf` |
| **Local clone** | Working directory | `~/workspace/parameter-golf/` |

## AI Models (Research Army)
| Model | Alias | Cost | Best For |
|-------|-------|------|----------|
| **Claude Opus 4.6** | opus | $$$ | Orchestration, review, strategy |
| **Claude Sonnet 4.6** | sonnet (Artemis) | $$ | Code review, refinement |
| **GPT 5.4 Codex** | codex | Free | Code implementation |
| **DeepSeek V3.2** | deepseek (Prometheus) | ¢ | Deep research, reliable |
| **Grok 4.1 Fast** | grok (Hermes) | $ | Social intel, broad research |
| **MiMo V2 Pro** | hunter (Achilles) | Free | Strategic analysis |
| **MiMo V2 Omni** | healer (Chiron) | Free | Planning, cross-domain |
| **Nemotron 120B** | nemotron | Free | Research (unreliable) |
| **DeepSeek R1** | r1 (Pythia) | Free | Deep reasoning, first principles |
| **Qwen3 Coder** | qwen-coder | Free | Chinese ecosystem (unreliable) |

## Autonomous Systems
| Tool | Purpose | Credentials |
|------|---------|-------------|
| **AgentMail** | Email for signups/verification | `atlas_email@agentmail.to` |
| **2Captcha** | CAPTCHA solving service | `~/.openclaw/credentials/2captcha-api-key` |
| **Browser (OpenClaw)** | Web automation, signups | Built-in |
| **Twilio** | Phone number for verification | +1-855-780-4437 |

## Competition Links
| Resource | URL |
|----------|-----|
| Competition page | https://openai.com/index/parameter-golf/ |
| GitHub repo | https://github.com/openai/parameter-golf |
| Our fork | https://github.com/nickferrantelive/parameter-golf |
| Compute grant form | https://openai.com/index/parameter-golf/#credit-form |
| Job interest form | https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf |
| RunPod console | https://www.runpod.io/console/pods |
| RunPod template | `y5cejece4j` (ALWAYS use this) |
| Kaggle profile | https://www.kaggle.com/atlasairesearch |

## Research Files (17 total)
| File | Content |
|------|---------|
| `RESEARCH-FINDINGS.md` | Novel techniques (BitNet, RWKV, etc.) |
| `LANDSCAPE-RESEARCH.md` | Global company/lab landscape |
| `RESEARCH-SUMMARY.md` | Executive summary |
| `SOCIAL-INTEL.md` | Competition social intel |
| `STRATEGIC-ANALYSIS.md` | Competitive gap analysis (Achilles) |
| `IMPLEMENTATION-ROADMAP.md` | 5-week plan |
| `AUTO-RESEARCH-SYSTEM.md` | Auto-research loop design |
| `CHINA-ML-LANDSCAPE.md` | Chinese ML ecosystem |
| `BROAD-AI-LANDSCAPE.md` | 10 categories, 50+ architectures (Grok) |
| `FIRST-PRINCIPLES-ANALYSIS.md` | First-principles reasoning (R1) |
| `TRAINING-PARADIGM-SHIFTS.md` | Paradigm shifts |
| `CROSS-DOMAIN-INSIGHTS.md` | Info theory, neuroscience |
| `QUANTUM-AND-EMERGING-COMPUTE.md` | Quantum, tensor networks |
| `HIDDEN-AI-FRONTIER.md` | Stealth startups, breakthroughs |
| `HIDDEN-AI-FRONTIER-GROK.md` | Parallel angle (Grok) |
| `ACHILLES-STRATEGIC-SYNTHESIS.md` | Strategic synthesis + winning arch |
| `TECHNIQUE-DEEP-DIVE.md` | Implementation-level technique analysis |
| `NATURE-INSPIRED-SOLUTIONS.md` | Biology/physics inspired approaches |
| `HUNTER-ALPHA-SYNTHESIS.md` | 916-line full playbook (Nick's session) |
| `CODEC-ARCHITECTURE-SPEC.md` | Codec model spec |
| `DEVILS-ADVOCATE.md` | R1 critique of all 5 models |

## Model Files (8 models, 4 built)
| Model | File | Status |
|-------|------|--------|
| 1 Codec | `train_gpt_model1.py` | ✅ Built |
| 2 Recursive | `train_gpt_model2.py` | ✅ Built |
| 3 Hybrid | `train_gpt_model3.py` | ✅ Built |
| 4 Optimized | `train_gpt_model4.py` | ✅ Built |
| 5 Frankenstein | — | ⏳ After testing |
| 6 Hive | `specs/model6-hive.md` | Specced only |
| 7 Immune | `specs/model7-immune.md` | Specced only |
| 8 Crystal | `specs/model8-crystal.md` | Specced only |

## Budget
| Item | Amount |
|------|--------|
| RunPod starting balance | $20 |
| Spent so far | ~$7 |
| Remaining | ~$13 |
| Compute grant (pending) | Up to $1,000 |
