#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_O2 = Path('/Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o2/reports/o2_results.json')
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / 'runs'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser(description='Initialize non-claiming benchmark-run scaffold for VITA->parameter-golf')
    p.add_argument('--run-tag', required=True, help='Run tag, e.g. 090-c2-benchmark-attempt-01')
    p.add_argument('--o2', type=Path, default=DEFAULT_O2)
    p.add_argument('--source-repo-id', default='090')
    p.add_argument('--source-config-id', default='C2')
    p.add_argument('--hardware', default='TBD')
    p.add_argument('--benchmark-track', default='parameter-golf')
    return p.parse_args()


def main():
    args = parse_args()

    o2 = read_json(args.o2)
    winner = o2.get('decision', {}).get('winner')
    cfgs = {c.get('config_id'): c for c in o2.get('configs', [])}
    source_cfg = cfgs.get(args.source_config_id, {})

    run_dir = RUNS_DIR / args.run_tag
    evidence_dir = run_dir / 'evidence'
    evidence_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'created_at': now_iso(),
        'run_tag': args.run_tag,
        'status': 'initialized_non_claiming',
        'benchmark_track': args.benchmark_track,
        'non_claiming': True,
        'claims_allowed': False,
        'source': {
            'o2_results_json': str(args.o2),
            'source_repo_id': args.source_repo_id,
            'source_config_id': args.source_config_id,
            'o2_reported_winner': winner,
            'source_config_metrics': source_cfg.get('metrics', {}),
            'source_config_artifacts': source_cfg.get('artifacts', {}),
        },
        'benchmark_execution': {
            'hardware': args.hardware,
            'pipeline_executed': False,
            'training_executed': False,
            'evaluation_executed': False,
            'final_metric_extracted': False,
            'artifact_size_verified': False,
        },
        'blockers': [
            'Task/model/metric adaptation required before benchmark comparability.',
            'Real benchmark pipeline not yet executed.',
        ],
        'expected_evidence_files': [
            'evidence/train.log',
            'evidence/eval.log',
            'evidence/submission.json',
            'evidence/artifact_sizes.json',
            'evidence/environment.json',
        ],
    }

    claims_guard = {
        'created_at': now_iso(),
        'run_tag': args.run_tag,
        'non_claiming': True,
        'claims_allowed': False,
        'forbidden_claims': [
            'leaderboard competitiveness',
            'challenge-equivalent performance',
            'record/SOTA claims',
            'hardware-comparable efficiency',
        ],
        'unlock_requirements': [
            'Real benchmark logs in evidence/',
            'Exact benchmark metric extracted from logs',
            'Artifact-size accounting verified',
            'Protocol/hardware comparability statement completed',
        ],
    }

    commands_sh = """#!/usr/bin/env bash
set -euo pipefail

# NON-CLAIMING scaffold commands.
# Replace TBD commands with real benchmark pipeline commands before execution.

RUN_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
EVIDENCE_DIR=\"${RUN_DIR}/evidence\"
mkdir -p \"${EVIDENCE_DIR}\"

echo \"[info] benchmark scaffold only; no benchmark run executed yet\"

echo \"TODO: run benchmark training pipeline and tee logs to ${EVIDENCE_DIR}/train.log\"
echo \"TODO: run benchmark eval pipeline and tee logs to ${EVIDENCE_DIR}/eval.log\"
echo \"TODO: produce ${EVIDENCE_DIR}/submission.json and ${EVIDENCE_DIR}/artifact_sizes.json\"
echo \"TODO: record environment details to ${EVIDENCE_DIR}/environment.json\"
"""

    expected_outputs_md = """# Expected outputs and acceptance checks

Required outputs (from real benchmark execution):
- evidence/train.log
- evidence/eval.log
- evidence/submission.json
- evidence/artifact_sizes.json
- evidence/environment.json

Acceptance checks:
- [ ] Benchmark pipeline actually executed (not scaffold-only)
- [ ] Exact benchmark metric line extracted from logs
- [ ] Artifact-size accounting verified from produced artifacts
- [ ] Hardware/protocol comparability documented
- [ ] claims_guard updated from non-claiming only when all checks pass
"""

    notes_md = f"""# Notes for {args.run_tag}

Initialized: {now_iso()}
Source repo/config: {args.source_repo_id}/{args.source_config_id}
O2 winner at init time: {winner}

This directory is initialized as NON-CLAIMING.
"""

    write(run_dir / 'benchmark_manifest.json', json.dumps(manifest, indent=2) + '\n')
    write(run_dir / 'benchmark_claims_guard.json', json.dumps(claims_guard, indent=2) + '\n')
    write(run_dir / 'benchmark_commands.sh', commands_sh)
    write(run_dir / 'expected_outputs.md', expected_outputs_md)
    write(run_dir / 'notes.md', notes_md)

    print(str(run_dir / 'benchmark_manifest.json'))
    print(str(run_dir / 'benchmark_claims_guard.json'))
    print(str(run_dir / 'benchmark_commands.sh'))
    print(str(run_dir / 'expected_outputs.md'))
    print(str(run_dir / 'notes.md'))


if __name__ == '__main__':
    main()
