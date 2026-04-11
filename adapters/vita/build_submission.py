#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_O1 = Path('/Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o1/reports/o1_results.json')
DEFAULT_O2 = Path('/Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o2/reports/o2_results.json')
BASE_OUT = Path(__file__).resolve().parent / 'out'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def fmt(x):
    if x is None:
        return ''
    if isinstance(x, float):
        return f'{x:.4f}'.rstrip('0').rstrip('.')
    return str(x)


def nearly_equal(a, b, tol: float) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def build_results_tsv(rec: dict) -> str:
    m = rec.get('metrics', {})
    sweep = m.get('sweep_accuracy_percent', {}) or {}
    rows = ['prune_ratio\taccuracy_percent\tpasses_floor']
    for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]:
        key = str(r).rstrip('0').rstrip('.') if r < 1 else '1.0'
        if key not in sweep:
            key = str(r)
        acc = sweep.get(key)
        passes = '' if acc is None else ('1' if acc >= ARGS.floor else '0')
        rows.append(f'{key}\t{fmt(acc)}\t{passes}')
    return '\n'.join(rows) + '\n'


def build_readme(tag: str, repo_id: str, repo_name: str, o1: dict, o2: dict, winner: dict, submission: dict) -> str:
    m = winner.get('metrics', {})
    lines = [
        f'# VITA submission package: {tag}',
        '',
        'This package mirrors parameter-golf record style for VITA objective-first optimization.',
        '',
        '## Leaderboard-style summary',
        f"- Run: {submission['name']}",
        f"- Winner config: {submission['winner_config_id']}",
        f"- Objective: {submission['objective']['type']} (accuracy floor {submission['objective']['accuracy_floor_percent']} on {submission['objective']['dataset']})",
        f"- Score tuple: max_ratio={submission['score']['max_prune_ratio_passing_floor']}, acc_at_max={submission['score']['accuracy_at_max_ratio']}, mean_band={submission['score']['mean_accuracy_0_4_to_0_7']}",
        f"- Default operating point: prune_ratio={submission['default_operating_point']['prune_ratio']}",
        '',
        '## Evidence chain',
        f"- O1 ranking source: {ARGS.o1}",
        f"- O2 ranking source: {ARGS.o2}",
        f"- O2 reported winner: {o2.get('decision', {}).get('winner')}",
        f"- Winner train accuracy: {fmt(m.get('train_final_test_accuracy_percent'))}",
        f"- Winner confirm@0.5: {fmt(m.get('confirm_accuracy_percent'))}",
        f"- Confirm matches sweep: {m.get('confirm_matches_sweep_at_ratio')}",
        '',
        '## Files in this folder',
        '- submission.json',
        '- leaderboard_row.json',
        '- results.tsv',
        '- README.md',
        '',
        '## Repro note',
        f'- Repo id/name: {repo_id} / {repo_name}',
        '- This is a packaging adapter; source training code and logs remain in vita-autoresearch artifacts.',
        '',
    ]
    return '\n'.join(lines)


def verify_gate(winner_id: str, metrics: dict):
    expected_fields = {
        'expected_winner': ARGS.expected_winner,
        'expected_max_ratio': ARGS.expected_max_ratio,
        'expected_acc_at_max': ARGS.expected_acc_at_max,
        'expected_mean_band': ARGS.expected_mean_band,
        'expected_confirm_acc': ARGS.expected_confirm_acc,
    }

    if ARGS.verify_only:
        missing = [k for k, v in expected_fields.items() if v is None]
        if missing:
            raise SystemExit(f'ERROR: --verify-only requires: {", ".join("--" + k.replace("_", "-") for k in missing)}')

    checks = []
    if ARGS.expected_winner is not None:
        checks.append((winner_id == ARGS.expected_winner, f'winner expected={ARGS.expected_winner} actual={winner_id}'))

    if ARGS.expected_max_ratio is not None:
        checks.append((nearly_equal(metrics.get('max_prune_ratio_passing_floor'), ARGS.expected_max_ratio, ARGS.tolerance),
                       f"max_ratio expected={ARGS.expected_max_ratio} actual={metrics.get('max_prune_ratio_passing_floor')}"))

    if ARGS.expected_acc_at_max is not None:
        checks.append((nearly_equal(metrics.get('accuracy_at_max_passing_ratio'), ARGS.expected_acc_at_max, ARGS.tolerance),
                       f"acc_at_max expected={ARGS.expected_acc_at_max} actual={metrics.get('accuracy_at_max_passing_ratio')}"))

    if ARGS.expected_mean_band is not None:
        checks.append((nearly_equal(metrics.get('mean_accuracy_0_4_to_0_7'), ARGS.expected_mean_band, ARGS.tolerance),
                       f"mean_band expected={ARGS.expected_mean_band} actual={metrics.get('mean_accuracy_0_4_to_0_7')}"))

    if ARGS.expected_confirm_acc is not None:
        checks.append((nearly_equal(metrics.get('confirm_accuracy_percent'), ARGS.expected_confirm_acc, ARGS.tolerance),
                       f"confirm_acc expected={ARGS.expected_confirm_acc} actual={metrics.get('confirm_accuracy_percent')}"))

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        print('VERIFY_FAIL')
        for msg in failed:
            print(f'- {msg}')
        raise SystemExit(2)

    if checks:
        print('VERIFY_OK')
        for _, msg in checks:
            print(f'- {msg}')


def parse_args():
    p = argparse.ArgumentParser(description='Build VITA submission-shaped package from O1/O2 reports')
    p.add_argument('--o1', type=Path, default=DEFAULT_O1)
    p.add_argument('--o2', type=Path, default=DEFAULT_O2)
    p.add_argument('--repo-id', default='090')
    p.add_argument('--repo-name', default='SFW Once-for-All Pruning')
    p.add_argument('--author', default='ever')
    p.add_argument('--github-id', default='ever-oli')
    p.add_argument('--floor', type=float, default=74.5)
    p.add_argument('--tag', default=f'vita-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}')

    p.add_argument('--verify-only', action='store_true', help='Only run expectation checks; do not write package files')
    p.add_argument('--expected-winner', default=None)
    p.add_argument('--expected-max-ratio', type=float, default=None)
    p.add_argument('--expected-acc-at-max', type=float, default=None)
    p.add_argument('--expected-mean-band', type=float, default=None)
    p.add_argument('--expected-confirm-acc', type=float, default=None)
    p.add_argument('--tolerance', type=float, default=1e-6)

    return p.parse_args()


ARGS = parse_args()


def main():
    o1 = read_json(ARGS.o1)
    o2 = read_json(ARGS.o2)

    winner_id = o2.get('decision', {}).get('winner')
    if not winner_id:
        raise SystemExit('ERROR: o2 decision.winner missing')

    cfgs = {c.get('config_id'): c for c in o2.get('configs', [])}
    winner = cfgs.get(winner_id)
    if not winner:
        raise SystemExit(f'ERROR: winner config {winner_id} not found in o2 configs')

    m = winner.get('metrics', {})

    # Optional strict gate: fail hard if expected values drift.
    verify_gate(winner_id, m)

    if ARGS.verify_only:
        return

    submission = {
        'name': f'VITA-{ARGS.repo_id} Objective-B package ({ARGS.tag})',
        'author': ARGS.author,
        'github_id': ARGS.github_id,
        'date': datetime.now(timezone.utc).isoformat(),
        'track': 'vita-objective-b',
        'repo_id': ARGS.repo_id,
        'repo_name': ARGS.repo_name,
        'winner_config_id': winner_id,
        'objective': {
            'type': o2.get('objective', {}).get('type', 'B'),
            'dataset': o2.get('objective', {}).get('dataset', 'cifar10'),
            'accuracy_floor_percent': ARGS.floor,
        },
        'score': {
            'max_prune_ratio_passing_floor': m.get('max_prune_ratio_passing_floor'),
            'accuracy_at_max_ratio': m.get('accuracy_at_max_passing_ratio'),
            'mean_accuracy_0_4_to_0_7': m.get('mean_accuracy_0_4_to_0_7'),
            'confirm_accuracy_percent': m.get('confirm_accuracy_percent'),
        },
        'default_operating_point': o2.get('decision', {}).get('default_operating_point', {'prune_ratio': 0.5}),
        'blurb': (
            f"Objective-B optimized package from O1/O2 for repo {ARGS.repo_id}; "
            f"winner={winner_id}, max_ratio={m.get('max_prune_ratio_passing_floor')}, "
            f"acc@max={m.get('accuracy_at_max_passing_ratio')}"
        ),
        'source_reports': {
            'o1_results_json': str(ARGS.o1),
            'o2_results_json': str(ARGS.o2),
        },
        'artifacts': winner.get('artifacts', {}),
    }

    leaderboard_row = {
        'run': submission['name'],
        'repo_id': ARGS.repo_id,
        'repo_name': ARGS.repo_name,
        'winner': winner_id,
        'max_ratio': m.get('max_prune_ratio_passing_floor'),
        'acc_at_max': m.get('accuracy_at_max_passing_ratio'),
        'mean_0.4_0.7': m.get('mean_accuracy_0_4_to_0_7'),
        'confirm_0.5': m.get('confirm_accuracy_percent'),
        'author': ARGS.author,
        'date': submission['date'],
        'info': 'README.md',
    }

    out = BASE_OUT / ARGS.tag
    out.mkdir(parents=True, exist_ok=True)

    (out / 'submission.json').write_text(json.dumps(submission, indent=2) + '\n', encoding='utf-8')
    (out / 'leaderboard_row.json').write_text(json.dumps(leaderboard_row, indent=2) + '\n', encoding='utf-8')
    (out / 'results.tsv').write_text(build_results_tsv(winner), encoding='utf-8')
    (out / 'README.md').write_text(build_readme(ARGS.tag, ARGS.repo_id, ARGS.repo_name, o1, o2, winner, submission), encoding='utf-8')

    print(str(out / 'submission.json'))
    print(str(out / 'README.md'))
    print(str(out / 'results.tsv'))
    print(str(out / 'leaderboard_row.json'))


if __name__ == '__main__':
    main()
