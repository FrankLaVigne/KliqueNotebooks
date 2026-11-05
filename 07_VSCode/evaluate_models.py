"""Small reproducible gateway comparison. Dry run unless --live is supplied."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = 'klique-eval-system-v1'
SYSTEM = 'Follow the user task precisely. Do not invent missing facts. Use the requested output format.'
MAX_CALLS = 24


def load_suite(path):
    raw = Path(path).read_bytes()
    suite = json.loads(raw)
    if not isinstance(suite.get('version'), str) or not suite['version']:
        raise ValueError('Question suite needs a version')
    questions = suite.get('questions', [])
    if not 1 <= len(questions) <= 12:
        raise ValueError('Use 1..12 questions')
    ids = set()
    for q in questions:
        if not isinstance(q.get('id'), str) or q['id'] in ids:
            raise ValueError('Question IDs must be unique strings')
        ids.add(q['id'])
        if q.get('check') not in ('exact', 'json') or 'expected' not in q:
            raise ValueError('Each question needs an exact or json check and expected value')
        if not isinstance(q.get('prompt'), str) or not 1 <= len(q['prompt']) <= 4000:
            raise ValueError('Question prompts must contain 1..4000 characters')
    return suite, hashlib.sha256(raw).hexdigest()


def score_answer(text, question):
    if question['check'] == 'exact':
        return text.strip() == question['expected']
    try:
        parsed = json.loads(text)
        # Canonical JSON comparison distinguishes JSON true from integer 1.
        return json.dumps(parsed, sort_keys=True) == json.dumps(question['expected'], sort_keys=True)
    except (ValueError, TypeError):
        return False


def run_evaluation(client, models, suite, suite_hash, profile='unknown', repeats=1, max_tokens=256):
    if not models or len(set(models)) != len(models):
        raise ValueError('Use distinct aliases')
    calls = len(models) * len(suite['questions']) * repeats
    if not 1 <= repeats <= 3 or calls > MAX_CALLS or not 1 <= max_tokens <= 1024:
        raise ValueError(f'Limit: {MAX_CALLS} calls, 1..3 repeats, 1..1024 output tokens')
    rows = []
    halted_kind = None
    # Alternate alias order between repeats to reduce (not eliminate) order bias.
    for repeat in range(repeats):
        for q in suite['questions']:
            for model in (models if repeat % 2 == 0 else list(reversed(models))):
                row = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), profile=profile,
                           requested_alias=model, served_model=None, finish_reason=None, question_id=q['id'], repeat=repeat + 1,
                           prompt_version=PROMPT_VERSION, prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
                           suite_version=suite['version'], suite_sha256=suite_hash,
                           temperature=0, max_tokens=max_tokens, status='error', error_kind=None,
                           http_status=None, answer=None, passed=None, prompt_tokens=None,
                           completion_tokens=None, total_tokens=None, latency_seconds=None)
                if halted_kind:
                    row.update(status='skipped', error_kind='halted_after_' + halted_kind)
                    rows.append(row)
                    continue
                started = time.perf_counter()
                try:
                    reply = client.complete([{'role': 'system', 'content': SYSTEM},
                                             {'role': 'user', 'content': q['prompt']}],
                                            model=model, max_tokens=max_tokens, temperature=0)
                    # Preserve accounting and truncation evidence even when no answer text exists.
                    row['served_model'] = reply.get('model')
                    usage = reply.get('usage')
                    usage = usage if isinstance(usage, dict) else {}
                    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        value = usage.get(key)
                        row[key] = value if type(value) is int and value >= 0 else None
                    choice = reply['choices'][0]
                    row['finish_reason'] = choice.get('finish_reason')
                    text = choice['message']['content']
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('Missing text answer')
                    row.update(status='ok', answer=text, passed=score_answer(text, q))
                except Exception as exc:
                    # Never export raw exception strings: they may contain credentials or response bodies.
                    row.update(status='error', passed=None)
                    row['error_kind'] = getattr(exc, 'kind', type(exc).__name__)
                    row['http_status'] = getattr(exc, 'status_code', None)
                    if row['error_kind'] in {'authentication', 'budget', 'ambiguous_limit', 'rate_limit', 'transport'}:
                        halted_kind = row['error_kind']
                row['latency_seconds'] = round(time.perf_counter() - started, 6)
                rows.append(row)
    return rows


def export_rows(rows, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'results.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (directory / 'results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            # CSV is for human spreadsheet review; neutralize formula injection.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v
                             for k, v in row.items()})


def plot_rows(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aliases = list(dict.fromkeys(row['requested_alias'] for row in rows))
    fig, ax = plt.subplots(figsize=(8, 5))
    missing = []
    for alias in aliases:
        subset = [r for r in rows if r['requested_alias'] == alias]
        attempts = [r for r in subset if r['status'] != 'skipped']
        successful = [r for r in attempts if r['status'] == 'ok']
        if not successful:
            missing.append(alias)
            continue
        rate = sum(r['passed'] is True for r in attempts) / len(attempts)
        latency = sum(r['latency_seconds'] for r in successful) / len(successful)
        ax.scatter([latency], [rate])
        ax.annotate(f"{alias} (n={len(attempts)})", (latency, rate), xytext=(5, 5), textcoords='offset points')
    ax.set(ylim=(-.05, 1.12), xlabel='Mean client-observed seconds per successful request',
           ylabel='Passes / attempts (errors count as non-passes)',
           title='Task pass rate versus request latency; skipped rows excluded')
    if missing:
        ax.text(.02, .02, 'No successful responses: ' + ', '.join(missing), transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--models', nargs='+', default=['alias-a', 'alias-b'])
    parser.add_argument('--questions', type=Path, default=Path(__file__).with_name('eval_questions.json'))
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('lab_outputs') / ('bakeoff-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    suite, digest = load_suite(args.questions)
    count = len(args.models) * len(suite['questions']) * args.repeats
    if not 1 <= args.repeats <= 3 or count > MAX_CALLS or len(set(args.models)) != len(args.models):
        parser.error(f'Use distinct aliases, 1..3 repeats and no more than {MAX_CALLS} calls')
    print(f'Plan: {count} requests; at most {count * 256} requested output tokens; no automatic retries.')
    if not args.live:
        print('Dry run only. Replace aliases with names in your profile and add --live to make requests.')
        return
    from gateway_client import GatewayClient, GatewayConfig
    config = GatewayConfig.from_environment()
    if any(model not in config.models for model in args.models):
        parser.error('Every alias must be listed in KLIQUE_MODEL_ALIASES')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new directory before making calls')
    rows = run_evaluation(GatewayClient(config), args.models, suite, digest, config.profile, args.repeats)
    export_rows(rows, args.output)
    print(f'Saved {len(rows)} rows to {args.output}; errors={sum(r["status"] == "error" for r in rows)}; skipped={sum(r["status"] == "skipped" for r in rows)}')
    if args.plot:
        plot_rows(rows, args.output / 'quality_latency.png')


if __name__ == '__main__':
    main()
