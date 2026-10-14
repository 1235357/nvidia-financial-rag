"""Installed evaluator over generated PDF text; synthetic checks are not gold.

Runs the console command outside the checkout, checks the complete frozen graph,
imports null and explicitly synthetic judgments, and refuses a changed packet.
No network, encoder, generation, or live-service process is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib import util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import sysconfig
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def invoke(command, *, cwd, environment, expected=0):
    result = subprocess.run(command, cwd=cwd, env=environment, text=True,
                            capture_output=True, timeout=90)
    require(result.returncode == expected, 'installed command exit differs: ' + result.stderr[:200])
    require('Traceback' not in result.stderr, 'command leaked traceback')
    return result


def run(wheel, output):
    root = Path(__file__).resolve().parents[1]
    require(not Path.cwd().resolve().is_relative_to(root), 'run consumer outside checkout')
    helper = runpy.run_path(str(Path(__file__).with_name('serving_startup.py')))
    identity = helper['installed_identity'](wheel)
    from nvidia_rag.evaluation_run import load_completed_run
    from nvidia_rag.evaluation_inputs import canonical_json
    require(all(util.find_spec(name) is None for name in
                ('torch', 'sentence_transformers', 'transformers', 'gradio')), 'heavyweight runtime present')
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'fresh output required')
    script = Path(sysconfig.get_paths()['scripts']) / ('nvidia-rag-evaluate.exe' if os.name == 'nt' else 'nvidia-rag-evaluate')
    require(script.is_file(), 'installed console entry point missing')
    environment = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'NVIDIA_RAG_API_TOKEN', 'NVIDIA_RAG_MODEL_TOKEN'):
        environment.pop(name, None)
    with tempfile.TemporaryDirectory(prefix='rag-evaluation-consumer-') as temporary:
        work = Path(temporary).resolve()
        fixture = helper['create_fixture'](work / 'native')
        helper['verify_fixture'](fixture['root'])
        questions = [
            {'question_id': 'synthetic-positive', 'query': '  Net income in the annual report?  ',
             'answerable': True, 'evidence_pages': [2], 'evidence_document_ids': ['synthetic-annual']},
            {'question_id': 'synthetic-negative', 'query': 'Unsupported weather forecast?',
             'answerable': False, 'evidence_pages': [], 'evidence_document_ids': []},
        ]
        for question in questions:
            question['evaluation_usage'] = {'purpose': 'development', 'development_exposed': True,
                'frozen_on': '2026-10-14', 'registration_note': 'Generated synthetic transport fixture; no financial gold'}
        question_path = work / 'questions.jsonl'
        question_path.write_bytes(b''.join(canonical_json(q) for q in questions))
        before = {str(p.relative_to(work)): digest(p) for p in work.rglob('*') if p.is_file()}
        help_result = invoke([str(script), '--help'], cwd=work, environment=environment)
        require('import-review' in help_result.stdout, 'console help lacks import command')
        run_dir = work / 'run'
        arguments = [str(script), 'run', '--artifact-dir', str(fixture['artifact_dir']),
            '--source-registry', str(fixture['source_registry']), '--questions', str(question_path),
            '--output-dir', str(run_dir), '--mode', 'development', '--backend', 'lexical',
            '--top-k', '2', '--threshold', '0.01']
        result = invoke(arguments, cwd=work, environment=environment)
        status = json.loads(result.stdout)
        require(status == {'status': 'complete', 'receipt_sha256': digest(run_dir / 'receipt.json')}, 'CLI receipt binding differs')
        require('Net income' not in result.stdout and len(result.stdout) < 200, 'CLI echoed question')
        bundle = load_completed_run(run_dir)
        require([r['query'] for r in bundle['results']['calls']] == [q['query'] for q in questions], 'query changed')
        chunks = {r['chunk_id']: r for r in json.loads((fixture['root'] / 'fixture-receipt.json').read_bytes())['chunks']}
        for row in bundle['packet']['rows']:
            require(len(row['candidates']) == 2, 'complete lexical candidate denominator differs')
            for candidate in row['candidates']:
                native = chunks[candidate['chunk']['chunk_id']]
                require(all(candidate['chunk'][key] == value for key, value in native.items()), 'candidate differs from native PDF text')
        frozen = {str(p.relative_to(run_dir)): digest(p) for p in run_dir.rglob('*') if p.is_file()}
        review = work / 'review.json'
        review.write_bytes(canonical_json(bundle['packet']))
        unknown_output = work / 'unknown-import.json'
        import_command = [str(script), 'import-review', '--run-dir', str(run_dir), '--review-path', str(review)]
        invoke([*import_command, '--output-path', str(unknown_output)], cwd=work, environment=environment)
        unknown = json.loads(unknown_output.read_bytes())
        require(unknown['summary']['question_count'] == 2 and unknown['summary']['unjudged_count'] == 4
                and unknown['summary']['judgment_coverage'] == 0.0, 'null judgment denominator changed')
        reviewed = deepcopy(bundle['packet'])
        for slot, grade in zip(reviewed['rows'][0]['judgments'], (1, 0), strict=True):
            slot.update(grade=grade, reviewer='Synthetic acceptance fixture',
                        source_check='Generated PDF text bookkeeping; not human review or financial adjudication')
        review.write_bytes(canonical_json(reviewed))
        imported_output = work / 'reviewed-import.json'
        invoke([*import_command, '--output-path', str(imported_output)], cwd=work, environment=environment)
        imported = json.loads(imported_output.read_bytes())
        summary = imported['summary']
        require(summary['judged_relevant_count'] == summary['judged_irrelevant_count'] == 1
                and summary['unjudged_count'] == 2 and summary['judgment_coverage'] == .5, 'explicit grade coverage differs')
        require(imported['generation_runs'] == 0 and imported['financial_correctness'] == 'unestablished', 'scope changed')
        require(imported['review_sha256'] == digest(review)
                and imported['source_run']['receipt_sha256'] == digest(run_dir / 'receipt.json'), 'review provenance differs')
        changed = deepcopy(reviewed)
        changed['rows'][0]['candidates'][0]['rank'] = 2
        review.write_bytes(canonical_json(changed))
        refused = work / 'refused.json'
        failure = invoke([*import_command, '--output-path', str(refused)], cwd=work, environment=environment, expected=1)
        require(not refused.exists() and len(failure.stderr) < 200, 'invalid review created output')
        require(frozen == {str(p.relative_to(run_dir)): digest(p) for p in run_dir.rglob('*') if p.is_file()}, 'review changed source run')
        require(all(digest(work / name) == wanted for name, wanted in before.items()), 'consumer changed source inputs')
        require(not any(name in sys.modules for name in ('sentence_transformers', 'transformers', 'torch')), 'semantic runtime imported')
        record = {'schema_version': 1, 'status': 'complete', 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'installed lexical console and result-bound review over generated native PDF text; synthetic grades only',
            'installed_identity': identity, 'receipt': bundle['receipt'], 'receipt_sha256': digest(run_dir / 'receipt.json'),
            'question_count': 2, 'candidate_count': 4, 'unknown_summary': unknown['summary'], 'reviewed_summary': summary,
            'review_import_sha256': digest(imported_output), 'tampered_rank_refused': True,
            'input_and_run_closures_unchanged': True, 'generation_runs': 0, 'encoder_runs': 0,
            'human_gold_count': 0, 'held_out_count': 0, 'financial_correctness': 'unestablished'}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'output_sha256': digest(output), 'question_count': 2,
                      'candidate_count': 4, 'generation_runs': 0}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.wheel, args.output)
