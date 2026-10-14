"""Installed retrieval evaluation and result-bound relevance review command."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


class _BoundedParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes invalid values. Operator arguments can
        # contain private question text or accidental credentials.
        self.exit(2, json.dumps({'status': 'failed', 'error_type': 'ArgumentError'}) + '\n')


_ERROR_TYPES = frozenset({'ValueError', 'RuntimeError', 'OSError', 'FileNotFoundError',
    'PermissionError', 'ImportError', 'ModuleNotFoundError', 'TypeError', 'MemoryError',
    'EvaluationError'})


def _worker_command(config_path, status_path):
    # Bind the worker to this imported package, including in a source checkout.
    # -I ignores environment import paths; no dependency executable is selected.
    bootstrap = ('import runpy,sys;sys.path.insert(0,sys.argv.pop(1));'
                 'runpy.run_module("nvidia_rag.evaluation_worker",run_name="__main__")')
    return [sys.executable, '-I', '-B', '-c', bootstrap,
            str(Path(__file__).resolve().parent.parent), str(config_path), str(status_path)]


def _status_from_worker(options):
    with tempfile.TemporaryDirectory(prefix='rag-evaluation-command-') as temporary:
        folder = Path(temporary)
        config_path, status_path = folder / 'command.json', folder / 'status.json'
        config_path.write_text(json.dumps({key: str(value) if isinstance(value, Path) else value
            for key, value in options.items()}, sort_keys=True, allow_nan=False), encoding='utf-8')
        result = subprocess.run(_worker_command(config_path, status_path),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with status_path.open('rb') as stream:
            body = stream.read(1025)
        if len(body) > 1024:
            raise ValueError('Invalid worker status')
        status = json.loads(body)
        if not isinstance(status, dict):
            raise ValueError('Invalid worker status')
        if result.returncode == 0:
            key = 'receipt_sha256' if options['command'] == 'run' else 'review_import_sha256'
            if (set(status) != {'status', key} or status['status'] != 'complete'
                or not isinstance(status[key], str) or re.fullmatch('[0-9a-f]{64}', status[key]) is None):
                raise ValueError('Invalid worker status')
            artifact = (options['output_dir'] / 'receipt.json' if options['command'] == 'run'
                        else options['output_path'])
            if sha256(artifact.read_bytes()).hexdigest() != status[key]:
                raise ValueError('Worker artifact changed')
        elif (result.returncode != 1 or set(status) != {'status', 'error_type'}
              or status['status'] != 'failed' or not isinstance(status['error_type'], str)
              or status['error_type'] not in _ERROR_TYPES):
            raise ValueError('Invalid worker status')
        return status


def _parser():
    parser = _BoundedParser(description="Frozen local retrieval evaluation; no answer generation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a new explicit retrieval run")
    for name in ("artifact-dir", "source-registry", "questions", "output-dir"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--mode", choices=("development", "acceptance-candidate"), required=True)
    run.add_argument("--backend", choices=("lexical", "cached-bge"), required=True)
    run.add_argument("--top-k", type=int, required=True)
    run.add_argument("--threshold", type=float, required=True)
    for name in ("reference", "vector-dir", "model-cache-dir", "statement-source-dir", "statement-fiscal-declarations"):
        run.add_argument("--" + name, type=Path)
    run.add_argument("--document-id", action="append", dest="document_ids")
    run.add_argument("--routing-policy", choices=("none", "source-bound-period"), default="none")
    review = commands.add_parser("import-review", help="validate supplied relevance judgments against a completed run")
    for name in ("run-dir", "review-path", "output-path"):
        review.add_argument("--" + name, type=Path, required=True)
    return parser


def main(argv=None):
    """Return bounded terminal status; never echo queries or exception text."""
    options = vars(_parser().parse_args(argv))
    try:
        status = _status_from_worker(options)
    except Exception:
        status = {'status': 'failed', 'error_type': 'WorkerError'}
    failed = status['status'] != 'complete'
    print(json.dumps(status, sort_keys=True, separators=(',', ':')),
          file=sys.stderr if failed else sys.stdout)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
