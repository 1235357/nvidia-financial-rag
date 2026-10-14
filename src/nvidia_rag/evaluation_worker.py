"""Private command worker; its parent owns the standard output handles."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


def execute(options):
    """Execute explicit parsed options without emitting a terminal record."""
    options = dict(options)
    command = options.pop('command')
    paths = {'artifact_dir', 'source_registry', 'questions', 'output_dir', 'reference',
             'vector_dir', 'model_cache_dir', 'statement_source_dir',
             'statement_fiscal_declarations', 'run_dir', 'review_path', 'output_path'}
    for key in paths & options.keys():
        if options[key] is not None:
            options[key] = Path(options[key])
    if command == 'run':
        from .evaluation_run import run_evaluation
        if options['document_ids'] is not None:
            options['document_ids'] = frozenset(options['document_ids'])
        run_evaluation(**options)
        return {'status': 'complete', 'receipt_sha256':
                sha256((options['output_dir'] / 'receipt.json').read_bytes()).hexdigest()}
    if command != 'import-review':
        raise ValueError('Unknown worker command')
    from .evaluation_review import import_review
    import_review(**options)
    return {'status': 'complete', 'review_import_sha256':
            sha256(options['output_path'].read_bytes()).hexdigest()}


def main(config_path, status_path):
    try:
        with Path(config_path).open('rb') as stream:
            body = stream.read(1_048_577)
        if len(body) > 1_048_576:
            raise ValueError('Command configuration exceeds bound')
        status = execute(json.loads(body))
        code = 0
    except Exception as error:
        from .evaluation_cli import _ERROR_TYPES
        name = type(error).__name__
        status = {'status': 'failed', 'error_type': name if name in _ERROR_TYPES else 'EvaluationError'}
        code = 1
    with Path(status_path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(status, sort_keys=True, separators=(',', ':')) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
