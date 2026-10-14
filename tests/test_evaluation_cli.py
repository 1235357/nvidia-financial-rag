"""Bounded installed command surface without implicit optional model loading."""
from __future__ import annotations

import importlib
import json
import sys
import subprocess
import os
from pathlib import Path

import pytest

from test_evaluation_run import case


def library():
    try:
        return importlib.import_module("nvidia_rag.evaluation_cli")
    except ModuleNotFoundError:
        pytest.fail("evaluation command interface is missing")


def test_help_does_not_import_sentence_transformers(capsys):
    module = library()
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    assert "import-review" in capsys.readouterr().out
    assert "sentence_transformers" not in sys.modules


def test_lexical_command_prints_bounded_identity_without_question(tmp_path, capsys):
    module = library()
    _, options = case(tmp_path)
    arguments = ["run"]
    for key in ("artifact_dir", "source_registry", "questions", "output_dir", "mode", "backend", "top_k", "threshold"):
        arguments.extend(["--" + key.replace("_", "-"), str(options[key])])
    assert module.main(arguments) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "complete" and len(status["receipt_sha256"]) == 64
    assert "Revenue" not in json.dumps(status)


def test_failure_prints_only_bounded_error_type(tmp_path, capsys):
    module = library()
    assert module.main(["run", "--artifact-dir", str(tmp_path), "--source-registry", str(tmp_path / "missing"),
        "--questions", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out"),
        "--mode", "development", "--backend", "lexical", "--top-k", "1", "--threshold", "0.78"]) == 1
    captured = capsys.readouterr()
    assert len(captured.err) < 200 and "Traceback" not in captured.err


@pytest.mark.parametrize('arguments', [
    ['private-synthetic-value-' * 1000],
    ['run', '--top-k', 'private-synthetic-value-' * 1000],
    ['import-review', '--unexpected', 'private-synthetic-value-' * 1000],
])
def test_parser_failures_are_bounded_without_echoing_argument_values(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        library().main(arguments)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert len(captured.err) < 200 and 'private-synthetic-value' not in captured.err
    assert json.loads(captured.err) == {'status': 'failed', 'error_type': 'ArgumentError'}


def test_evaluation_import_does_not_require_optional_http_generation_client():
    program = '''
import sys
class NoOptionalClients:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'httpx', 'torch', 'sentence_transformers', 'transformers'}:
            raise ImportError('Optional dependency forbidden in lexical evaluator')
sys.meta_path.insert(0, NoOptionalClients())
from nvidia_rag.evaluation_run import load_completed_run
from nvidia_rag.evaluation_review import import_review
'''
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-c', program], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('module_name', ['nvidia_rag', 'nvidia_rag.table_structure'])
def test_package_import_does_not_emit_legacy_pdf_alias_warning(module_name):
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-c', 'import ' + module_name], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''


@pytest.mark.parametrize('kind', ['python', 'descriptor', 'crt'])
@pytest.mark.parametrize('fail', [False, True])
def test_worker_output_cannot_corrupt_bounded_terminal_record(tmp_path, kind, fail):
    if kind == 'crt' and sys.platform != 'win32':
        pytest.skip('Windows mixed CRT output regression')
    program = '''
import os
from pathlib import Path
import sys
from nvidia_rag import evaluation_cli
child = """
import ctypes, hashlib, json, os, sys
from pathlib import Path
configuration = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
kind, fail = sys.argv[3], sys.argv[4] == 'True'
if kind == 'python':
    print('SYNTHETIC-DIAGNOSTIC-' * 500)
    print('SYNTHETIC-DIAGNOSTIC-' * 500, file=sys.stderr)
elif kind == 'descriptor':
    os.write(1, b'SYNTHETIC-DIAGNOSTIC-' * 500)
    os.write(2, b'SYNTHETIC-DIAGNOSTIC-' * 500)
else:
    crt = ctypes.CDLL('msvcrt')
    crt.printf(b'SYNTHETIC-BUFFERED-NATIVE-DIAGNOSTIC')
if fail:
    status = {'status': 'failed', 'error_type': 'RuntimeError'}
else:
    output = Path(configuration['output_dir'])
    output.mkdir()
    receipt = output / 'receipt.json'
    receipt.write_bytes(b'{}')
    status = {'status': 'complete', 'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
Path(sys.argv[2]).write_text(json.dumps(status), encoding='utf-8')
if kind == 'crt':
    crt.fflush(None)
raise SystemExit(int(fail))
"""
evaluation_cli._worker_command = lambda config, status: [sys.executable, '-I', '-B', '-c',
    child, str(config), str(status), sys.argv[2], sys.argv[3]]
raise SystemExit(evaluation_cli.main(['run', '--artifact-dir', sys.argv[1],
    '--source-registry', sys.argv[1], '--questions', sys.argv[1],
    '--output-dir', str(Path(sys.argv[1]) / 'out'), '--mode', 'development',
    '--backend', 'lexical', '--top-k', '1', '--threshold', '0.01']))
'''
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    result = subprocess.run([sys.executable, '-c', program, str(tmp_path), kind, str(fail)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == int(fail)
    assert len(result.stdout) < 200 and len(result.stderr) < 200
    assert 'SYNTHETIC-' not in result.stdout + result.stderr
    if fail:
        assert result.stdout == ''
        assert json.loads(result.stderr) == {'status': 'failed', 'error_type': 'RuntimeError'}
    else:
        assert result.stderr == ''
        assert json.loads(result.stdout)['status'] == 'complete'


@pytest.mark.parametrize('body,exit_code', [
    (None, 1), ('x' * 1025, 1), ('{}', 0),
    ('{"status":"failed","error_type":"unbounded"}', 1),
    ('{"status":"failed","error_type":"RuntimeError"}', 0),
    ('{"status":"complete","receipt_sha256":"' + '0' * 64 + '"}', 1),
])
def test_missing_or_invalid_worker_status_is_bounded(tmp_path, monkeypatch, capsys, body, exit_code):
    module = library()
    def command(config, status):
        script = 'import sys;from pathlib import Path;'
        if body is not None:
            script += 'Path(sys.argv[1]).write_text(sys.argv[2],encoding="utf-8");'
        script += 'raise SystemExit(int(sys.argv[3]))'
        return [sys.executable, '-I', '-B', '-c', script, str(status), body or '', str(exit_code)]
    monkeypatch.setattr(module, '_worker_command', command)
    assert module.main(['run', '--artifact-dir', str(tmp_path), '--source-registry', str(tmp_path),
        '--questions', str(tmp_path), '--output-dir', str(tmp_path / 'out'), '--mode', 'development',
        '--backend', 'lexical', '--top-k', '1', '--threshold', '0.01']) == 1
    result = capsys.readouterr()
    assert result.out == ''
    assert json.loads(result.err) == {'status': 'failed', 'error_type': 'WorkerError'}
