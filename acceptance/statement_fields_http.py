"""Installed-wheel typed statement selection over owned real loopback HTTP.

Run a fresh API-only venv's Python with -I -B and an absolute --wheel/--output.
The independently authored native PDFs are synthetic transport fixtures. They
provide no original-report financial gold, model execution or public hosting.
Only this process's child listeners are launched and stopped.
"""
from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import asdict, fields
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata, util
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile


ROUTE = '/v1/statement-fields'
SCORE_SEMANTICS = 'exact-source-field-selection;no-cosine-score'
PRIVATE = 'acceptance-private-source-path-and-candidate-value'
REPORTS = (
    {'document_id': 'typed-annual-a', 'fiscal_year': 2031, 'fiscal_quarter': None,
     'dates': ['Jan 26, 2031', 'Jan 27, 2030', 'Jan 28, 2029'],
     'values': [
         ['26,413', '19,857', '17,409'], ['(3,817)', '1,683', '(909)'],
         ['(0.47)', '0.22', '0.14'], ['(0.43)', '0.19', '0.12'],
         ['8,173', '7,649', '7,221'], ['8,871', '8,043', '7,849']]},
    {'document_id': 'typed-annual-b', 'fiscal_year': 2030, 'fiscal_quarter': None,
     'dates': ['Jan 27, 2030', 'Jan 28, 2029', 'Jan 30, 2028'],
     'values': [
         ['31,681', '24,319', '20,753'], ['2,917', '(1,379)', '883'],
         ['0.36', '(0.17)', '0.11'], ['0.33', '(0.15)', '0.10'],
         ['8,459', '8,107', '7,733'], ['9,071', '8,791', '8,479']]},
    {'document_id': 'typed-quarter', 'fiscal_year': 2031, 'fiscal_quarter': 2,
     'dates': ['Oct 27, 2030', 'Oct 28, 2029'] * 2,
     'values': [
         ['36,749', '28,411', '66,173', '51,927'],
         ['(5,279)', '4,113', '(9,683)', '7,891'],
         ['(0.63)', '0.41', '(1.16)', '0.97'],
         ['(0.59)', '0.38', '(1.08)', '0.91'],
         ['9,431', '9,217', '9,309', '9,104'],
         ['9,917', '9,667', '9,801', '9,579']]},
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def record_rows(text):
    rows = list(csv.reader(io.StringIO(text)))
    require(all(len(row) == 3 for row in rows) and len(rows) == len({row[0] for row in rows}),
            'invalid or duplicate RECORD rows')
    return {row[0]: row[1:] for row in rows}


def installed_identity(wheel):
    import nvidia_rag
    require(sys.flags.isolated and sys.flags.dont_write_bytecode and sys.prefix != sys.base_prefix,
            'Use a fresh venv Python with -I -B')
    require(wheel.is_absolute() and wheel.is_file(), 'wheel must be absolute and exist')
    package = Path(nvidia_rag.__file__).resolve().parent
    purelib = Path(sysconfig.get_paths()['purelib']).resolve()
    checkout = Path(__file__).resolve().parents[1]
    require(package.parent == purelib and not package.is_relative_to(checkout),
            'production imports must be installed outside checkout source')
    for name in ('torch', 'transformers', 'sentence_transformers', 'gradio'):
        require(util.find_spec(name) is None, 'unexpected heavyweight dependency: ' + name)
    distribution = metadata.distribution('nvidia-financial-rag')
    installed_record = distribution.read_text('RECORD')
    require(installed_record is not None, 'installed RECORD missing')
    installed_rows = record_rows(installed_record)
    hashes, module_paths, record_checks = {}, {}, []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate wheel archive members')
        records = [name for name in names if name.endswith('.dist-info/RECORD')]
        require(len(records) == 1, 'wheel RECORD must be unique')
        wheel_rows = record_rows(archive.read(records[0]).decode('utf-8'))
        modules = {name for name in names if name.startswith('nvidia_rag/') and name.endswith('.py')}
        actual = {'nvidia_rag/' + str(path.relative_to(package)).replace('\\', '/')
                  for path in package.rglob('*.py')}
        require(modules == actual and {'nvidia_rag/statement_fields.py',
                'nvidia_rag/statement_fields_api.py'} <= modules, 'installed/wheel production module set differs')
        for name in sorted(modules):
            raw = archive.read(name)
            path = package / name.removeprefix('nvidia_rag/')
            require(path.resolve().is_relative_to(package) and path.read_bytes() == raw,
                    'installed module bytes differ from wheel: ' + name)
            encoded = base64.urlsafe_b64encode(sha256(raw).digest()).rstrip(b'=').decode('ascii')
            expected = ['sha256=' + encoded, str(len(raw))]
            require(wheel_rows.get(name) == expected and installed_rows.get(name) == expected,
                    'wheel/installed RECORD mismatch: ' + name)
            hashes[name], module_paths[name] = sha256(raw).hexdigest(), str(path)
            record_checks.append(name)
    return {'distribution': distribution.version, 'purelib': str(purelib),
            'wheel_sha256': digest(wheel), 'wheel_bytes': wheel.stat().st_size,
            'module_sha256': hashes, 'module_paths': module_paths,
            'record_verified_python_modules': record_checks, 'isolated_import': True,
            'heavyweight_dependencies_absent': True}


def generated_pdf(path, definition):
    """Fixed-profile geometry; all dates and amounts authored for this acceptance."""
    import pymupdf
    annual = definition['fiscal_quarter'] is None
    with pymupdf.open() as pdf:
        pdf.new_page()
        page = pdf.new_page()
        def text(x, y, value):
            page.insert_text((x, y), value, fontsize=8)
        text(175, 110, 'Consolidated Statements of Income' if annual else
             'Condensed Consolidated Statements of Income')
        text(205, 130, '(In millions, except per share data)')
        glyphs, rights = ([300, 400, 500], [390, 490, 590]) if annual else (
            [240, 330, 420, 510], [315, 405, 495, 585])
        if annual:
            text(420, 150, 'Year Ended')
        else:
            text(290, 150, 'Three Months Ended')
            text(470, 150, 'Six Months Ended')
        for x, date in zip(glyphs, definition['dates'], strict=True):
            text(x + 15, 180, date)
        rows = [('Revenue', 210), ('Net income', 250), ('Basic', 330),
                ('Diluted', 350), ('Basic', 430), ('Diluted', 450)]
        text(20, 300, 'Net income per share:')
        text(20, 400, 'Weighted average shares used in per share computation:')
        for row_number, ((label, y), values) in enumerate(zip(rows, definition['values'], strict=True)):
            text(20, y, label)
            for x, right, value in zip(glyphs, rights, values, strict=True):
                if row_number < 4:
                    text(x, y, '$')
                text(right - pymupdf.get_text_length(value, fontsize=8), y, value)
        text(100, 500, 'See accompanying Notes to the Consolidated Financial Statements.' if annual else
             'See accompanying Notes to Condensed Consolidated Financial Statements.')
        pdf.save(path)


def fixture(root):
    from nvidia_rag.structured_income import ReportInput, build_income_index
    root.mkdir()
    reports, identities = [], []
    for definition in REPORTS:
        path = root / (definition['document_id'] + '.pdf')
        generated_pdf(path, definition)
        reports.append(ReportInput(definition['document_id'], path, digest(path),
                                   definition['fiscal_year'], definition['fiscal_quarter']))
        identities.append({'document_id': definition['document_id'], 'pdf_sha256': digest(path),
                           'fiscal_year': definition['fiscal_year'], 'fiscal_quarter': definition['fiscal_quarter']})
    index = build_income_index(tuple(reports))
    require(len(index.cells) == 60 and not index.omissions, 'independent native fixture must contain 60 cells without omissions')
    snapshot = asdict(index)
    snapshot.pop('_construction_token')
    return {'reports': identities, 'source_index_sha256': sha256(canonical(snapshot)).hexdigest(),
            'source_cells': {cell.cell_id: json.loads(json.dumps({**asdict(cell), 'period_kind': cell.period_kind}))
                             for cell in index.cells}}


def verify_cell(cell, *, document_id, row_path, column, literal, amount, fixture_root, proof):
    """Replay complete DTO, source identity, native character spans and word geometry."""
    import pymupdf
    from nvidia_rag.structured_income import IncomeCell
    require(set(cell) == {field.name for field in fields(IncomeCell)} | {'period_kind'}, 'source DTO fields differ')
    require(cell == proof['source_cells'].get(cell['cell_id']), 'complete source-cell transport differs')
    require(cell['document_id'] == document_id and cell['row_path'] == row_path and
            cell['column_number'] == column and cell['value_literal'] == literal and cell['amount'] == amount,
            'independently specified source value differs')
    path = fixture_root / (document_id + '.pdf')
    require(digest(path) == cell['pdf_sha256'] and cell['physical_page'] == 2, 'PDF or physical-page identity differs')
    with pymupdf.open(path) as pdf:
        page = pdf[cell['physical_page'] - 1]
        require(not pdf[0].get_text().strip() and cell['pdf_page_label'] == (page.get_label() or None),
                'native PDF page identity differs')
        text, words = page.get_text('text', sort=True), page.get_text('words', sort=True)
    require(sha256(text.encode()).hexdigest() == cell['page_text_sha256'], 'native page text digest differs')
    require(cell['pymupdf_version'] == pymupdf.version[0], 'native parser version differs')
    bindings = cell['bindings']
    expected_bindings = {'statement', 'unit', 'row', 'date', 'duration', 'value'}
    if cell['unit_kind'] != 'monetary':
        expected_bindings.add('section')
    if cell['unit_kind'] != 'count':
        expected_bindings.add('currency')
    require(len(bindings) == len(expected_bindings) and {pair[0] for pair in bindings} == expected_bindings,
            'native required bindings differ')
    for name, span in bindings:
        require(type(span['start']) is int and type(span['end']) is int and
                0 <= span['start'] < span['end'] <= len(text) and
                text[span['start']:span['end']] == span['literal'], 'native source character span differs')
        box = span['bbox']
        require(len(box) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in box),
                'source geometry invalid')
        contained = [word for word in words if word[0] >= box[0] - .001 and word[1] >= box[1] - .001 and
                     word[2] <= box[2] + .001 and word[3] <= box[3] + .001]
        require(contained and ' '.join(word[4] for word in contained) == ' '.join(span['literal'].split()),
                'native source literal geometry differs')
        envelope = (min(w[0] for w in contained), min(w[1] for w in contained),
                    max(w[2] for w in contained), max(w[3] for w in contained))
        require(all(abs(a-b) <= .001 for a, b in zip(box, envelope, strict=True)), 'source bbox envelope differs')
    identity = {key: value for key, value in cell.items() if key not in {'cell_id', 'period_kind'}}
    require(cell['cell_id'] == sha256(canonical(identity)[:-1]).hexdigest(), 'source cell identity differs')
    return {'cell_id': cell['cell_id'], 'document_id': document_id, 'row_path': row_path,
            'amount': amount, 'value_literal': literal, 'native_spans': len(bindings), 'geometry_replayed': True}


def request(**changes):
    return {'row_path': 'monetary/net_income', 'period_kind': 'fiscal_year', 'period_end': '2031-01-26',
            'duration_months': None, 'unit_kind': 'monetary', 'scale': '1000000',
            'currency_literal': '$', **changes}


def child_main(args):
    """Outside-checkout subprocess helper; changes affect this child memory only."""
    import logging
    import pymupdf
    import uvicorn
    from fastapi import HTTPException
    from nvidia_rag import api, statement_api, statement_fields_api, structured_income
    from nvidia_rag.core import build_grounded_context
    token = os.environ.pop('NVIDIA_RAG_ACCEPTANCE_TOKEN')
    run_id = os.environ.pop('NVIDIA_RAG_ACCEPTANCE_RUN_ID')
    require(len(token) >= 32, 'child token absent')
    purelib = Path(sysconfig.get_paths()['purelib']).resolve()
    for module in (api, statement_api, statement_fields_api, structured_income):
        require(Path(module.__file__).resolve().is_relative_to(purelib), 'child import outside installed purelib')
    report_rows = json.loads(args.child_manifest.read_bytes())
    index = None
    if args.child_mode != 'unconfigured':
        reports = tuple(structured_income.ReportInput(row['document_id'], args.child_sources / (row['document_id'] + '.pdf'),
            row['pdf_sha256'], row['fiscal_year'], row['fiscal_quarter']) for row in report_rows)
        index = structured_income.build_income_index(reports)
    state = {'query_calls': [], 'source_io_attempts': 0, 'literal_parse_attempts': 0,
             'startup_source_files_deleted': False}
    class Service:
        def query(self, question):
            state['query_calls'].append(question)
            return build_grounded_context(question, evidence=[])
    app = api.create_app(service=Service(), api_token=token, statement_index=index, expose_docs=True)
    if index is not None:
        for path in args.child_sources.glob('*.pdf'):
            path.unlink()
        state['startup_source_files_deleted'] = not any(args.child_sources.iterdir())
    def no_io(*unused_args, **unused_kwargs):
        state['source_io_attempts'] += 1
        raise RuntimeError(PRIVATE)
    def no_parse(*unused_args, **unused_kwargs):
        state['literal_parse_attempts'] += 1
        raise RuntimeError(PRIVATE)
    pymupdf.open = no_io
    structured_income.build_income_index = no_io
    structured_income.parse_statement_request = no_parse
    statement_api.parse_statement_request = no_parse
    if args.child_mode in {'selector-fault', 'selector-http-fault'}:
        def failure(*unused_args, **unused_kwargs):
            if args.child_mode == 'selector-http-fault':
                raise HTTPException(status_code=418, detail=PRIVATE)
            raise RuntimeError(PRIVATE)
        statement_fields_api.select_statement_fields = failure
    if args.child_mode == 'dto-fault':
        def invalid_source(*unused_args, **unused_kwargs):
            raise TypeError(PRIVATE)
        statement_fields_api.SourceCandidate.model_validate = invalid_source
    if args.child_mode == 'render-fault':
        original_response = statement_fields_api.JSONResponse
        class BrokenCandidateResponse(original_response):
            def render(self, content):
                if isinstance(content, dict) and content.get('cells'):
                    content = {**content, 'private': PRIVATE, 'nonfinite': float('nan')}
                return super().render(content)
        statement_fields_api.JSONResponse = BrokenCandidateResponse
    @app.get('/acceptance-state', include_in_schema=False)
    def acceptance_state():
        return state
    logging.getLogger('nvidia_rag.api').disabled = True
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        pending_ready = args.child_ready.with_suffix('.pending')
        write_json(pending_ready, {'port': port, 'pid': os.getpid(),
                                  'run_id': run_id, 'purelib': str(purelib)})
        pending_ready.replace(args.child_ready)
        server = uvicorn.Server(uvicorn.Config(app, log_level='critical', access_log=False,
            proxy_headers=False, server_header=False, lifespan='off'))
        server.run(sockets=[listener])


def stop_owned(process):
    termination_requested = False
    forced_kill_used = False
    windows_owned_tree_kill = False
    if process.poll() is None:
        termination_requested = True
        if os.name == 'nt':
            # The venv executable is a redirector. Its exact owned tree includes
            # the native interpreter; terminating only the launcher can leak it.
            stopped = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                capture_output=True, timeout=10, check=False)
            require(stopped.returncode == 0, 'owned Windows process tree did not stop')
            forced_kill_used = True
            windows_owned_tree_kill = True
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            forced_kill_used = True
            process.kill()
            process.wait(timeout=5)
    require(process.poll() is not None, 'owned server subprocess remains alive')
    return {'process_returncode': process.returncode,
            'termination_requested': termination_requested, 'forced_kill_used': forced_kill_used,
            'windows_owned_tree_kill': windows_owned_tree_kill}


def listener_closed(port):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=.2):
                pass
        except OSError:
            return
        time.sleep(.05)
    raise ValueError('owned listener remains reachable; unrelated processes are never stopped')


def launch(mode, *, workspace, fixture_root, proof):
    import httpx
    root = workspace / mode
    root.mkdir()
    sources = root / 'startup-pdfs'
    sources.mkdir()
    for path in fixture_root.glob('*.pdf'):
        shutil.copyfile(path, sources / path.name)
    manifest, ready = root / 'report-manifest.json', root / 'listener.json'
    write_json(manifest, proof['reports'])
    helper = root / 'acceptance_child.py'
    shutil.copyfile(Path(__file__), helper)
    token = secrets.token_urlsafe(32)
    run_id = secrets.token_hex(16)
    command = [sys.executable, '-I', '-B', str(helper), '--child-mode', mode,
               '--child-sources', str(sources), '--child-manifest', str(manifest), '--child-ready', str(ready)]
    environment = dict(os.environ, NVIDIA_RAG_ACCEPTANCE_TOKEN=token, NVIDIA_RAG_ACCEPTANCE_RUN_ID=run_id)
    environment.pop('NVIDIA_RAG_MODEL_TOKEN', None)
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    port = None
    termination = None
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            require(process.poll() is None, 'owned helper exited before binding')
            require(time.monotonic() < deadline, 'owned helper binding deadline expired')
            time.sleep(.05)
        info = json.loads(ready.read_bytes())
        port = info['port']
        require(info['run_id'] == run_id and type(info['pid']) is int and info['pid'] > 0 and
                info['purelib'] == str(Path(sysconfig.get_paths()['purelib']).resolve()),
                'owned listener child attestation differs')
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5,
                          follow_redirects=False, trust_env=False) as client:
            while True:
                try:
                    if client.get('/healthz').status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                require(process.poll() is None and time.monotonic() < deadline, 'owned helper startup deadline expired')
                time.sleep(.05)
            records = http_checks(client, token, mode=mode, fixture_root=fixture_root, proof=proof)
            require(process.poll() is None, 'owned listener exited during checks')
    except Exception as error:
        termination = stop_owned(process)
        diagnostics = process.stderr.read(4096).decode('utf-8', errors='replace') if process.stderr else ''
        raise ValueError(str(error) + ('; child diagnostics: ' + diagnostics if diagnostics else '')) from error
    finally:
        if termination is None:
            termination = stop_owned(process)
        if process.stderr:
            process.stderr.close()
        if port is not None:
            listener_closed(port)
    return {'mode': mode, 'launcher_pid': process.pid, 'listener_process_pid': info['pid'],
            'cwd_outside_checkout': True,
            'owned_process_stopped': True, 'owned_listener_closed': True,
            **termination, 'checks': records}


def http_checks(client, token, *, mode, fixture_root, proof):
    checks = []
    deadline = time.monotonic() + 60
    headers = {'Authorization': 'Bearer ' + token, 'X-Request-ID': 'installed-typed-fields'}
    def checked(name, condition):
        require(time.monotonic() < deadline, mode + ': bounded HTTP check deadline expired')
        require(condition, mode + ': ' + name)
        checks.append({'check': name})
    def send(payload):
        return client.post(ROUTE, headers=headers, json=payload)
    checked('unauthenticated-malformed-body-first-401', client.post(ROUTE, content=b'not-json').status_code == 401)
    checked('wrong-token-oversized-body-first-401', client.post(ROUTE,
        headers={'Authorization': 'Bearer ' + 'x' * 40}, content=b'x' * 9000).status_code == 401)
    if mode != 'configured':
        response = send(request())
        body = response.json()
        checked('safe-unavailable-503', response.status_code == 503 and body['disposition'] == 'unavailable' and
            body['cells'] == [] and body['selection_request'] is None and body['selection_request_sha256'] is None and
            bool(body['error_code']) and PRIVATE not in response.text and 'parsed_request' not in body)
        checked('failure-index-identity', body['index_sha256'] == (
            None if mode == 'unconfigured' else proof['source_index_sha256']))
        checked('failure-quality-and-score', body['financial_correctness'] == 'unestablished' and
            body['score_semantics'] == SCORE_SEMANTICS and body['request_id'] == response.headers['X-Request-ID'])
        if mode == 'unconfigured':
            absent = client.post(ROUTE, headers=headers, content=b'not-json')
            checked('missing-index-before-body-503', absent.status_code == 503 and absent.json()['cells'] == [])
        state = client.get('/acceptance-state').json()
        checked('fault-no-source-io-language-parse-or-query-fallback', state['query_calls'] == [] and
                state['source_io_attempts'] == state['literal_parse_attempts'] == 0)
        return checks
    def completed(payload, disposition, label):
        response = send(payload)
        checked(label + '-real-http-200', response.status_code == 200)
        body = response.json()
        checked(label + '-disposition', body['disposition'] == disposition and
                len(body['cells']) == (1 if disposition == 'source_candidate' else 0))
        expected_echo = {'document_ids': None, 'fiscal_year': None,
                         'fiscal_quarter': None, 'column_role': None, **payload}
        checked(label + '-exact-canonical-request-echo', body['selection_request'] == expected_echo)
        checked(label + '-canonical-request-digest', body['selection_request_sha256'] ==
                sha256(canonical(body['selection_request'])).hexdigest())
        checked(label + '-source-index-and-quality', body['index_sha256'] == proof['source_index_sha256'] and
                body['financial_correctness'] == 'unestablished' and body['score_semantics'] == SCORE_SEMANTICS)
        checked(label + '-request-id', body['request_id'] == response.headers['X-Request-ID'] == headers['X-Request-ID'])
        forbidden = {'parsed_request', 'question', 'mentions', 'answer', 'evidence', 'score', 'cosine_score', 'generation'}
        checked(label + '-no-invented-language-or-rag-output', not forbidden.intersection(body) and
                not forbidden.intersection(body['selection_request']))
        return body
    cases = (
        ('annual-negative-money', request(document_ids=['typed-annual-a']), 'typed-annual-a', 'monetary/net_income', 1, '(3,817)', '-3817'),
        ('annual-negative-per-share', request(document_ids=['typed-annual-a'], row_path='per_share/basic',
            unit_kind='per_share', scale='1'), 'typed-annual-a', 'per_share/basic', 1, '(0.47)', '-0.47'),
        ('annual-count-null-currency', request(document_ids=['typed-annual-a'], row_path='count/basic',
            unit_kind='count', currency_literal=None), 'typed-annual-a', 'count/basic', 1, '8,173', '8173'),
        ('quarter-negative-per-share', request(document_ids=['typed-quarter'], row_path='per_share/diluted',
            period_kind='months', period_end='2030-10-27', duration_months=6, unit_kind='per_share', scale='1'),
            'typed-quarter', 'per_share/diluted', 3, '(1.08)', '-1.08'),
        ('quarter-prior-count', request(document_ids=['typed-quarter'], row_path='count/diluted', period_kind='months',
            period_end='2029-10-28', duration_months=3, unit_kind='count', currency_literal=None, column_role='prior_comparison'),
            'typed-quarter', 'count/diluted', 2, '9,667', '9667'),
        ('annual-prior-role', request(period_end='2030-01-27', column_role='prior_comparison'),
            'typed-annual-a', 'monetary/net_income', 2, '1,683', '1683'),
        ('annual-current-role', request(period_end='2030-01-27', column_role='current'),
            'typed-annual-b', 'monetary/net_income', 1, '2,917', '2917'),
    )
    for label, payload, document_id, row_path, column, literal, amount in cases:
        body = completed(payload, 'source_candidate', label)
        replay = verify_cell(body['cells'][0], document_id=document_id, row_path=row_path,
            column=column, literal=literal, amount=amount, fixture_root=fixture_root, proof=proof)
        checks.append({'check': label + '-complete-source-native-replay', **replay})
    all_scope = completed(request(), 'source_candidate', 'resolved-all-scope')
    explicit = completed(request(document_ids=['typed-annual-a']), 'source_candidate', 'explicit-scope')
    checked('explicit-and-resolved-all-request-identities-differ',
            all_scope['selection_request_sha256'] != explicit['selection_request_sha256'])
    explicit_defaults = completed(request(document_ids=None, fiscal_year=None, fiscal_quarter=None,
        column_role=None), 'source_candidate', 'explicit-null-defaults')
    checked('omitted-and-explicit-null-optionals-have-same-canonical-digest',
            all_scope['selection_request_sha256'] == explicit_defaults['selection_request_sha256'])
    reversed_keys = completed(dict(reversed(list(request().items()))), 'source_candidate', 'reordered-input-members')
    checked('input-JSON-member-order-does-not-change-canonical-digest',
            all_scope['selection_request_sha256'] == reversed_keys['selection_request_sha256'])
    for label, changes, disposition in (
        ('ambiguous-current-prior-sources', {'period_end': '2030-01-27'}, 'clarify'),
        ('ambiguous-two-prior-sources', {'period_end': '2029-01-28', 'column_role': 'prior_comparison'}, 'clarify'),
        ('unknown-scope-refuses-known-source-too', {'document_ids': ['typed-annual-a', 'unknown-source']}, 'refuse'),
        ('report-scope-conflict', {'document_ids': ['typed-annual-a'], 'fiscal_year': 2030}, 'clarify'),
        ('annual-quarter-contract-conflict', {'document_ids': ['typed-quarter'], 'fiscal_quarter': 1}, 'clarify'),
        ('wrong-quarter-report-scope', {'document_ids': ['typed-quarter'], 'fiscal_quarter': 1,
            'period_kind': 'months', 'period_end': '2030-10-27', 'duration_months': 3}, 'refuse'),
        ('unknown-report-fiscal-year', {'fiscal_year': 2032}, 'refuse'),
        ('unknown-row', {'row_path': 'monetary/no_such_source_row'}, 'refuse'),
        ('unprinted-date', {'period_end': '2031-01-27'}, 'refuse'),
        ('unit-contract-conflict', {'unit_kind': 'count'}, 'unsupported'),
        ('scale-conversion-withheld', {'scale': '1'}, 'unsupported'),
        ('ISO-currency-conversion-withheld', {'currency_literal': 'USD'}, 'unsupported'),
        ('count-currency-withheld', {'row_path': 'count/basic', 'unit_kind': 'count'}, 'unsupported'),
        ('per-share-source-millions-withheld', {'row_path': 'per_share/basic', 'unit_kind': 'per_share'}, 'unsupported'),
    ):
        completed(request(**changes), disposition, label)
    valid = request(document_ids=['typed-annual-a'])
    invalid = [
        ('question-extra', {**valid, 'question': PRIVATE}),
        ('strict-bool-year', {**valid, 'fiscal_year': True}),
        ('strict-string-year', {**valid, 'fiscal_year': '2031'}),
        ('strict-integer-scale', {**valid, 'scale': 1000000}),
        ('strict-string-duration', {**valid, 'period_kind': 'months', 'duration_months': '3'}),
        ('strict-bool-duration', {**valid, 'period_kind': 'months', 'duration_months': True}),
        ('noncanonical-date', {**valid, 'period_end': '2031-1-26'}),
        ('invalid-calendar-date', {**valid, 'period_end': '2031-02-29'}),
        ('annual-duration-conflict', {**valid, 'duration_months': 12}),
        ('months-null-duration', {**valid, 'period_kind': 'months'}),
        ('months-duration-outside-bound', {**valid, 'period_kind': 'months', 'duration_months': 13}),
        ('duplicate-document-scope', {**valid, 'document_ids': ['typed-annual-a'] * 2}),
        ('empty-document-scope', {**valid, 'document_ids': []}),
        ('source-path-extra', {**valid, 'pdf_path': PRIVATE}),
        ('source-amount-extra', {**valid, 'amount': PRIVATE}),
    ]
    for key in ('duration_months', 'currency_literal', 'row_path', 'period_kind', 'period_end', 'unit_kind', 'scale'):
        invalid.append(('required-field-' + key, {name: value for name, value in valid.items() if name != key}))
    for label, payload in invalid:
        response = send(payload)
        checked(label + '-generic-422-withholding', response.status_code == 422 and
                response.json() == {'detail': 'Invalid statement field request.'} and PRIVATE not in response.text)
    raw_headers = {**headers, 'Content-Type': 'application/json'}
    for label, raw in (
        ('duplicate-key', b'{"row_path":"a","row_path":"b"}'),
        ('escaped-duplicate-key', b'{"row_path":"a","row_\\u0070ath":"b"}'),
        ('nonfinite-constant', b'{"fiscal_year":NaN}'),
        ('invalid-utf8', b'{"row_path":"\xff"}'),
        ('not-an-object', b'[]'),
        ('too-deep-JSON', b'[' * 1100 + b']' * 1100),
    ):
        response = client.post(ROUTE, headers=raw_headers, content=raw)
        checked(label + '-generic-422', response.status_code == 422 and
                response.json() == {'detail': 'Invalid statement field request.'})
    checked('actual-body-byte-ceiling-413', client.post(ROUTE, headers=raw_headers, content=b' ' * 8193).status_code == 413)
    checked('exact-body-byte-ceiling-is-not-overflow', client.post(ROUTE, headers=raw_headers,
            content=b'{}' + b' ' * 8190).status_code == 422)
    checked('actual-chunked-body-byte-ceiling-413', client.post(ROUTE, headers=raw_headers,
            content=iter([b' ' * 4096, b' ' * 4097])).status_code == 413)
    checked('wrong-media-type-415', client.post(ROUTE, headers=headers, content=b'{}').status_code == 415)
    state = client.get('/acceptance-state').json()
    checked('startup-PDF-copies-deleted-before-requests', state['startup_source_files_deleted'])
    checked('no-source-io-language-parser-or-dense-fallback', state['query_calls'] == [] and
            state['source_io_attempts'] == state['literal_parse_attempts'] == 0)
    query = client.post('/v1/query', headers=headers, json={'question': 'explicit ordinary query'})
    checked('ordinary-query-service-preserved', query.status_code == 200 and query.json()['answer'] == '' and
            client.get('/acceptance-state').json()['query_calls'] == ['explicit ordinary query'])
    openapi = client.get('/openapi.json').json()
    checked('new-and-existing-route-OpenAPI', {ROUTE, '/v1/query', '/v1/statement-candidates'} <= set(openapi['paths']) and
            'requestBody' in openapi['paths'][ROUTE]['post'])
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child-mode', choices=('configured', 'unconfigured', 'selector-fault', 'selector-http-fault', 'dto-fault', 'render-fault'), help=argparse.SUPPRESS)
    parser.add_argument('--child-sources', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--child-manifest', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--child-ready', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_mode:
        child_main(args)
        return
    try:
        require(args.wheel is not None and args.output is not None, '--wheel and --output are required')
        require(args.output.is_absolute() and args.output.parent.is_dir() and not os.path.lexists(args.output),
                'output must be absolute, parent must exist, and file must not exist')
        identity = installed_identity(args.wheel)
        with tempfile.TemporaryDirectory(prefix='installed-typed-statement-fields-') as temp:
            workspace = Path(temp).resolve()
            require(not workspace.is_relative_to(Path(__file__).resolve().parents[1]), 'child cwd overlaps checkout')
            fixture_root = workspace / 'native-verification-pdfs'
            proof = fixture(fixture_root)
            before = {path.name: digest(path) for path in fixture_root.glob('*.pdf')}
            transports = [launch(mode, workspace=workspace, fixture_root=fixture_root, proof=proof) for mode in
                          ('configured', 'unconfigured', 'selector-fault', 'selector-http-fault', 'dto-fault', 'render-fault')]
            after = {path.name: digest(path) for path in fixture_root.glob('*.pdf')}
            require(before == after and installed_identity(args.wheel) == identity, 'verified inputs/installed modules changed')
            receipt = {'schema_version': 1, 'status': 'passed', 'created_at_utc': datetime.now(UTC).isoformat(),
                'python': sys.version.split()[0], 'acceptance_script_sha256': digest(Path(__file__)),
                'installed_identity': identity, 'synthetic_PDF_sha256': before, 'source_cells': 60,
                'source_index_sha256': proof['source_index_sha256'], 'verification_inputs_unchanged': True,
                'transports': transports, 'scope': 'installed wheel; independent native synthetic PDFs; owned real loopback HTTP',
                'fixture_scope': 'synthetic exact-field transport and source-provenance contracts; no original financial reports',
                'encoder_runs': 0, 'generator_runs': 0, 'tokenizer_runs': 0, 'human_gold_count': 0,
                'financial_correctness': 'unestablished', 'not_accepted': ['financial answer accuracy',
                    'held-out natural-language retrieval', 'public hosting', 'TLS', 'load', 'browser integration', 'model execution']}
            write_json(args.output, receipt)
            require(json.loads(args.output.read_bytes()) == receipt, 'saved receipt differs')
            print(json.dumps({'status': 'passed', 'checks': sum(len(row['checks']) for row in transports),
                              'transport_modes': len(transports), 'production_modules': len(identity['module_sha256']),
                              'source_cells': 60, 'scope': receipt['scope']}))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
