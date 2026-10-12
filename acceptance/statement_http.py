"""Installed API, native synthetic PDFs and real loopback source selection.

Run with an isolated Python from a fresh API-only wheel environment. All PDFs
and amounts are generated transport fixtures, never financial reference facts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.util import find_spec
from importlib import metadata
import json
from pathlib import Path
import secrets
import socket
import sys
import sysconfig
import tempfile
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.flags.isolated and sys.prefix != sys.base_prefix, 'Use a fresh venv and Python -I'
    assert args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists()
    for name in ('torch', 'gradio', 'transformers', 'sentence_transformers'):
        assert find_spec(name) is None, 'The API-only environment must not include ' + name
    import httpx
    import pymupdf
    import uvicorn
    import nvidia_rag.api as api
    import nvidia_rag.statement_api as statement_api
    import nvidia_rag.structured_income as income
    from nvidia_rag.core import build_grounded_context
    installed_root = Path(sysconfig.get_paths()['purelib']).resolve()
    for module in (api, statement_api, income):
        assert Path(module.__file__).resolve().is_relative_to(installed_root)

    def generated_report(directory, *, annual):
        pdf = pymupdf.open()
        pdf.new_page()
        page = pdf.new_page()
        def text(x, y, value):
            page.insert_text((x, y), value, fontsize=8)
        text(175, 110, income.ANNUAL_TITLE if annual else income.TITLE)
        text(205, 130, income.UNIT)
        glyphs = [300, 400, 500] if annual else [240, 330, 420, 510]
        rights = [390, 490, 590] if annual else [315, 405, 495, 585]
        if annual:
            text(420, 150, 'Year Ended')
            dates = ['Jan 26, 2025', 'Jan 28, 2024', 'Jan 29, 2023']
        else:
            text(290, 150, 'Three Months Ended')
            text(470, 150, 'Six Months Ended')
            dates = ['Jul 28, 2024', 'Jul 30, 2023'] * 2
        for x, date_text in zip(glyphs, dates):
            text(x + 15, 180, date_text)
        def row(label, y, values, *, dollars=False):
            text(20, y, label)
            for x, right, value in zip(glyphs, rights, values):
                if dollars:
                    text(x, y, '$')
                text(right - pymupdf.get_text_length(value, fontsize=8), y, value)
        row('Revenue', 210, ['14,907', '12,501', '10,777', '8,501'], dollars=True)
        row('Net income', 250, ['9,207', '8,031', '7,024'] if annual else
            ['4,461', '3,502', '8,004', '6,913'], dollars=True)
        text(20, 300, income.EPS)
        row('Basic', 330, ['1.37', '1.08', '0.91', '0.77'], dollars=True)
        row('Diluted', 350, ['1.29', '1.01', '0.89', '0.75'], dollars=True)
        text(20, 400, income.SHARES)
        row('Basic', 430, ['6,701', '6,612', '6,521', '6,501'])
        row('Diluted', 450, ['7,137', '7,087', '7,001', '6,981'])
        text(100, 500, income.ANNUAL_FOOTER if annual else income.FOOTER)
        path = (directory / ('annual.pdf' if annual else 'quarter.pdf')).resolve()
        pdf.save(path)
        pdf.close()
        return income.ReportInput('synthetic-annual' if annual else 'synthetic-quarter',
            path, sha256(path.read_bytes()).hexdigest(), 2025, None if annual else 2)

    class SyntheticQueryService:
        def __init__(self):
            self.calls = []
        def query(self, question):
            self.calls.append(question)
            return build_grounded_context(question, evidence=[])

    checks = []
    def checked(name, condition):
        assert condition, name
        checks.append(name)

    with tempfile.TemporaryDirectory(prefix='statement-http-') as temp:
        reports = tuple(generated_report(Path(temp), annual=value) for value in (True, False))
        index = income.build_income_index(reports)
        checked('native-synthetic-index', len(index.cells) == 42 and not index.omissions)
        snapshot = asdict(index)
        snapshot.pop('_construction_token')
        expected_sha = sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode() + b'\n').hexdigest()
        token = secrets.token_urlsafe(32)  # Ephemeral; never printed or persisted.
        service = SyntheticQueryService()
        app = api.create_app(service=service, api_token=token, statement_index=index, expose_docs=True)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False,
            proxy_headers=False, server_header=False))
        worker = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            checked('real-loopback-listener-started', server.started and worker.is_alive())
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5,
                              follow_redirects=False, trust_env=False) as client:
                path = '/v1/statement-candidates'
                checked('unauthenticated-malformed-body-401', client.post(path, content=b'not-json').status_code == 401)
                headers = {'Authorization': 'Bearer ' + token, 'X-Request-ID': 'installed-statement-http'}
                annual_question = 'What was net income for the year ended January 26, 2025, in millions of dollars?'
                quarterly_question = 'What was net income for the three months ended July 28, 2024, in millions of dollars?'
                for question, report, amount, kind, months in (
                    (annual_question, reports[0], '9207', 'fiscal_year', None),
                    (quarterly_question, reports[1], '4461', 'months', 3),
                ):
                    response = client.post(path, headers=headers,
                        json={'question': question, 'document_ids': [report.document_id]})
                    body = response.json()
                    checked(kind + '-real-http-200', response.status_code == 200)
                    checked(kind + '-candidate', body['disposition'] == 'source_candidate' and len(body['cells']) == 1)
                    cell = body['cells'][0]
                    checked(kind + '-exact-native-fields', cell['amount'] == amount and
                        cell['pdf_sha256'] == report.pdf_sha256 and cell['physical_page'] == 2 and
                        cell['period_kind'] == kind and cell['duration_months'] == months)
                    checked(kind + '-native-bindings', dict(cell['bindings'])['value']['literal'] in ('9,207', '4,461'))
                    source_cells = [source for source in index.cells
                                    if source.cell_id == cell['cell_id']]
                    checked(kind + '-unique-source-identity', len(source_cells) == 1)
                    expected_cell = {**asdict(source_cells[0]), 'period_kind': kind}
                    checked(kind + '-complete-source-cell-transport',
                            cell == json.loads(json.dumps(expected_cell, allow_nan=False)))
                    with pymupdf.open(report.pdf_path) as native_pdf:
                        native_text = native_pdf[cell['physical_page'] - 1].get_text('text', sort=True)
                    checked(kind + '-native-text-span-replay',
                            sha256(native_text.encode()).hexdigest() == cell['page_text_sha256'] and
                            all(native_text[span['start']:span['end']] == span['literal']
                                for _, span in cell['bindings']))
                    checked(kind + '-identity-and-verdict', body['index_sha256'] == expected_sha and
                        body['financial_correctness'] == 'unestablished' and
                        body['request_id'] == response.headers['X-Request-ID'] == 'installed-statement-http')
                for change, expected in (
                    (annual_question.replace('January 26', 'January 27'), 'refuse'),
                    (annual_question.replace('year ended', 'annual three months ended'), 'clarify'),
                    (annual_question.replace('dollars', 'USD'), 'unsupported'),
                ):
                    response = client.post(path, headers=headers,
                        json={'question': change, 'document_ids': [reports[0].document_id]})
                    checked(expected + '-not-a-service-failure', response.status_code == 200 and
                        response.json()['disposition'] == expected and response.json()['cells'] == [])
                duplicate = client.post(path, headers={**headers, 'Content-Type': 'application/json'},
                    content=b'{"question":"first","question":"second"}')
                checked('duplicate-JSON-safe-422', duplicate.status_code == 422 and
                    duplicate.json() == {'detail': 'Invalid statement request.'})
                oversized = client.post(path, headers={**headers, 'Content-Type': 'application/json'},
                    content=b'{"question":"' + b'x' * 8200 + b'"}')
                checked('actual-body-ceiling-413', oversized.status_code == 413)
                checked('source-mode-does-not-call-dense', service.calls == [])
                old_query = client.post('/v1/query', headers=headers, json={'question': 'ordinary query'})
                checked('existing-query-service-preserved', old_query.status_code == 200 and
                    old_query.json()['answer'] == '' and service.calls == ['ordinary query'])
                openapi = client.get('/openapi.json').json()
                checked('source-request-OpenAPI', 'requestBody' in openapi['paths'][path]['post'])
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            listener.close()
            checked('temporary-server-stopped', not worker.is_alive())

    receipt = {'schema_version': 1, 'status': 'passed', 'checks': checks,
        'python': sys.version.split()[0], 'distribution': metadata.version('nvidia-financial-rag'),
        'package_root': str(installed_root), 'isolated_import': True,
        'source_index_sha256': expected_sha, 'source_cells': 42,
        'production_module_sha256': {module.__name__: sha256(Path(module.__file__).read_bytes()).hexdigest()
                                    for module in (api, statement_api, income)},
        'fixture_scope': 'generated-native-PDF-table-and-real-loopback-HTTP;no-original-financial-reports',
        'encoder_runs': 0, 'generation_runs': 0, 'human_gold_count': 0,
        'financial_correctness': 'unestablished',
        'not_accepted': ['financial accuracy', 'public API hosting', 'load or TLS', 'browser integration']}
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write('\n')
    assert json.loads(args.output.read_bytes()) == receipt
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'source_cells': 42,
                      'scope': receipt['fixture_scope']}))


if __name__ == '__main__':
    main()
