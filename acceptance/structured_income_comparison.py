"""Build a source-only index, then compare the original frozen query/scope rows.

Expected targets enter only post-selection assessment. No encoder, model,
retriever or generator runs, and no synthetic cosine score is manufactured.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import os
from pathlib import Path

from acceptance.expanded_cases import load_case_bundle, load_experiment
from acceptance.expanded_source_binding_replay import (
    canonical_bytes, local_path, plain, read_body, read_completed_replay, snapshot,
    strict_json, write_replay_exclusive,
)
from nvidia_rag.sources import load_source_manifest
from nvidia_rag.structured_income import (
    ReportInput, build_income_index, parse_statement_request, select_statement_candidates,
)


def require(condition, code):
    if not condition:
        raise ValueError(code)


def write_completed(path, value, final_check):
    """Retain a failure guard when input/output closure fails after unlink."""
    require(not any(os.path.lexists(candidate) for candidate in
        (path, path.with_name(path.name+'.pending.json'), path.with_name(path.name+'.failure.json'))),
        'OUTPUT_MUST_BE_FRESH')
    expected_body = canonical_bytes(value) + b'\n'
    try:
        write_replay_exclusive(path, value, final_check)
        final_check()
        require(read_body(path, 16_000_000) == expected_body, 'OWNED_OUTPUT_CHANGED')
        require(read_completed_replay(path) == plain(value), 'COMPLETED_OUTPUT_CHANGED')
        final_check()
        require(read_body(path, 16_000_000) == expected_body
                and not any(os.path.lexists(path.with_name(path.name+'.'+kind+'.json'))
                            for kind in ('pending', 'failure')), 'OUTPUT_CLOSURE_CHANGED')
    except Exception:
        failure = path.with_name(path.name+'.failure.json')
        if not os.path.lexists(failure):
            try:
                with failure.open('xb') as handle:
                    handle.write(b'{"status":"failed","code":"COMPARISON_INCOMPLETE"}\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError:
                pass
        raise


def compare_source_candidate(case, selection):
    """Post-hoc exposed-target comparison, never a financial or human verdict."""
    if case.expected['outcome'] != 'evidence':
        return {'kind': 'provisional-negative-withholding',
                'withheld': selection.disposition != 'source_candidate', 'human_gold': False}
    expected_ids = set(case.expected['cell_ids'])
    expected = [cell for cell in case.source_cells if cell['cell_id'] in expected_ids]
    fields = ('document_id', 'pdf_sha256', 'physical_page', 'column_role', 'period_end',
              'duration_months', 'amount', 'unit_kind', 'scale', 'currency_literal')
    def target_key(cell):
        return tuple(cell[name] for name in fields) + (cell['row'],
                cell['report_fiscal_year'], cell['report_fiscal_quarter'])
    def selected_key(cell):
        return tuple(getattr(cell, name) for name in fields) + (cell.row_literal,
                cell.fiscal_year, cell.fiscal_quarter)
    actual = [selected_key(cell) for cell in selection.cells]
    wanted = [target_key(cell) for cell in expected]
    return {'kind': 'mechanical-exposed-source-target-comparison',
            'source_fields_match': bool(wanted) and sorted(actual) == sorted(wanted),
            'currency_basis': 'printed-literal-only;no-ISO-currency-verdict',
            'human_gold': False, 'financial_correctness': 'unestablished'}


def run_comparison(*, source_dir: Path, source_registry: Path, cases: Path,
                   case_freeze: Path, experiment: Path, output_dir: Path):
    local_path(source_dir, directory=True)
    local_path(output_dir.parent, directory=True)
    for path in (source_registry, cases, case_freeze, experiment):
        local_path(path)
    require(output_dir.is_absolute() and not output_dir.exists()
            and not output_dir.is_symlink(), 'OUTPUT_MUST_BE_FRESH')
    require(not output_dir.resolve().is_relative_to(source_dir.resolve())
            and not source_dir.resolve().is_relative_to(output_dir.resolve()), 'OUTPUT_SOURCE_OVERLAP')
    require(not any(path.resolve().is_relative_to(output_dir.resolve())
                    for path in (source_registry, cases, case_freeze, experiment)), 'OUTPUT_INPUT_OVERLAP')
    config = load_experiment(experiment)
    # Hash-only preflight does not expose annotated targets to the builder.
    require(sha256(read_body(cases, 16_000_000)).hexdigest() == config.data['input_sha256']['cases']
            and sha256(read_body(case_freeze, 512_000)).hexdigest() == config.data['input_sha256']['case_freeze'],
            'CASE_FREEZE_IDENTITY_MISMATCH')
    registry_body = read_body(source_registry)
    strict_json(registry_body)  # Reject duplicate keys before the legacy registry loader.
    require(sha256(registry_body).hexdigest() == config.data['input_sha256']['registry'],
            'REGISTRY_IDENTITY_MISMATCH')
    records = load_source_manifest(source_registry)
    configured = {row['document_id']: row for row in config.data['documents']}
    require(len(records) == len(configured) and {r.document_id for r in records} == set(configured),
            'CONFIGURED_SOURCE_SET_MISMATCH')
    reports = []
    for record in records:
        declared = configured[record.document_id]
        require(record.sha256 == declared['pdf_sha256'] and record.url == declared['issuer_url'],
                'DECLARED_SOURCE_IDENTITY_MISMATCH')
        reports.append(ReportInput(record.document_id, source_dir / record.filename,
            record.sha256, declared['report_fiscal_year'], declared['report_fiscal_quarter']))
    implementation = Path(__file__).resolve().parents[1] / 'src/nvidia_rag/structured_income.py'
    watched = {str(path): path for path in (source_registry, cases, case_freeze, experiment,
               implementation, Path(__file__).resolve(), *(r.pdf_path for r in reports))}
    before = snapshot(watched)
    validated_inputs = {
        str(experiment): config.sha256,
        str(source_registry): sha256(registry_body).hexdigest(),
        str(cases): config.data['input_sha256']['cases'],
        str(case_freeze): config.data['input_sha256']['case_freeze'],
        **{str(report.pdf_path): report.pdf_sha256 for report in reports},
    }
    require(all(before[name] == expected for name, expected in validated_inputs.items()),
            'INITIAL_INPUT_IDENTITY_MISMATCH')
    def unchanged():
        require(snapshot(watched) == before, 'COMPARISON_INPUT_CHANGED')
    output_dir.mkdir(exist_ok=False)
    # The builder cannot see cases, labels, queries or retrieval outputs.
    index = build_income_index(tuple(reports))
    index_data = asdict(index)
    index_data.pop('_construction_token')  # Ephemeral factory ownership never serializes.
    write_completed(output_dir / 'index.json', index_data, unchanged)
    index_body = read_body(output_dir / 'index.json', 16_000_000)
    require(index_body == canonical_bytes(index_data) + b'\n', 'OWNED_INDEX_CHANGED')
    index_sha = sha256(index_body).hexdigest()
    def final_check():
        unchanged()
        require(read_completed_replay(output_dir / 'index.json') == plain(index_data),
                'COMPLETED_INDEX_CHANGED')
        require(sha256(read_body(output_dir / 'index.json', 16_000_000)).hexdigest() == index_sha,
                'INDEX_OUTPUT_CHANGED')
    # Only now load case annotations. RetrievalInput excludes every target field.
    bundle = load_case_bundle(cases, case_freeze, config)
    inputs = bundle.retrieval_inputs()
    known_documents = tuple(row[0] for row in index.reports)
    selections, traces = [], []
    for item in inputs:
        original_scope = item.document_ids
        scope = known_documents if original_scope is None else tuple(original_scope)
        parsed = parse_statement_request(item.query, operator_document_ids=scope, index=index)
        selected = select_statement_candidates(index, parsed)
        selections.append(selected)
        traces.append({'case_id': item.case_id, 'scope_id': item.scope_id,
            'original_operator_document_ids': plain(original_scope),
            'query_sha256': parsed.query_sha256, 'parsed_request': asdict(parsed),
            'selection': asdict(selected)})
    # Annotated cases are consumed only after every selection has completed.
    by_id = {case.case_id: case for case in bundle.cases}
    for trace, selected in zip(traces, selections, strict=True):
        trace['posthoc_development_assessment'] = compare_source_candidate(by_id[trace['case_id']], selected)
    final_check()
    receipt = {'schema_version': 1, 'status': 'source-index-and-original-query-comparison',
        'created_at': datetime.now(timezone.utc).isoformat(), 'index_sha256': index_sha,
        'registry_sha256': sha256(registry_body).hexdigest(), 'cases_sha256': bundle.cases_sha256,
        'case_freeze_sha256': bundle.freeze_sha256, 'experiment_sha256': config.sha256,
        'implementation_sha256': before[str(implementation)],
        'runner_sha256': before[str(Path(__file__).resolve())],
        'registered_reports': len(index.reports), 'supported_statement_reports': len({c.document_id for c in index.cells}),
        'cells': len(index.cells), 'report_cell_counts': dict(Counter(c.document_id for c in index.cells)),
        'section_cell_counts': dict(Counter(c.unit_kind for c in index.cells)),
        'period_kind_counts': dict(Counter(c.period_kind for c in index.cells)),
        'omissions': [asdict(o) for o in index.omissions], 'ordered_requests': len(traces),
        'dispositions': dict(Counter(s.disposition for s in selections)),
        'human_gold_count': 0, 'encoder_runs': 0, 'generation_runs': 0,
        'financial_readiness': 'unestablished',
        'source_fiscal_metadata': 'operator-declaration;not-inferred-from-calendar-year',
        'scope_semantics': 'same-registered-document-scopes;source-bound-income-layouts;no-dense-fallback',
        'rows': traces}
    write_completed(output_dir / 'comparison.json', receipt, final_check)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'source-registry', 'cases', 'case-freeze', 'experiment', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    result = run_comparison(**vars(parser.parse_args()))
    print(canonical_bytes({key: value for key, value in result.items() if key != 'rows'}).decode())


if __name__ == '__main__':
    main()
