"""Source-built report routing; deterministic vectors are not financial gold."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import importlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from nvidia_rag.core import Chunk
from nvidia_rag.ingest import load_chunks
from nvidia_rag.retrieval import DenseRetriever, FiscalPeriodRoutingRetriever
from nvidia_rag.runtime import BGE_QUERY_INSTRUCTION, build_bge_service
from nvidia_rag.sources import SourceRecord
from nvidia_rag.structured_income import IncomeIndex, ReportInput, build_income_index
from test_statement_startup import Inputs, write_json
from test_annual_income import annual_pdf


def routing():
    assert importlib.util.find_spec('nvidia_rag.period_routing') is not None, \
        'source-bound report catalog and period routing are missing'
    return importlib.import_module('nvidia_rag.period_routing')


@pytest.fixture
def configured(tmp_path):
    inputs = Inputs(tmp_path)
    manifest, chunks_path = inputs.corpus_files()
    snapshot = inputs.load(corpus_manifest=manifest, chunks_path=chunks_path)
    return inputs, manifest, chunks_path, snapshot


def catalog(configured):
    _, manifest, chunks_path, snapshot = configured
    return routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)


def decision(configured, question):
    return routing().decide_period_scope(catalog(configured), question)


@pytest.mark.parametrize('question', [
    "According to NVIDIA's fiscal 2025 second-quarter report, what was net income for the three-month period ended July 28, 2024?",
    'What was net income in FY2025 Q2 for the six months ended 2024-07-28?',
    'FY2025 SECOND QUARTER net income for 3 months ended Jul 28, 2024',
    'FY2025Q2 net income for three months ended July 28, 2024',
    'FY2025 Q2. Net income for three months ended July 28, 2024',
    'FY2025 Q2. 3 months ended July 28, 2024 income',
    'FY2025 Q2 revenue for 3-month period ended July 28, 2024',
])
def test_explicit_quarter_selects_source_report_before_dense_ranking(configured, question):
    inputs, _, chunks_path, _ = configured
    chunks = load_chunks(chunks_path)
    seen = []
    class Model:
        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return [[1., 0.] for _ in texts]
    assert 'report_catalog' in inspect.signature(build_bge_service).parameters, \
        'configured BGE service cannot receive the bound report catalog'
    service = build_bge_service(chunks=chunks, model=Model(), threshold=.78, top_k=5,
        report_catalog=catalog(configured), passage_embeddings=[[1., 0.], [.9, .1]])
    trace = service.inspect(question)
    assert [c.chunk.document_id for c in trace.retrieval.candidates] == [inputs.reports[1].document_id]
    assert trace.retrieval.routed_document_id == inputs.reports[1].document_id
    assert trace.retrieval.route_disposition == 'selected'
    assert trace.retrieval.route_reason == 'EXPLICIT_SOURCE_REPORT'
    assert trace.retrieval.considered_chunks == 1
    assert trace.retrieval.considered_scores == pytest.approx((.9 / np.hypot(.9, .1),))
    assert trace.retrieval.query_vector == (1., 0.)
    assert trace.context.can_answer and trace.threshold == .78 and trace.top_k == 5
    assert seen == [BGE_QUERY_INSTRUCTION + question]


@pytest.mark.parametrize('question,document', [
    ('FY2025 annual report revenue for the year ended January 26, 2025', 0),
    ('FY2025 Q2 net income for the three months ended July 30, 2023', 1),
    ('Fiscal year 2025 second-quarter report six-month income ended 2023-07-30', 1),
    ('Three months ended July 28, 2024 net income', 1),
])
def test_valid_report_and_comparison_column_scopes(configured, question, document):
    scope = decision(configured, question)
    assert scope.disposition == 'selected'
    assert scope.document_ids == frozenset({configured[0].reports[document].document_id})


def test_date_only_column_shared_by_distinct_reports_requires_clarification(configured):
    inputs, _, _, _ = configured
    extra = replace(annual_pdf(inputs.source_dir, split_date=False),
                    document_id='opaque-next-year-annual', fiscal_year=2026)
    inputs.reports += (extra,)
    inputs.rows.append(dict(document_id=extra.document_id, filename=extra.pdf_path.name,
        sha256=extra.pdf_sha256, label='Generated different annual report',
        url='https://s201.q4cdn.com/141608511/files/test/'+extra.pdf_path.name))
    inputs.payload['reports'].append(dict(document_id=extra.document_id, pdf_sha256=extra.pdf_sha256,
        fiscal_year=2026, fiscal_quarter=None, reviewed_by='Synthetic fixture author',
        source_locator='Native table with a shared comparison date on physical page 2'))
    inputs.sync()
    manifest, chunks_path = inputs.corpus_files()
    inputs.manifest_value.update(document_count=3, chunk_count=3)
    write_json(manifest, inputs.manifest_value)
    snapshot = inputs.load(corpus_manifest=manifest, chunks_path=chunks_path)
    bound = routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)
    scope = routing().decide_period_scope(bound, 'Annual income for year ended January 26, 2025')
    assert (scope.disposition, scope.reason, scope.document_ids) == ('clarify', 'AMBIGUOUS_REPORT_SCOPE', frozenset())
    assert routing().decide_period_scope(bound, 'FY2026 annual year ended January 26, 2025').document_ids == frozenset({extra.document_id})


def test_generic_quarter_cannot_degrade_to_an_annual_date_only_scope(configured):
    question = 'Quarter ended January 26, 2025 net income'
    bound = catalog(configured)
    scope = routing().decide_period_scope(bound, question)
    assert (scope.disposition, scope.reason, scope.document_ids) == ('refuse', 'UNSUPPORTED_PERIOD_SCOPE', frozenset())
    dense = DenseRetriever(load_chunks(configured[2]), query_encoder=lambda texts: [[1., 0.]],
        precomputed_embeddings=[[1., 0.], [.8, .6]])
    trace = routing().SourceBoundPeriodRetriever(dense, report_catalog=bound).inspect(question, top_k=5)
    assert trace.candidates == () and trace.considered_chunks == 0
    assert trace.routed_document_ids == () and trace.route_disposition == 'refuse'


@pytest.mark.parametrize('question,disposition,report_indices', [
    ('Question: What was NVIDIA annual revenue in FY2025?', 'selected', (0,)),
    ('Quality of evidence supporting the company discussion', 'global', (0, 1)),
])
def test_ordinary_q_prefixed_prose_is_not_a_malformed_quarter(configured, question, disposition, report_indices):
    bound = catalog(configured)
    expected = frozenset(configured[0].reports[index].document_id for index in report_indices)
    scope = routing().decide_period_scope(bound, question)
    assert scope.disposition == disposition and scope.document_ids == expected
    dense = DenseRetriever(load_chunks(configured[2]), query_encoder=lambda texts: [[1., 0.]],
        precomputed_embeddings=[[1., 0.], [.8, .6]])
    trace = routing().SourceBoundPeriodRetriever(dense, report_catalog=bound).inspect(question, top_k=5)
    assert trace.considered_chunks == len(expected)
    assert {candidate.chunk.document_id for candidate in trace.candidates} == expected


def test_source_date_and_duration_are_retained_as_pairs_not_cartesian_product(configured):
    inputs, manifest, chunks_path, snapshot = configured
    # A generated Q2 table has a distinct prior comparison date in the six-month
    # column. Both dates and durations exist; the cross-pair does not.
    quarter = inputs.reports[1]
    pdf = pymupdf.open(stream=quarter.pdf_path.read_bytes(), filetype='pdf')
    page = pdf[1]
    box = page.search_for('Jul 30, 2023')[-1]
    page.add_redact_annot(box, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_text((box.x0, 185), 'Apr 30, 2023', fontsize=8)
    body = pdf.tobytes()
    pdf.close()
    quarter.pdf_path.write_bytes(body)
    quarter = replace(quarter, pdf_sha256=sha256(body).hexdigest())
    inputs.reports = (inputs.reports[0], quarter)
    inputs.rows[1].update(document_id=quarter.document_id, sha256=quarter.pdf_sha256)
    inputs.payload['reports'][1].update(document_id=quarter.document_id,
        pdf_sha256=quarter.pdf_sha256)
    inputs.sync()
    manifest, chunks_path = inputs.corpus_files()
    snapshot = inputs.load(corpus_manifest=manifest, chunks_path=chunks_path)
    bound = routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)
    assert not snapshot.index.omissions
    assert routing().decide_period_scope(bound, 'FY2025 Q2 six months ended April 30, 2023').disposition == 'selected'
    scope = routing().decide_period_scope(bound, 'FY2025 Q2 three months ended April 30, 2023')
    assert scope.disposition == 'refuse' and scope.reason == 'UNAVAILABLE_SOURCE_COLUMN'


@pytest.mark.parametrize('question,disposition,reason', [
    ('FY2025 net income', 'clarify', 'AMBIGUOUS_REPORT_SCOPE'),
    ('FY2030 Q2 net income', 'refuse', 'UNKNOWN_REPORT_SCOPE'),
    ('FY2025 Q4 net income', 'refuse', 'UNKNOWN_REPORT_SCOPE'),
    ('FY2025 annual Q2 net income', 'clarify', 'CONFLICTING_PERIOD_SCOPE'),
    ('FY2025 Q2 income for year ended July 28, 2024', 'clarify', 'CONFLICTING_PERIOD_SCOPE'),
    ('FY2025 annual income for six months ended January 26, 2025', 'clarify', 'CONFLICTING_PERIOD_SCOPE'),
    ('Compare FY2024 and FY2025 income', 'clarify', 'MULTIPLE_REPORT_SCOPE'),
    ('FY2025 Q1 and Q2 income', 'clarify', 'MULTIPLE_REPORT_SCOPE'),
    ('FY2025 Q2 for three months and six months ended July 28, 2024', 'clarify', 'CONFLICTING_PERIOD_SCOPE'),
    ('FY2025 Q2 ended July 28, 2024 and July 30, 2023', 'clarify', 'CONFLICTING_PERIOD_SCOPE'),
    ('FY2025 Q2 income for three months ended April 28, 2024', 'refuse', 'UNAVAILABLE_SOURCE_COLUMN'),
    ('FY2025 Q2 income for nine months ended July 28, 2024', 'refuse', 'UNAVAILABLE_SOURCE_COLUMN'),
    ('FY2025 Q2 income ended February 30, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 income ended 2024-02-30', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 income ended Smarch 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q5 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2.5 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 income.-3 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 income.+3 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 income.− 3 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 income.-2nd quarter', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 income.+2nd quarter', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 income.− 2nd quarter', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3./6 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3/6. months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 1e-3 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3e0/6 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3. months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2e0 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('Q.2 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 .2 quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for .3 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3.0/6 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3/.6 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for -.3 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2/-3 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2.0/3 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('Q-2 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q+2 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q−2 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2/3 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 / 3 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2/Q3 net income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 -2nd quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 2.3 quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 2/3 quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 1st/2nd quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 1st / 2 quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for -3 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for +3 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for −3 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 3/6 months ended July 28, 2024 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY1899 Q2 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2100 Q2 income', 'refuse', 'UNKNOWN_REPORT_SCOPE'),
    ('FY2025 Q0 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 fifth-quarter income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 for 9 months ended 2024-07-28 income', 'refuse', 'UNAVAILABLE_SOURCE_COLUMN'),
    ('FY2025 Q2 for seven months ended July 28, 2024', 'refuse', 'UNAVAILABLE_SOURCE_COLUMN'),
    ('FY2025 Q2 for 3.5 months ended July 28, 2024', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 3rd quarter income', 'refuse', 'UNKNOWN_REPORT_SCOPE'),
    ('FYabc Q2 income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Qfoo income', 'refuse', 'INVALID_PERIOD_LITERAL'),
    ('FY2025 Q2 income for calendar year 2024', 'refuse', 'UNSUPPORTED_PERIOD_SCOPE'),
    ('calendar Q2 2024 income', 'refuse', 'UNSUPPORTED_PERIOD_SCOPE'),
    ('Compare FY2025 Q2 and 2024 Q2 income', 'clarify', 'MULTIPLE_REPORT_SCOPE'),
])
def test_unknown_conflicting_and_ambiguous_requests_never_fall_back(configured, question, disposition, reason):
    scope = decision(configured, question)
    assert (scope.disposition, scope.reason, scope.document_ids) == (disposition, reason, frozenset())
    dense = DenseRetriever(load_chunks(configured[2]), query_encoder=lambda texts: [[1., 0.]],
        precomputed_embeddings=[[1., 0.], [1., 0.]])
    trace = routing().SourceBoundPeriodRetriever(dense, report_catalog=catalog(configured)).inspect(question, top_k=5)
    assert trace.candidates == () and trace.considered_chunks == 0
    assert trace.considered_scores == () and trace.query_vector == (1., 0.)
    assert trace.route_disposition == disposition and trace.route_reason == reason
    assert trace.routed_document_ids == ()


def test_unrestricted_query_retains_global_arithmetic_and_raw_text(configured):
    question = 'What risks did NVIDIA describe?'
    scope = decision(configured, question)
    assert scope.disposition == 'global' and scope.reason == 'NO_LITERAL_PERIOD_SCOPE'
    captured = []
    def encode(texts):
        captured.extend(texts)
        return [[3., 4.]]
    dense = DenseRetriever(load_chunks(configured[2]), query_encoder=encode,
        precomputed_embeddings=[[3., 4.], [4., 3.]])
    trace = routing().SourceBoundPeriodRetriever(dense, report_catalog=catalog(configured)).inspect(question, top_k=5)
    assert trace.considered_chunks == 2 and trace.considered_scores == pytest.approx((1., .96))
    assert captured == [question]


def test_unrestricted_query_in_selected_catalog_never_leaks_omitted_document(configured):
    inputs, manifest, chunks_path, _ = configured
    snapshot = inputs.load(document_ids=frozenset({inputs.reports[1].document_id}),
        corpus_manifest=manifest, chunks_path=chunks_path)
    bound = routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)
    dense = DenseRetriever(load_chunks(chunks_path), query_encoder=lambda texts: [[1., 0.]],
        precomputed_embeddings=[[1., 0.], [.8, .6]])
    trace = routing().SourceBoundPeriodRetriever(dense, report_catalog=bound).inspect('What risks?', top_k=5)
    assert trace.route_disposition == 'global'
    assert trace.routed_document_ids == (inputs.reports[1].document_id,)
    assert trace.considered_chunks == 1
    assert trace.considered_scores == pytest.approx((.8,))
    assert [c.chunk.document_id for c in trace.candidates] == [inputs.reports[1].document_id]
    assert routing().decide_period_scope(bound, 'FY2025 annual income').disposition == 'refuse'


def test_catalog_is_immutable_and_uses_source_declarations_not_names(configured):
    bound = catalog(configured)
    assert isinstance(bound.reports, tuple)
    assert all(isinstance(report.columns, frozenset) for report in bound.reports)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        bound.reports = ()
    inputs, manifest, chunks_path, _ = configured
    annual, quarter = inputs.reports
    inputs.reports = (replace(annual, document_id='opaque-a'), replace(quarter, document_id='opaque-b'))
    for old, new in zip(inputs.rows, inputs.reports):
        old['document_id'] = new.document_id
    for old, new in zip(inputs.payload['reports'], inputs.reports):
        old['document_id'] = new.document_id
    inputs.sync()
    manifest, chunks_path = inputs.corpus_files()
    snapshot = inputs.load(corpus_manifest=manifest, chunks_path=chunks_path)
    bound = routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)
    assert routing().decide_period_scope(bound, 'FY2025 Q2 net income').document_ids == frozenset({'opaque-b'})


def test_catalog_rejects_unowned_index(configured):
    inputs, manifest, chunks_path, snapshot = configured
    unowned = IncomeIndex(snapshot.index.reports, snapshot.index.cells, snapshot.index.omissions)
    with pytest.raises(ValueError, match='SELECTION|source-built'):
        routing().build_report_catalog(replace(snapshot, index=unowned), corpus_manifest=manifest, chunks_path=chunks_path)


def test_catalog_rejects_duplicate_report_identity(configured):
    inputs, manifest, chunks_path, snapshot = configured
    annual, quarter = inputs.reports
    conflicting = build_income_index((annual, replace(quarter, fiscal_quarter=None)))
    with pytest.raises(ValueError, match='duplicate report identity'):
        routing().build_report_catalog(replace(snapshot, index=conflicting), corpus_manifest=manifest, chunks_path=chunks_path)


@pytest.mark.parametrize('change', ['missing-bound-input', 'missing-pdf-input', 'missing-declaration-input',
                                  'unregistered-report', 'wrong-pdf-hash', 'changed-corpus'])
def test_catalog_rejects_unbound_source_or_corpus(configured, change):
    inputs, manifest, chunks_path, snapshot = configured
    if change == 'missing-bound-input':
        snapshot = replace(snapshot, input_identities=tuple(i for i in snapshot.input_identities if i.path != manifest))
    elif change == 'missing-pdf-input':
        snapshot = replace(snapshot, input_identities=tuple(i for i in snapshot.input_identities if i.path != inputs.reports[1].pdf_path))
    elif change == 'missing-declaration-input':
        snapshot = replace(snapshot, input_identities=tuple(i for i in snapshot.input_identities if i.path != inputs.declarations))
    elif change == 'unregistered-report':
        snapshot = replace(snapshot, index=build_income_index((replace(inputs.reports[1], document_id='unknown'),)))
    elif change == 'wrong-pdf-hash':
        snapshot = replace(snapshot, sources=(replace(snapshot.sources[0], sha256='0'*64), snapshot.sources[1]))
    else:
        manifest.write_bytes(manifest.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)


def test_known_layout_omission_does_not_fabricate_column(configured):
    inputs, manifest, chunks_path, snapshot = configured
    # Annual extraction from a quarter PDF is unsupported, but its explicit
    # report declaration remains retained without invented financial columns.
    inputs.payload['reports'][1]['fiscal_year'] = 2026
    inputs.payload['reports'][1]['fiscal_quarter'] = None
    inputs.sync()
    snapshot = inputs.load(document_ids=frozenset({inputs.reports[1].document_id}),
        corpus_manifest=manifest, chunks_path=chunks_path)
    assert snapshot.index.omissions and not snapshot.index.cells
    bound = routing().build_report_catalog(snapshot, corpus_manifest=manifest, chunks_path=chunks_path)
    assert bound.reports[0].columns == frozenset()
    assert routing().decide_period_scope(bound, 'FY2026 annual risks').disposition == 'selected'
    scope = routing().decide_period_scope(bound, 'FY2026 annual year ended July 28, 2024 income')
    assert scope.disposition == 'refuse' and scope.reason == 'UNAVAILABLE_SOURCE_COLUMN'


def test_source_built_index_cannot_replace_retained_fiscal_declarations(configured):
    inputs, manifest, chunks_path, snapshot = configured
    substituted = build_income_index((replace(inputs.reports[1], fiscal_year=2026),))
    with pytest.raises(ValueError, match='declaration'):
        routing().build_report_catalog(replace(snapshot, index=substituted), corpus_manifest=manifest, chunks_path=chunks_path)


def test_invalid_routing_composition_refuses_before_passage_encoding(configured):
    class ForbiddenModel:
        def encode(self, *args, **kwargs):
            pytest.fail('incompatible source policies reached model encoding')
    assert 'report_catalog' in inspect.signature(build_bge_service).parameters
    with pytest.raises(ValueError, match='cannot be combined'):
        build_bge_service(chunks=load_chunks(configured[2]), model=ForbiddenModel(), threshold=.78, top_k=5,
            report_catalog=catalog(configured), source_task='reported_results', source_records=list(configured[3].sources))


def test_direct_retriever_cannot_rank_unbound_chunks_with_matching_document_ids(configured):
    original = load_chunks(configured[2])
    altered = [replace(chunk, text='different unbound source bytes') for chunk in original]
    dense = DenseRetriever(altered, query_encoder=lambda texts: pytest.fail('unbound ranking reached encoder'),
        precomputed_embeddings=[[1., 0.], [1., 0.]])
    with pytest.raises(ValueError, match='bound corpus chunks'):
        routing().SourceBoundPeriodRetriever(dense, report_catalog=catalog(configured))


@pytest.mark.parametrize('mode', ['unconfigured', 'annual-legacy', 'explicit-task'])
def test_existing_unconfigured_and_explicit_task_compositions_remain_available(configured, mode):
    chunks = load_chunks(configured[2])
    class Model:
        def encode(self, texts, **kwargs):
            return [[1., 0.] for _ in texts]
    options = {}
    expected = {chunk.document_id for chunk in chunks}
    if mode == 'annual-legacy':
        options['route_fiscal_year'] = True
        expected = {chunks[0].document_id}
    elif mode == 'explicit-task':
        options.update(source_task='reported_results', source_records=[
            SourceRecord(chunks[1].document_id, 'Generated scoped report',
                'https://s201.q4cdn.com/141608511/files/test/quarter.pdf', 'quarter.pdf', 'a'*64,
                provenance={'source_type':'official_filing', 'permission_status':'reviewed',
                            'permitted_operations':('index','display')})])
        expected = {chunks[1].document_id}
    service = build_bge_service(chunks=chunks, model=Model(), threshold=.78, top_k=5,
        passage_embeddings=[[1., 0.], [.9, .1]], **options)
    response = service.query('FY2025 net income')
    assert {candidate.chunk.document_id for candidate in response.evidence} == expected


def test_experimental_router_ordinal_and_comparison_pitfalls_are_not_accepted(configured):
    chunks = load_chunks(configured[2])
    annual, quarter = configured[0].reports
    dense = DenseRetriever(chunks, query_encoder=lambda texts: [[1., 0.]], precomputed_embeddings=[[1.,0.],[1.,0.]])
    experimental = FiscalPeriodRoutingRetriever(dense, annual_documents={2025: annual.document_id},
        quarterly_documents={(2025,2): quarter.document_id}, quarter_end_dates={'July 28, 2024': quarter.document_id})
    # These assertions preserve the isolated experiment's observed limitations.
    assert experimental.retrieve('FY2025 second-quarter net income', top_k=5) == []
    assert experimental.retrieve('FY2025 Q2 three months ended July 30, 2023 income', top_k=5) == []
