"""Owned, explicitly declared statement inputs for API startup.

Hashes establish local byte identity, not authenticity or financial correctness.
No downloads, fiscal inference, models or request-time source refresh occur here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from .sources import SourceRecord, file_sha256, load_source_manifest, require_source_operation
from .structured_income import IncomeIndex, ReportInput, build_income_index

_JSON_LIMIT = 1_048_576
_SOURCE_LIMIT = 256_000_000
_ID = re.compile(r'[A-Za-z0-9_.-]{1,128}\Z')
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')
_REVIEW_FIELDS = {'source_type', 'publisher', 'attribution', 'published_date',
    'event_date', 'financial_period', 'source_locator', 'permission_status',
    'permission_evidence', 'permitted_operations'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, label, *, limit=2048):
    _require(type(value) is str and 0 < len(value) <= limit
        and value == value.strip() and bool(value.strip()), f'{label} must be canonical nonblank text')


def _digest(value, label):
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None,
             f'{label} must be a lowercase SHA-256 digest')


def _keys(value, required, *, optional=(), label):
    _require(type(value) is dict and set(required) <= value.keys() <= set(required) | set(optional),
             f'{label} has missing or unsupported fields')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON member')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('non-finite JSON constant')


def _json(body):
    return json.loads(body.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)


def _safe_path(path, *, directory=False):
    _require(isinstance(path, Path) and path.is_absolute(), 'startup input must be an absolute local path')
    _require(not any(p.is_symlink() or (hasattr(p, 'is_junction') and p.is_junction())
        for p in (path, *path.parents)), 'startup input must not contain a symlink or junction')
    _require(path.is_dir() if directory else path.is_file(), 'startup input is not the expected local entry')
    return path


@dataclass(frozen=True, slots=True)
class InputIdentity:
    path: Path
    sha256: str
    bytes: int

    def verify_unchanged(self):
        _safe_path(self.path)
        _require(self.path.stat().st_size == self.bytes and file_sha256(self.path) == self.sha256,
                 'startup input changed after snapshot')


@dataclass(frozen=True, slots=True)
class StatementSnapshot:
    index: IncomeIndex
    sources: tuple[SourceRecord, ...]
    input_identities: tuple[InputIdentity, ...]

    def verify_unchanged(self):
        for identity in self.input_identities:
            identity.verify_unchanged()

    def verify_report_bindings(self):
        """Recheck the selected index against its retained fiscal declarations.

        The loader records registry then declaration identity before optional
        model inputs and PDFs. Replaying their strict metadata prevents replacing
        a source-built index with different FY/Q declarations after startup.
        It does not reparse PDF tables or establish financial correctness.
        """
        from .structured_income import validate_income_index
        validate_income_index(self.index)
        self.verify_unchanged()
        _require(len(self.input_identities) >= 2, 'statement declaration identities are missing')
        registry_identity, declaration_identity = self.input_identities[:2]
        registry_body, current_registry = _read(registry_identity.path)
        declaration_body, current_declaration = _read(declaration_identity.path)
        _require(current_registry == registry_identity and current_declaration == declaration_identity,
                 'statement declaration input identity changed')
        _require(_registry(registry_identity.path, registry_body) == self.sources,
                 'statement registry differs from retained source declaration')
        selected = _declarations(declaration_body, registry_hash=registry_identity.sha256,
            sources=self.sources, document_ids=frozenset(record[0] for record in self.index.reports))
        expected = tuple((source.document_id, source.sha256, year, quarter)
                         for source, year, quarter in selected)
        _require(self.index.reports == expected, 'source index differs from retained fiscal declaration')
        for source, _, _ in selected:
            pdf_inputs = [identity for identity in self.input_identities[2:]
                          if identity.path.name == source.filename and identity.sha256 == source.sha256]
            _require(len(pdf_inputs) == 1 and pdf_inputs[0].bytes > 0,
                     'selected report PDF input identity is missing or ambiguous')


def _read(path, *, limit=_JSON_LIMIT):
    _safe_path(path)
    _require(path.stat().st_size <= limit, 'startup input exceeds byte budget')
    with path.open('rb') as stream:
        body = stream.read(limit + 1)
    _require(len(body) <= limit, 'startup input exceeds byte budget')
    return body, InputIdentity(path, sha256(body).hexdigest(), len(body))


def _registry(path, body):
    raw = _json(body)
    _keys(raw, {'schema_version', 'sources'}, label='source registry')
    version = raw['schema_version']
    _require(type(version) is int and version in (1, 2), 'unsupported source registry schema')
    rows = raw['sources']
    _require(type(rows) is list and 0 < len(rows) <= 128, 'source registry report budget')
    for row in rows:
        _keys(row, {'document_id', 'label', 'url', 'filename', 'sha256'},
            optional={'reporting_period'} | (_REVIEW_FIELDS if version == 2 else set()), label='registered source')
        for field in ('document_id', 'label', 'url', 'filename'):
            _text(row[field], field)
        _require(_ID.fullmatch(row['document_id']) is not None, 'invalid registered document ID')
        _digest(row['sha256'], 'registered PDF sha256')
        filename = row['filename']
        _require(not any(c in filename for c in ('/', '\\', ':')) and filename not in ('.', '..')
            and filename.lower().endswith('.pdf'), 'unsafe registered PDF basename')
    sources = load_source_manifest(path)
    _require(path.read_bytes() == body, 'source registry changed during loading')
    for source, row in zip(sources, rows, strict=True):
        observed = dict(document_id=source.document_id, label=source.label, url=source.url,
                        filename=source.filename, sha256=source.sha256)
        if source.reporting_period is not None:
            observed['reporting_period'] = dict(source.reporting_period)
        if source.provenance is not None:
            observed.update(source.provenance)
            observed['permitted_operations'] = list(observed['permitted_operations'])
        _require(observed == row, 'registered source was normalized or changed')
    return sources


def _scope(document_ids, sources):
    if document_ids is None:
        return
    _require(type(document_ids) is frozenset and bool(document_ids), 'document scope must be a nonempty frozenset')
    _require(all(type(value) is str and _ID.fullmatch(value) is not None for value in document_ids),
             'invalid document scope ID')
    _require(document_ids <= {source.document_id for source in sources}, 'document scope is not registered')


def _declarations(body, *, registry_hash, sources, document_ids):
    raw = _json(body)
    _keys(raw, {'schema_version', 'source_registry_sha256', 'reports'}, label='fiscal declarations')
    _require(type(raw['schema_version']) is int and raw['schema_version'] == 1, 'unsupported fiscal declaration schema')
    _digest(raw['source_registry_sha256'], 'source_registry_sha256')
    _require(raw['source_registry_sha256'] == registry_hash, 'fiscal declarations registry hash mismatch')
    rows = raw['reports']
    _require(type(rows) is list and 0 < len(rows) <= 128, 'fiscal declaration report budget')
    by_id = {source.document_id: source for source in sources}
    seen = set()
    selected = []
    for row in rows:
        _keys(row, {'document_id', 'pdf_sha256', 'fiscal_year', 'fiscal_quarter', 'reviewed_by', 'source_locator'},
              label='fiscal report declaration')
        for field in ('document_id', 'reviewed_by', 'source_locator'):
            _text(row[field], field)
        doc_id = row['document_id']
        _require(doc_id in by_id and doc_id not in seen, 'unknown or duplicate declared report')
        seen.add(doc_id)
        _digest(row['pdf_sha256'], 'declared PDF sha256')
        source = by_id[doc_id]
        _require(row['pdf_sha256'] == source.sha256, 'declared PDF identity differs from registry')
        year, quarter = row['fiscal_year'], row['fiscal_quarter']
        _require(type(year) is int and 1900 <= year <= 9999, 'invalid declared fiscal year')
        _require(quarter is None or (type(quarter) is int and quarter in (1, 2, 3, 4)), 'invalid declared fiscal quarter')
        if source.reporting_period is not None:
            _require(year == source.reporting_period['fiscal_year'] and quarter == source.reporting_period['quarter'],
                     'fiscal declaration contradicts registered reporting period')
        if document_ids is None or doc_id in document_ids:
            require_source_operation(source, 'index')
            require_source_operation(source, 'display')
            if source.provenance is not None:
                _require(source.provenance['source_type'] == 'official_filing', 'statement source must be an official filing')
            selected.append((source, year, quarter))
    _require(bool(selected), 'document scope contains no declared statement reports')
    return selected


def _model_binding(manifest_path, chunks_path, sources):
    manifest_body, manifest_identity = _read(manifest_path)
    chunks_body, chunks_identity = _read(chunks_path, limit=_SOURCE_LIMIT)
    manifest = _json(manifest_body)
    _require(type(manifest) is dict and type(manifest.get('schema_version')) is int
        and manifest['schema_version'] == 1, 'unsupported model corpus manifest')
    _require(manifest.get('chunks_sha256') == chunks_identity.sha256, 'model corpus chunks hash mismatch')
    documents = manifest.get('documents')
    _require(type(documents) is list and len(documents) == len(sources), 'model corpus document count mismatch')
    declared = {}
    by_id = {source.document_id: source for source in sources}
    for row in documents:
        _require(type(row) is dict and row.get('document_id') in by_id
                 and row['document_id'] not in declared, 'model corpus document identity mismatch')
        source = by_id[row['document_id']]
        expected = dict(document_label=source.label, source_url=source.url,
                        source_filename=source.filename, source_sha256=source.sha256)
        _require(all(type(row.get(key)) is str and row[key] == value for key, value in expected.items()),
                 'model corpus source identity mismatch')
        _require(type(row.get('chunk_count')) is int and row['chunk_count'] > 0, 'invalid model corpus chunk count')
        declared[source.document_id] = row['chunk_count']
    counts = Counter()
    seen = set()
    for line in chunks_body.decode('utf-8').split('\n'):
        if not line.strip():
            continue
        row = _json(line.encode('utf-8'))
        _require(type(row) is dict and type(row.get('document_id')) is str
            and row['document_id'] in by_id and type(row.get('chunk_id')) is str
            and bool(row['chunk_id'].strip()) and row['chunk_id'] not in seen
            and type(row.get('page_number')) is int and row['page_number'] > 0
            and type(row.get('text')) is str and bool(row['text'].strip()), 'invalid model corpus chunk identity')
        seen.add(row['chunk_id'])
        counts[row['document_id']] += 1
    _require(dict(counts) == declared and type(manifest.get('document_count')) is int
        and manifest['document_count'] == len(sources) and type(manifest.get('chunk_count')) is int
        and manifest['chunk_count'] == sum(counts.values()), 'model corpus actual counts mismatch')
    return manifest_identity, chunks_identity


def load_statement_snapshot(*, source_registry: Path, source_dir: Path, fiscal_declarations: Path,
        document_ids: frozenset[str] | None = None, expected_sources: tuple[SourceRecord, ...] | None = None,
        corpus_manifest: Path | None = None, chunks_path: Path | None = None) -> StatementSnapshot:
    """Build a source-bound index from owned startup inputs; never infer fiscal identity.

The declaration file can cover more reports than the operator's selected scope.
Unselected PDFs are not read or indexed. Optional model artifacts must be paired
and exactly bind the full source registration, independently of selected reports.
"""
    _require((corpus_manifest is None) == (chunks_path is None), 'model corpus manifest and chunks must be paired')
    _safe_path(source_dir, directory=True)
    registry_body, registry_identity = _read(source_registry)
    sources = _registry(source_registry, registry_body)
    if expected_sources is not None:
        _require(type(expected_sources) is tuple and sources == expected_sources,
                 'statement registry differs from verified corpus source snapshot')
    _scope(document_ids, sources)
    declaration_body, declaration_identity = _read(fiscal_declarations)
    selected = _declarations(declaration_body, registry_hash=registry_identity.sha256,
                             sources=sources, document_ids=document_ids)
    identities = [registry_identity, declaration_identity]
    if corpus_manifest is not None:
        identities.extend(_model_binding(corpus_manifest, chunks_path, sources))
    reports = []
    for source, year, quarter in selected:
        path = source_dir / source.filename
        _safe_path(path)
        _require(path.resolve(strict=True).parent == source_dir.resolve(strict=True), 'registered PDF escapes source directory')
        size = path.stat().st_size
        _require(0 < size <= _SOURCE_LIMIT and file_sha256(path) == source.sha256, 'statement source PDF identity mismatch')
        identities.append(InputIdentity(path, source.sha256, size))
        reports.append(ReportInput(source.document_id, path, source.sha256, year, quarter))
    index = build_income_index(tuple(reports))
    snapshot = StatementSnapshot(index, sources, tuple(identities))
    snapshot.verify_unchanged()
    return snapshot
