"""Source-only typed income-field passages, separate from corpus/vector gates.

Every passage is deterministic metadata assembled from a factory-owned native
IncomeCell. It is not a contiguous PDF quotation, generated answer, cosine
score, financial verdict or serialized-index trust boundary. Loading requires
fresh source replay; no model, benchmark annotation or question is consumed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

from .core import Chunk
from .sources import file_sha256
from .statement_startup import StatementSnapshot, _model_binding
from .structured_income import IncomeCell, ReportInput, build_income_index, validate_income_index

SERIALIZER_PROFILE = 'native-income-field-passages-v1;source-only;no-ISO-currency-inference'
_MAX_CELLS = 65536
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_COMPLETE_FILES = frozenset({'passages.jsonl', 'manifest.json', 'build-receipt.json'})
_MODULES = ('source_passages.py', 'structured_income.py', 'statement_startup.py', 'sources.py', 'core.py')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'duplicate JSON member')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('non-finite JSON constant')


def _json(body):
    try:
        return json.loads(body.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('invalid UTF-8 JSON artifact') from error


def _linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def _path(path, *, directory=False, existing=True):
    _require(isinstance(path, Path) and path.is_absolute(), 'artifact path must be an absolute local path')
    _require(not any(_linked(part) for part in (path, *path.parents)), 'linked artifact path is unsupported')
    if existing:
        _require(path.is_dir() if directory else path.is_file(), 'artifact entry is absent or has wrong type')
    return path.resolve(strict=existing)


def _source_snapshot(snapshot):
    """Verify declarations and replay all native cells, including omissions."""
    _require(type(snapshot) is StatementSnapshot, 'passages require a source-built statement snapshot')
    validate_income_index(snapshot.index)
    _require(type(snapshot.index.cells) is tuple and len(snapshot.index.cells) <= _MAX_CELLS,
             'source cell budget or shape is invalid')
    _require(type(snapshot.index.reports) is tuple and 0 < len(snapshot.index.reports) <= 128,
             'source report budget or shape is invalid')
    _require(type(snapshot.input_identities) is tuple and 2 <= len(snapshot.input_identities) <= 260,
             'source identity budget or shape is invalid')
    _require(all(type(identity.bytes) is int and identity.bytes >= 0 for identity in snapshot.input_identities),
             'source input byte identity is invalid')
    _require(sum(identity.bytes for identity in snapshot.input_identities) <= _MAX_SOURCE_BYTES,
             'source input byte budget exceeded')
    _require(len({identity.path for identity in snapshot.input_identities}) == len(snapshot.input_identities),
             'duplicate source input identity')
    _require(all(type(cell) is IncomeCell for cell in snapshot.index.cells), 'invalid source cell type')
    _require(len({cell.cell_id for cell in snapshot.index.cells}) == len(snapshot.index.cells),
             'duplicate source cell identity')
    _require(len({record[0] for record in snapshot.index.reports}) == len(snapshot.index.reports),
             'duplicate source report identity')
    snapshot.verify_unchanged()
    snapshot.verify_report_bindings()
    sources = {source.document_id: source for source in snapshot.sources}
    reports = []
    selected_pdf_paths = set()
    for document_id, digest, year, quarter in snapshot.index.reports:
        source = sources[document_id]
        paths = [identity.path for identity in snapshot.input_identities[2:]
                 if identity.path.name == source.filename and identity.sha256 == digest]
        _require(len(paths) == 1, 'native source PDF input is missing or ambiguous')
        selected_pdf_paths.add(paths[0])
        reports.append(ReportInput(document_id, paths[0], digest, year, quarter))
    extras = tuple(identity for identity in snapshot.input_identities[2:]
                   if identity.path not in selected_pdf_paths)
    _require(len(extras) in (0, 2), 'unsupported optional corpus identity layout')
    if extras:
        _require(tuple(snapshot.input_identities[2:4]) == extras,
                 'optional corpus identities must precede selected PDF inputs')
        _require(_model_binding(extras[0].path, extras[1].path, snapshot.sources) == extras,
                 'optional corpus binding differs from snapshot identities')
    fresh = build_income_index(tuple(reports))
    # Python equality treats True == 1 and 1 == 1.0. Canonical bytes retain the
    # exact native field types and prevent that accidental identity substitution.
    def index_payload(index):
        return _encoded({'reports':index.reports, 'cells':tuple(asdict(cell) for cell in index.cells),
                         'omissions':tuple(asdict(item) for item in index.omissions)})
    _require(index_payload(fresh) == index_payload(snapshot.index),
             'source index cells or omissions differ from native replay')
    snapshot.verify_unchanged()
    snapshot.verify_report_bindings()


@dataclass(frozen=True, slots=True)
class SourcePassage:
    serializer_profile: str
    cell: IncomeCell
    chunk: Chunk


def _passage(cell):
    spans = dict(cell.bindings)
    report_kind = 'annual' if cell.fiscal_quarter is None else f'Q{cell.fiscal_quarter}'
    date_literals = ' | '.join(spans[name].literal for name in ('date', 'date_year') if name in spans)
    section = spans['section'].literal if 'section' in spans else 'monetary statement rows'
    currency = ('Currency: not applicable' if cell.unit_kind == 'count' else
                f'Printed currency: {cell.currency_literal}; ISO currency unverified')
    text = (
        f'Report: {cell.document_id}; Fiscal year: {cell.fiscal_year}; Report period: {report_kind}; '
        f'Physical PDF page: {cell.physical_page}; Statement: {spans["statement"].literal}; '
        f'Section: {section}; Row: {cell.row_literal}; Row path: {cell.row_path}; '
        f'Column: {cell.column_number}; Column role: {cell.column_role}; '
        f'Duration header: {spans["duration"].literal}; Period end (ISO): {cell.period_end}; '
        f'Native date spans: {date_literals}; Printed value: {cell.value_literal}; '
        f'Unit kind: {cell.unit_kind}; Source scale: {cell.scale}; '
        f'Native unit header: {spans["unit"].literal}; {currency}.')
    _require(len(text) <= 8192, 'source passage text budget exceeded')
    identity = _encoded({'cell_id': cell.cell_id, 'serializer_profile': SERIALIZER_PROFILE, 'text': text})
    passage_id = f'{cell.document_id}:p{cell.physical_page}:source-cell:{sha256(identity).hexdigest()}'
    chunk = Chunk(passage_id, cell.document_id, cell.physical_page, text,
                  spans['value'].bbox, spans['row'].bbox, spans['date'].bbox)
    return SourcePassage(SERIALIZER_PROFILE, cell, chunk)


def derive_source_passages(snapshot: StatementSnapshot) -> tuple[SourcePassage, ...]:
    """Derive every retained cell, without altering native cell ID/field meaning."""
    _source_snapshot(snapshot)
    result = tuple(_passage(cell) for cell in snapshot.index.cells)
    _require(len({record.chunk.chunk_id for record in result}) == len(result), 'duplicate passage identity')
    snapshot.verify_unchanged()
    return result


def _implementation():
    directory = Path(__file__).resolve().parent
    return {name: file_sha256(directory / name) for name in _MODULES}


def _payload(snapshot, records):
    body = b''.join(_encoded({'schema_version': 1, 'serializer_profile': record.serializer_profile,
        'cell': asdict(record.cell), 'chunk': asdict(record.chunk)}) for record in records)
    _require(len(body) <= _MAX_ARTIFACT_BYTES, 'source passage artifact byte budget exceeded')
    manifest = {'schema_version': 1, 'artifact_kind': 'source-bound-income-field-passages',
        'serializer_profile': SERIALIZER_PROFILE, 'passages_sha256': sha256(body).hexdigest(),
        'passages_bytes': len(body), 'counts': {'reports': len(snapshot.index.reports),
            'passages': len(records), 'omissions': len(snapshot.index.omissions),
            'unit_kinds': dict(sorted(Counter(record.cell.unit_kind for record in records).items()))},
        'reports': snapshot.index.reports, 'omissions': tuple(asdict(item) for item in snapshot.index.omissions),
        # Byte identities are portable; caller-owned absolute source paths are
        # rechecked in the supplied snapshot, not serialized as public identity.
        'input_identities': tuple({'sha256': identity.sha256, 'bytes': identity.bytes}
                                  for identity in snapshot.input_identities),
        'financial_correctness': 'unestablished', 'human_acceptance': False,
        'text_semantics': 'deterministic source-field assembly;not a contiguous PDF quotation',
        'score_semantics': 'none;source-only artifact', 'model_runs': 0, 'embedding_runs': 0}
    manifest_body = _encoded(manifest)
    receipt = {'schema_version': 1, 'status': 'complete', 'scope': 'source-only-field-passage-build',
        'serializer_profile': SERIALIZER_PROFILE, 'passage_count': len(records),
        'passages_sha256': sha256(body).hexdigest(), 'passages_bytes': len(body),
        'manifest_sha256': sha256(manifest_body).hexdigest(), 'manifest_bytes': len(manifest_body),
        'implementation_sha256': _implementation(), 'post_build_verified': True,
        'model_runs': 0, 'embedding_runs': 0, 'human_acceptance': False}
    return {'passages.jsonl': body, 'manifest.json': manifest_body, 'build-receipt.json': _encoded(receipt)}, receipt


def _output_path(output_dir, snapshot):
    output = _path(output_dir, existing=False)
    _require(not os.path.lexists(output_dir), 'output directory already exists')
    _path(output.parent, directory=True)
    for identity in snapshot.input_identities:
        actual_input = _path(identity.path)
        _require(not actual_input.is_relative_to(output) and not output.is_relative_to(actual_input),
                 'output overlaps a read-only input')
    pdf_paths = [_path(identity.path) for identity in snapshot.input_identities
                 if identity.path.suffix.lower() == '.pdf']
    for path in pdf_paths:
        _require(not output.is_relative_to(path.parent), 'output overlaps original source directory')
    # The factory identity layout is registry, declarations, optional paired
    # model-corpus inputs, then PDFs. Preserve that separate corpus namespace.
    extra_paths = [_path(identity.path) for identity in snapshot.input_identities[2:]
                   if identity.path.suffix.lower() != '.pdf']
    for path in extra_paths:
        _require(not output.is_relative_to(path.parent), 'output overlaps bound corpus directory')
    return output


def _write_exclusive(path, body):
    with path.open('xb') as handle:
        handle.write(body)


def _artifact_bodies(directory, *, pending=False):
    _path(directory, directory=True)
    expected = _COMPLETE_FILES | ({'pending.json'} if pending else set())
    _require({path.name for path in directory.iterdir()} == expected,
             'artifact contains pending, failed, missing or unsupported entries')
    result = {}
    total = 0
    for name in sorted(_COMPLETE_FILES):
        path = _path(directory / name)
        size = path.stat().st_size
        _require(size <= _MAX_ARTIFACT_BYTES, 'artifact file byte budget exceeded')
        with path.open('rb') as stream:
            body = stream.read(_MAX_ARTIFACT_BYTES + 1)
        total += len(body)
        _require(len(body) == size and total <= _MAX_ARTIFACT_BYTES, 'artifact read byte budget exceeded')
        if name == 'passages.jsonl':
            lines = body.splitlines()
            _require(len(lines) <= _MAX_CELLS, 'artifact passage budget exceeded')
            for line in lines:
                _json(line)
        else:
            _json(body)
        result[name] = body
    return result


def build_source_passage_artifact(*, snapshot: StatementSnapshot, output_dir: Path) -> dict:
    """Create one owned sidecar; every failure leaves a refusal guard."""
    _require(type(snapshot) is StatementSnapshot, 'passages require a source-built statement snapshot')
    output = _output_path(output_dir, snapshot)
    records = derive_source_passages(snapshot)
    payload, receipt = _payload(snapshot, records)
    snapshot.verify_unchanged()
    output.mkdir(exist_ok=False)
    pending_body = _encoded({'schema_version': 1, 'status': 'pending', 'serializer_profile': SERIALIZER_PROFILE})
    try:
        _write_exclusive(output / 'pending.json', pending_body)
        for name, body in payload.items():
            _write_exclusive(output / name, body)
        snapshot.verify_unchanged()
        snapshot.verify_report_bindings()
        _require(_artifact_bodies(output, pending=True) == payload, 'written artifact differs from source-derived payload')
        _require(_implementation() == receipt['implementation_sha256'], 'implementation changed during build')
        marker = _path(output / 'pending.json')
        _require(marker.read_bytes() == pending_body, 'owned pending marker changed')
        snapshot.verify_unchanged()
        _require(_artifact_bodies(output, pending=True) == payload, 'artifact changed at completion')
        marker = _path(output / 'pending.json')
        _require(marker.read_bytes() == pending_body, 'owned pending marker changed')
        # Every fallible closure check completes under the refusal guard. Do
        # nothing fallible after final unlink: even failure-receipt storage may
        # be unavailable, and must never turn a failed closure into valid data.
        marker.unlink()
        return receipt
    except Exception as error:
        kind = type(error).__name__
        if kind not in {'ValueError','OSError','RuntimeError','FileNotFoundError','PermissionError'}:
            kind = 'BuildError'
        try:
            _write_exclusive(output / 'failure.json', _encoded({'schema_version':1,'status':'failed','error_type':kind}))
        except OSError:
            pass
        raise


def load_source_passage_artifact(*, snapshot: StatementSnapshot, artifact_dir: Path) -> tuple[SourcePassage, ...]:
    """Accept only exact canonical artifacts reproduced from fresh native sources."""
    directory = _path(artifact_dir, directory=True)
    before = _artifact_bodies(directory)
    records = derive_source_passages(snapshot)
    expected, _ = _payload(snapshot, records)
    _require(before == expected, 'artifact differs from fresh source replay and canonical schema')
    snapshot.verify_unchanged()
    snapshot.verify_report_bindings()
    _require(_artifact_bodies(directory) == before, 'artifact changed during source replay')
    return records
