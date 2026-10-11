"""Pinned, allowlisted source acquisition for the multi-year corpus."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path, PurePath
from types import MappingProxyType
from collections.abc import Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ALLOWED_SOURCE_HOSTS = frozenset({'s201.q4cdn.com'})


@dataclass(frozen=True, slots=True)
class SourceRecord:
    document_id: str
    label: str
    url: str
    filename: str
    sha256: str | None
    provenance: Mapping[str, object] | None = None
    reporting_period: Mapping[str, object] | None = None


def select_task_sources(records: list[SourceRecord], *, task: str) -> tuple[SourceRecord, ...]:
    """Select explicitly reviewed material for a declared task, never infer intent.

    This opt-in selection is not a legal review, query classifier or retrieval
    correctness check. Legacy records without provenance cannot participate.
    """
    kinds = {'reported_results': 'official_filing',
             'management_commentary': 'earnings_call_transcript',
             'third_party_view': 'third_party_analysis'}
    if task not in kinds:
        raise ValueError('declare a supported source task; cross-source comparison requires separate requests')
    return tuple(record for record in records
                 if record.provenance is not None
                 and record.provenance.get('source_type') == kinds[task]
                 and record.provenance.get('permission_status') == 'reviewed'
                 and {'index', 'display'}.issubset(record.provenance.get('permitted_operations', ())))


def _reporting_period(value: object) -> Mapping[str, object] | None:
    """Validate declared quarter metadata; never infer dates from display labels."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('reporting_period must be an object')
    year, quarter = value.get('fiscal_year'), value.get('quarter')
    if type(year) is not int or not 1900 <= year <= 9999:
        raise ValueError('reporting_period fiscal_year must be an integer year')
    if type(quarter) is not int or quarter not in (1, 2, 3, 4):
        raise ValueError('reporting_period quarter must be 1 through 4')
    end_date = value.get('end_date')
    try:
        parsed = date.fromisoformat(end_date) if isinstance(end_date, str) else None
    except ValueError as error:
        raise ValueError('reporting_period end_date must be a calendar date') from error
    if parsed is None or parsed.isoformat() != end_date:
        raise ValueError('reporting_period end_date must use YYYY-MM-DD')
    metadata = dict(fiscal_year=year, quarter=quarter, end_date=end_date)
    for key in ('source_locator', 'reviewed_by'):
        field = value.get(key)
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f'reporting_period {key} must be non-empty')
        metadata[key] = field.strip()
    return MappingProxyType(metadata)


def _reviewed_provenance(row: dict[str, object]) -> Mapping[str, object]:
    """Validate explicit review declarations, not infer legal rights from a host."""
    fields = ('source_type', 'publisher', 'attribution', 'published_date',
              'event_date', 'financial_period', 'source_locator',
              'permission_status', 'permission_evidence')
    metadata = {}
    for key in fields:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key} must be a non-empty review field')
        metadata[key] = value.strip()
    if metadata['source_type'] not in {
        'official_filing', 'earnings_call_transcript', 'third_party_analysis'
    }:
        raise ValueError('unsupported source_type')
    if metadata['permission_status'] != 'reviewed':
        raise ValueError('source permission has not been reviewed')
    for key in ('published_date', 'event_date'):
        try:
            parsed = date.fromisoformat(metadata[key])
        except ValueError as error:
            raise ValueError(f'{key} must be an ISO calendar date') from error
        if parsed.isoformat() != metadata[key]:
            raise ValueError(f'{key} must use YYYY-MM-DD')
    operations = row.get('permitted_operations')
    if (not isinstance(operations, list) or not operations
        or any(not isinstance(op, str) or op not in {'download', 'index', 'display'}
               for op in operations)):
        raise ValueError('permitted_operations must explicitly name reviewed uses')
    metadata['permitted_operations'] = tuple(dict.fromkeys(operations))
    return MappingProxyType(metadata)


def require_source_operation(record: SourceRecord, operation: str) -> None:
    """Legacy pinned manifests retain their original path; v2 uses explicit gates."""
    if record.provenance is not None:
        if operation not in record.provenance['permitted_operations']:
            raise ValueError(f'{record.document_id}: {operation} permission missing')


def _validate_sha256(value: object) -> str | None:
    if value in (None, ''):
        return None
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in '0123456789abcdef' for character in normalized):
        raise ValueError('sha256 must be 64 lowercase hexadecimal characters')
    return normalized


def _record_from_mapping(row: dict[str, object], *, schema_version: int = 1) -> SourceRecord:
    review_fields = {'source_type', 'publisher', 'attribution', 'published_date',
                     'event_date', 'financial_period', 'source_locator',
                     'permission_status', 'permission_evidence', 'permitted_operations'}
    if schema_version == 1 and review_fields.intersection(row):
        raise ValueError('review fields require schema_version 2; do not silently discard provenance')
    document_id = str(row.get('document_id', '')).strip()
    label = str(row.get('label', '')).strip()
    url = str(row.get('url', '')).strip()
    filename = str(row.get('filename', '')).strip()
    if not document_id or not label:
        raise ValueError('document_id and label must be non-empty')

    provenance = _reviewed_provenance(row) if schema_version == 2 else None
    pinned_hash = _validate_sha256(row.get('sha256'))
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('source URL must be HTTPS with a hostname')
    reviewed_local_external = (
        parsed.hostname not in ALLOWED_SOURCE_HOSTS
        and
        provenance is not None
        and provenance['source_type'] in {'earnings_call_transcript', 'third_party_analysis'}
        and 'index' in provenance['permitted_operations']
        and 'download' not in provenance['permitted_operations']
    )
    if parsed.hostname not in ALLOWED_SOURCE_HOSTS and not reviewed_local_external:
        raise ValueError('source URL must be HTTPS on an allowlisted NVIDIA host')
    if parsed.hostname not in ALLOWED_SOURCE_HOSTS and pinned_hash is None:
        raise ValueError('reviewed local external sources require a pinned sha256')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('source URL must not contain credentials, query, or fragment')
    # A reviewed local PDF may originate from an HTML article. Preserve the
    # original attribution URL; it is not an automatic download endpoint.
    if not reviewed_local_external and not parsed.path.lower().endswith('.pdf'):
        raise ValueError('source URL must point to a PDF')

    if (
        not filename
        or PurePath(filename).name != filename
        or not filename.lower().endswith('.pdf')
    ):
        raise ValueError('filename must be a safe PDF basename')

    return SourceRecord(
        document_id=document_id,
        label=label,
        url=url,
        filename=filename,
        sha256=pinned_hash,
        provenance=provenance,
        reporting_period=_reporting_period(row.get('reporting_period')),
    )


def load_source_manifest(path: Path) -> tuple[SourceRecord, ...]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    schema_version = payload.get('schema_version')
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError('unsupported source manifest schema')
    rows = payload.get('sources')
    if not isinstance(rows, list) or not rows:
        raise ValueError('source manifest must contain at least one source')
    records = tuple(_record_from_mapping(row, schema_version=schema_version)
                    for row in rows if isinstance(row, dict))
    if len(records) != len(rows):
        raise ValueError('every source record must be an object')
    if len({record.document_id for record in records}) != len(records):
        raise ValueError('document_id values must be unique')
    if len({record.filename for record in records}) != len(records):
        raise ValueError('filename values must be unique')
    return records


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_source(
    record: SourceRecord,
    *,
    destination_dir: Path,
    allow_unpinned: bool = False,
) -> tuple[Path, str, str]:
    """Download one trusted PDF atomically and verify its pinned digest."""
    require_source_operation(record, 'download')
    if urlparse(record.url).hostname not in ALLOWED_SOURCE_HOSTS:
        raise ValueError('automatic download requires an allowlisted source host')
    if record.sha256 is None and not allow_unpinned:
        raise ValueError(f'{record.document_id} has no pinned sha256')

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / record.filename
    if destination.exists():
        observed = file_sha256(destination)
        if record.sha256 is None or observed == record.sha256:
            return destination, observed, 'existing'
        raise ValueError(f'existing file hash mismatch for {record.document_id}')

    partial = destination.with_suffix(destination.suffix + '.part')
    partial.unlink(missing_ok=True)
    request = Request(record.url, headers={'User-Agent': 'NVIDIA-RAG-Learning/0.1'})
    try:
        with urlopen(request, timeout=60) as response, partial.open('wb') as target:
            final_url = response.geturl()
            final_host = urlparse(final_url).hostname
            if final_host not in ALLOWED_SOURCE_HOSTS:
                raise ValueError('download redirected outside the allowlisted host')
            for block in iter(lambda: response.read(1024 * 1024), b''):
                target.write(block)
        with partial.open('rb') as source:
            if source.read(5) != b'%PDF-':
                raise ValueError(f'{record.document_id} did not return a PDF')
        observed = file_sha256(partial)
        if record.sha256 is not None and observed != record.sha256:
            raise ValueError(f'downloaded file hash mismatch for {record.document_id}')
        partial.replace(destination)
        return destination, observed, 'downloaded'
    except Exception:
        partial.unlink(missing_ok=True)
        raise
