"""Replay provisional financial cell coordinates and exact corpus coverage.

No retriever or encoder runs here. A matching source tuple is an assistant
candidate, not independent human gold. Different extraction profiles retain
different offsets; ambiguous text-to-coordinate or chunk mappings never pass.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from hashlib import sha256
import argparse
import json
import math
import os
from pathlib import Path

from acceptance.expanded_cases import load_experiment, load_case_bundle


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def canonical_bytes(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def experiment_binding_sha256(config):
    bound = plain(config)
    bound['input_sha256'].pop('source_replay')
    return sha256(canonical_bytes(bound)).hexdigest()


def occurrences(text, needle):
    require(type(text) is str and type(needle) is str and needle,
            'nonempty exact source text is required')
    starts, offset = [], 0
    while (found := text.find(needle, offset)) >= 0:
        starts.append(found)
        offset = found + 1
    return starts


def checked_box(box):
    require(isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(n) in (int, float) and math.isfinite(n) for n in box)
            and box[0] < box[2] and box[1] < box[3], 'invalid source geometry box')
    return list(box)


def same_box(left, right):
    return all(abs(a - b) <= 1e-4 for a, b in zip(checked_box(left), checked_box(right)))


def verify_span(span, page_text, boxes, *, sort):
    require(isinstance(span, Mapping) and set(span) == {'start', 'end', 'text', 'bbox'},
            'source span fields differ')
    start, end, text = span['start'], span['end'], span['text']
    require(type(start) is int and type(end) is int and type(text) is str
            and 0 <= start < end <= len(page_text) and end - start == len(text)
            and page_text[start:end] == text, 'source span does not match page text')
    found = occurrences(page_text, text)
    geometry = [checked_box(box) for box in boxes.get(text, [])]
    if sort:
        geometry.sort(key=lambda box: (round(box[1], 3), box[0]))
    require(len(found) == len(geometry) and start in found,
            'source occurrence-to-geometry mapping is ambiguous')
    require(same_box(span['bbox'], geometry[found.index(start)]),
            'source occurrence bbox does not match its exact text offset')
    return checked_box(span['bbox'])


def printed_date(text):
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    try:
        month, day, year = text.replace(',', '').split()
        return date(int(year), months[month], int(day)).isoformat()
    except (ValueError, KeyError, TypeError):
        raise ValueError('unsupported exact printed date heading') from None


def printed_duration(text):
    words = {'One': 1, 'Two': 2, 'Three': 3, 'Six': 6, 'Nine': 9, 'Twelve': 12}
    for word, count in words.items():
        if text in (f'{word} Months Ended', f'{word} Month Ended'):
            return count
    raise ValueError('unsupported exact printed duration heading')


def center_x(box):
    return (box[0] + box[2]) / 2


def center_y(box):
    return (box[1] + box[3]) / 2


def verify_cell_geometry(cell, packet, page_text, boxes):
    fields = {'schema_version', 'cell_id', 'pdf_sha256', 'physical_page', 'page_text_sha256',
              'extraction_profile', 'selected_column_id', 'spans', 'columns', 'row_band',
              'review_status', 'human_acceptance'}
    require(type(packet) is dict and set(packet) == fields and type(packet['schema_version']) is int
            and packet['schema_version'] == 2, 'association receipt schema differs')
    require(packet['review_status'] == 'assistant-source-reviewed-candidate'
            and packet['human_acceptance'] is False, 'source replay does not establish human gold')
    binding = cell['bindings']
    for name in ('cell_id', 'pdf_sha256', 'physical_page'):
        require(type(packet[name]) is type(cell[name]) and packet[name] == cell[name],
                'association source identity differs')
    require(packet['extraction_profile'] == plain(binding['extraction_profile'])
            and packet['page_text_sha256'] == binding['page_text_sha256']
            == sha256(page_text.encode('utf-8')).hexdigest()
            and len(page_text) == binding['page_text_chars'], 'source extraction profile or page text differs')
    require(packet['spans'] == plain(binding['spans']), 'association selected spans differ')
    spans = binding['spans']
    essential = {'statement', 'unit', 'date', 'row', 'value'}
    if cell['period_kind'] == 'duration':
        essential.add('duration')
    if cell['unit_kind'] in ('monetary', 'per_share'):
        essential.add('currency')
    require(all(spans.get(name) is not None for name in essential), 'essential source binding is missing')
    sort = binding['extraction_profile']['sort']
    actual = {name: verify_span(spans[name], page_text, boxes, sort=sort) for name in essential}
    columns = packet['columns']
    require(type(columns) is list and 0 < len(columns) <= 32, 'column inventory is missing or exceeds budget')
    by_id = {}
    for column in columns:
        require(type(column) is dict and set(column) == {'column_id', 'column_role', 'period_end',
            'duration_months', 'date_span', 'duration_span', 'x_range'}, 'column inventory fields differ')
        require(type(column['column_id']) is str and column['column_id'] not in by_id,
                'duplicate or invalid column identity')
        by_id[column['column_id']] = column
        date_box = verify_span(column['date_span'], page_text, boxes, sort=sort)
        require(printed_date(column['date_span']['text']) == column['period_end'], 'column date qualifier differs')
        interval = column['x_range']
        require(type(interval) is list and len(interval) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in interval)
                and interval[0] < interval[1], 'invalid column horizontal interval')
        require(interval[0] <= center_x(date_box) <= interval[1], 'date heading lies outside declared column')
        if cell['period_kind'] == 'duration':
            group = verify_span(column['duration_span'], page_text, boxes, sort=sort)
            require(type(column['duration_months']) is int
                    and printed_duration(column['duration_span']['text']) == column['duration_months'],
                    'column duration qualifier differs')
            require(group[1] < date_box[1], 'duration heading must precede its dated columns')
        else:
            require(column['duration_months'] is None and column['duration_span'] is None,
                    'instant column must not acquire a duration')
    intervals = sorted(column['x_range'] for column in columns)
    require(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])),
            'financial column intervals overlap')
    # A centered group title is narrower than the columns it heads. Its frozen
    # member-column interval, rather than its glyph width, defines that group.
    if cell['period_kind'] == 'duration':
        groups = {}
        for column in columns:
            key = canonical_bytes(column['duration_span'])
            groups.setdefault(key, []).append(column)
        bands = []
        for members in groups.values():
            left = min(c['x_range'][0] for c in members)
            right = max(c['x_range'][1] for c in members)
            group = members[0]['duration_span']['bbox']
            require(left <= center_x(group) <= right, 'duration heading lies outside its frozen column group')
            bands.append((left, right))
        bands.sort()
        require(all(a[1] <= b[0] for a, b in zip(bands, bands[1:])), 'duration groups overlap')
    require(packet['selected_column_id'] in by_id, 'selected column is absent from inventory')
    selected = by_id[packet['selected_column_id']]
    require(selected['column_role'] == cell['column_role'] and selected['period_end'] == cell['period_end']
            and selected['duration_months'] == cell['duration_months'], 'selected financial column differs')
    require(selected['date_span'] == plain(spans['date']), 'selected date occurrence belongs to another column')
    if cell['period_kind'] == 'duration':
        require(selected['duration_span'] == plain(spans['duration']), 'selected duration belongs to another group')
    left, right = selected['x_range']
    require(left <= center_x(actual['value']) <= right, 'selected amount lies outside its dated column')
    band = packet['row_band']
    require(type(band) is list and len(band) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in band)
            and band[0] < band[1]
            and band[0] <= center_y(actual['row']) <= band[1]
            and band[0] <= center_y(actual['value']) <= band[1]
            and abs(center_y(actual['row']) - center_y(actual['value'])) <=
                max(actual['row'][3] - actual['row'][1], actual['value'][3] - actual['value'][1]) / 2
            and actual['row'][2] < actual['value'][0], 'selected amount is not in its metric row')
    require(actual['date'][3] < actual['value'][1]
            and actual['statement'][1] < actual['date'][1]
            and actual['unit'][1] < actual['date'][1], 'financial headings do not precede selected amount')
    if 'currency' in essential:
        require(left <= center_x(actual['currency']) <= right
                and actual['currency'][1] <= actual['value'][3], 'currency glyph belongs to another column')
    return {'source_binding_passed': True, 'human_gold': False,
            'essential_roles': sorted(essential), 'selected_column_id': selected['column_id']}


def map_chunk_ranges(page_text, chunks):
    mapped, seen = [], set()
    for chunk in chunks:
        require(type(chunk['chunk_id']) is str and chunk['chunk_id'] not in seen, 'duplicate chunk mapping identity')
        seen.add(chunk['chunk_id'])
        found = occurrences(page_text, chunk['text'])
        mapped.append({'chunk_id': chunk['chunk_id'],
                       'status': 'unique' if len(found) == 1 else ('ambiguous' if found else 'absent'),
                       'range': [found[0], found[0] + len(chunk['text'])] if len(found) == 1 else None,
                       'text_sha256': sha256(chunk['text'].encode('utf-8')).hexdigest()})
    return mapped


def cover_ranges(required, mappings, accepted_chunk_ids):
    require(type(accepted_chunk_ids) in (list, tuple) and len(set(accepted_chunk_ids)) == len(accepted_chunk_ids),
            'accepted chunk IDs must be distinct')
    by_id = {item['chunk_id']: item for item in mappings}
    require(set(accepted_chunk_ids) <= by_id.keys(), 'unknown accepted chunk mapping')
    if not required:
        return 'not_evaluated'
    selected = [by_id[key] for key in accepted_chunk_ids]
    if any(item['status'] != 'unique' for item in selected):
        return 'not_evaluated'
    intervals = sorted(item['range'] for item in selected)
    for target in required.values():
        cursor, stop = target
        require(type(cursor) is int and type(stop) is int and 0 <= cursor < stop, 'invalid essential corpus range')
        for start, end in intervals:
            if start > cursor:
                break
            if start <= cursor:
                cursor = max(cursor, end)
            if cursor >= stop:
                break
        if cursor < stop:
            return 'failed'
    return 'passed'


def map_cell_corpus_spans(cell, corpus_text, boxes):
    ranges = {}
    for name, span in cell['bindings']['spans'].items():
        if span is None:
            continue
        starts = occurrences(corpus_text, span['text'])
        geometry = boxes.get(span['text'], [])
        selected = [i for i, box in enumerate(geometry) if same_box(box, span['bbox'])]
        if len(starts) != len(geometry) or len(selected) != 1:
            return {'status': 'not_evaluated', 'ranges': {},
                    'reason': 'corpus occurrence-to-source geometry is not unique'}
        at = starts[selected[0]]
        ranges[name] = [at, at + len(span['text'])]
    return {'status': 'complete', 'ranges': ranges}


def evaluate_candidate_coverage(receipt, case_id, accepted_chunk_ids):
    require(case_id in receipt['ordered_case_ids'], 'unknown case coverage identity')
    require(len(set(accepted_chunk_ids)) == len(accepted_chunk_ids)
            and set(accepted_chunk_ids) <= set(receipt['chunk_ids']), 'unknown or duplicate accepted chunk identity')
    rows = []
    for cell in receipt['cells']:
        if cell['case_id'] != case_id:
            continue
        if not cell['source_binding_passed'] or cell['corpus_spans']['status'] != 'complete':
            status = 'not_evaluated'
        else:
            same_page = {m['chunk_id'] for m in cell['chunk_mappings']}
            status = cover_ranges(cell['corpus_spans']['ranges'], cell['chunk_mappings'],
                                  [key for key in accepted_chunk_ids if key in same_page])
        rows.append({'cell_id': cell['cell_id'], 'status': status})
    statuses = {row['status'] for row in rows}
    verdict = ('not_evaluated' if not rows or 'not_evaluated' in statuses
               else ('passed' if statuses == {'passed'} else 'failed'))
    return {'case_id': case_id, 'cells': rows, 'joint_source_candidate_coverage': verdict,
            'human_gold': False, 'generated_answer_evaluated': False}


def local_path(path, *, directory=False):
    require(isinstance(path, Path) and path.is_absolute(), 'input requires an absolute local path')
    require(not path.is_symlink() and not any(p.is_symlink() or
            (hasattr(p, 'is_junction') and p.is_junction()) for p in (path, *path.parents)),
            'linked input path is unsupported')
    require(path.is_dir() if directory else path.is_file(), 'local input is absent or has wrong type')
    return path


def read_body(path, limit=32_000_000):
    local_path(path)
    require(path.stat().st_size <= limit, 'input exceeds byte budget')
    with path.open('rb') as handle:
        body = handle.read(limit + 1)
    require(len(body) <= limit, 'input exceeds byte budget')
    return body


def strict_json(body):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('nonfinite JSON number')
    try:
        value = json.loads(body, object_pairs_hook=pairs, parse_constant=constant)
        # Also rejects decoded exponent overflow and non-UTF8 strings.
        canonical_bytes(value)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError('invalid bounded replay JSON') from None


def checked_source_body(path, expected_sha):
    body = read_body(path, 256_000_000)
    require(sha256(body).hexdigest() == expected_sha, 'source PDF SHA-256 differs')
    return body


def exact_boxes(page, texts):
    result = {}
    for text in texts:
        selected = []
        for box in page.search_for(text):
            # PyMuPDF search is case-insensitive; keep only the exact clipped
            # source spelling and require one rectangular occurrence.
            clipped = page.get_text('text', clip=box, sort=True).strip()
            if clipped == text:
                selected.append(list(box))
        result[text] = selected
    return result


def read_source_page(body, physical_page, profile, texts, expected_page_count):
    import pymupdf
    require(pymupdf.version[0] == profile['version'], 'PyMuPDF extraction version differs')
    with pymupdf.open(stream=body, filetype='pdf') as document:
        require(len(document) == expected_page_count and 1 <= physical_page <= len(document),
                'source physical page count differs')
        page = document[physical_page - 1]
        reviewed = page.get_text('text', sort=profile['sort'])
        if profile['normalization'] == 'strip':
            reviewed = reviewed.strip()
        corpus = page.get_text('text', sort=False).strip()
        return reviewed, corpus, exact_boxes(page, texts)


def guard_path(path, kind):
    return path.with_name(path.name + f'.{kind}.json')


def read_completed_replay(path):
    require(not os.path.lexists(guard_path(path, 'pending')), 'source replay is pending')
    require(not os.path.lexists(guard_path(path, 'failure')), 'source replay has a failure marker')
    body = read_body(path)
    value = strict_json(body)
    require(not os.path.lexists(guard_path(path, 'pending'))
            and not os.path.lexists(guard_path(path, 'failure')), 'source replay completion state changed')
    return value


def write_replay_exclusive(path, value, final_check):
    require(isinstance(path, Path) and path.is_absolute(), 'output requires an absolute local path')
    local_path(path.parent, directory=True)
    pending, failure = guard_path(path, 'pending'), guard_path(path, 'failure')
    require(not any(os.path.lexists(p) for p in (path, pending, failure)), 'source replay output already exists')
    with pending.open('xb') as handle:
        handle.write(b'{"status":"pending"}\n')
    try:
        final_check()
        with path.open('xb') as handle:
            handle.write(canonical_bytes(value) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        final_check()
        pending.unlink()
    except Exception:
        # No exception text or input values enter the failure diagnostic.
        try:
            with failure.open('xb') as handle:
                handle.write(b'{"status":"failed","code":"SOURCE_REPLAY_INCOMPLETE"}\n')
        except OSError:
            pass
        raise


def load_corpus(**kwargs):
    from nvidia_rag.corpus import load_verified_corpus
    return load_verified_corpus(**kwargs)


def snapshot(paths):
    return {name: sha256(read_body(path, 256_000_000)).hexdigest() for name, path in paths.items()}


def verify_rationale(kind, body, config, verified_columns):
    require(type(body) is dict, 'rationale receipt must be an object')
    if kind == 'registry-absence':
        require(set(body) == {'registry_sha256', 'missing_document_id'}
                and body['registry_sha256'] == config['input_sha256']['registry']
                and type(body['missing_document_id']) is str
                and body['missing_document_id'] not in {d['document_id'] for d in config['documents']},
                'registry absence rationale differs from actual registry')
    elif kind == 'source-column-inventory':
        require(set(body) == {'document_id', 'pdf_sha256', 'physical_page', 'columns', 'human_acceptance'}
                and body['human_acceptance'] is False and type(body['physical_page']) is int,
                'column rationale schema differs')
        registered = {d['document_id']: d['pdf_sha256'] for d in config['documents']}
        require(body['document_id'] in registered and body['pdf_sha256'] == registered[body['document_id']]
                and (body['document_id'], body['physical_page']) in verified_columns
                and body['columns'] == verified_columns[(body['document_id'], body['physical_page'])],
                'column rationale differs from replayed source inventory')
    elif kind == 'experiment-capability':
        require(set(body) == {'supported_currencies', 'fx_conversion'} and body['fx_conversion'] is False
                and body['supported_currencies'] == plain(config['supported_currencies']),
                'currency capability rationale differs from frozen experiment')
    elif kind == 'request-contradiction':
        require(set(body) == {'requested_duration_months', 'selected_column_duration_months'}
                and all(type(value) is int and 1 <= value <= 120 for value in body.values())
                and body['requested_duration_months'] != body['selected_column_duration_months'],
                'declared request contradiction rationale differs')
    else:
        raise ValueError('unsupported rationale receipt kind')
    return 'candidate-bound'


def replay_sources(*, source_dir, source_registry, cases, case_freeze, artifact_dir,
                   experiment, binding_dir):
    for directory in (source_dir, artifact_dir, binding_dir):
        local_path(directory, directory=True)
    cfg = load_experiment(experiment)
    bundle = load_case_bundle(cases, case_freeze, cfg)
    paths = {'registry': source_registry, 'cases': cases, 'case_freeze': case_freeze,
             'experiment': experiment, 'corpus_manifest': artifact_dir / 'manifest.json',
             'corpus_chunks': artifact_dir / 'chunks.jsonl', 'build_receipt': artifact_dir / 'build-receipt.json'}
    identities = snapshot(paths)
    require(identities['experiment'] == cfg.sha256, 'experiment changed between validation and input snapshot')
    for key in ('registry', 'cases', 'case_freeze', 'corpus_manifest', 'corpus_chunks'):
        require(identities[key] == cfg.data['input_sha256'][key], 'configured input SHA-256 differs')
    registry = strict_json(read_body(source_registry))
    require(type(registry) is dict and set(registry) == {'schema_version', 'sources'}
            and registry['schema_version'] == 1 and type(registry['sources']) is list,
            'source registry schema differs')
    registered = registry['sources']
    configured = cfg.data['documents']
    require([r['document_id'] for r in registered] == [r['document_id'] for r in configured],
            'registered document order differs')
    source_bodies, source_files = {}, []
    for actual, expected in zip(registered, configured, strict=True):
        require(actual['sha256'] == expected['pdf_sha256'] and actual['url'] == expected['issuer_url'],
                'registered source identity differs')
        filename = actual['filename']
        require(type(filename) is str and filename and not any(c in filename for c in '/\\:')
                and filename not in ('.', '..'), 'source filename is unsafe')
        name = 'pdf:' + actual['document_id']
        path = source_dir / filename
        body = checked_source_body(path, actual['sha256'])
        paths[name] = path
        identities[name] = actual['sha256']
        source_bodies[actual['document_id']] = body
        source_files.append({'document_id': actual['document_id'], 'pdf_sha256': actual['sha256'],
                             'source_bytes': len(body)})
    corpus = load_corpus(artifact_dir=artifact_dir, source_registry=source_registry)
    require(len(corpus.chunks) == cfg.data['chunk_count'] and corpus.manifest['document_count'] == cfg.data['document_count'],
            'configured corpus count differs')
    page_counts = {doc['document_id']: doc['page_count'] for doc in corpus.manifest['documents']}
    require(all(page_counts.get(doc['document_id']) == doc['page_count'] for doc in configured),
            'configured page counts differ from corpus')
    build = strict_json(read_body(paths['build_receipt']))
    corpus_profile = {'library': 'PyMuPDF', 'version': build['runtime']['PyMuPDF'],
                      'mode': 'text', 'sort': False, 'normalization': 'strip', 'text_encoding': 'utf-8',
                      'offset_unit': 'unicode_code_points', 'interval': 'half_open'}
    require(build['profile'] == {'name': 'plain-word-safe-400-0-v1', 'extraction_mode': 'plain',
        'strategy': 'word_safe', 'max_tokens': 400, 'overlap_tokens': 0, 'encoding_name': 'cl100k_base'},
        'corpus extraction/chunk profile differs')
    cells, verified_columns = [], {}
    docs = {doc['document_id']: doc for doc in configured}
    for case in bundle.cases:
        for cell in case.source_cells:
            binding = cell['bindings']
            require(binding['extraction_profile']['version'] == corpus_profile['version'],
                    'review and corpus extractor versions differ')
            packet_hash = binding['association_receipt_sha256']
            packet = None
            if packet_hash is not None:
                packet_path = binding_dir / (packet_hash + '.json')
                packet_body = read_body(packet_path, 2_000_000)
                require(sha256(packet_body).hexdigest() == packet_hash, 'association packet SHA-256 differs')
                paths['association:' + packet_hash] = packet_path
                identities['association:' + packet_hash] = packet_hash
                packet = strict_json(packet_body)
            texts = {span['text'] for span in binding['spans'].values() if span is not None}
            if packet is not None:
                for column in packet['columns']:
                    for name in ('date_span', 'duration_span'):
                        if column[name] is not None:
                            texts.add(column[name]['text'])
            reviewed, corpus_text, boxes = read_source_page(
                source_bodies[cell['document_id']], cell['physical_page'], binding['extraction_profile'],
                texts, docs[cell['document_id']]['page_count'])
            require(sha256(reviewed.encode()).hexdigest() == binding['page_text_sha256']
                    and len(reviewed) == binding['page_text_chars'], 'reviewed source page text differs')
            complete = cell['binding_complete'] and packet is not None
            if complete:
                result = verify_cell_geometry(cell, packet, reviewed, boxes)
                location = (cell['document_id'], cell['physical_page'])
                require(location not in verified_columns or verified_columns[location] == packet['columns'],
                        'source cells disagree on frozen column inventory')
                verified_columns[location] = packet['columns']
                corpus_spans = map_cell_corpus_spans(cell, corpus_text, boxes)
                # Coverage needs only essential roles from the verified tuple.
                corpus_spans['ranges'] = {name: value for name, value in corpus_spans['ranges'].items()
                                          if name in result['essential_roles']}
            else:
                result = {'source_binding_passed': False, 'human_gold': False, 'essential_roles': []}
                corpus_spans = {'status': 'not_evaluated', 'ranges': {}, 'reason': 'source bindings incomplete'}
            page_chunks = [{'chunk_id': chunk.chunk_id, 'text': chunk.text} for chunk in corpus.chunks
                           if chunk.document_id == cell['document_id'] and chunk.page_number == cell['physical_page']]
            cells.append({'case_id': case.case_id, 'cell_id': cell['cell_id'], 'document_id': cell['document_id'],
                          'pdf_sha256': cell['pdf_sha256'], 'physical_page': cell['physical_page'],
                          'association_receipt_sha256': packet_hash, 'source_profile': plain(binding['extraction_profile']),
                          'source_page_text_sha256': binding['page_text_sha256'],
                          'corpus_page_text_sha256': sha256(corpus_text.encode()).hexdigest(),
                          'corpus_page_text_chars': len(corpus_text),
                          'source_binding_passed': result['source_binding_passed'],
                          'essential_roles': result['essential_roles'], 'corpus_spans': corpus_spans,
                          'chunk_mappings': map_chunk_ranges(corpus_text, page_chunks)})
    referenced = {key for case in bundle.cases for key in case.expected['rationale_binding_ids']}
    rationales = []
    for binding in cfg.data['rationale_bindings']:
        if binding['binding_id'] not in referenced:
            continue
        digest = binding['receipt_sha256']
        rationale_path = binding_dir / (digest + '.json')
        rationale_bytes = read_body(rationale_path, 2_000_000)
        require(sha256(rationale_bytes).hexdigest() == digest, 'rationale receipt SHA-256 differs')
        rationale = strict_json(rationale_bytes)
        status = verify_rationale(binding['kind'], rationale, cfg.data, verified_columns)
        paths['rationale:' + digest] = rationale_path
        identities['rationale:' + digest] = digest
        rationales.append({'binding_id': binding['binding_id'], 'kind': binding['kind'],
                           'receipt_sha256': digest, 'status': status})
    require(snapshot(paths) == identities, 'input identity changed during source replay')
    expected = {cell_id for case in bundle.cases for cell_id in case.expected['cell_ids']}
    passed = bool(expected) and all(row['source_binding_passed'] for row in cells if row['cell_id'] in expected)
    return {'schema_version': 2, 'status': 'complete', 'scope': 'operator-local-source-candidate-replay',
            'experiment_binding_sha256': experiment_binding_sha256(cfg.data),
            'input_sha256': {key: identities[key] for key in ('registry', 'cases', 'case_freeze',
                              'corpus_manifest', 'corpus_chunks', 'build_receipt')},
            'source_files': source_files, 'ordered_case_ids': [c.case_id for c in bundle.cases],
            'counts': plain(bundle.counts), 'corpus_profile': corpus_profile,
            'chunk_ids': [chunk.chunk_id for chunk in corpus.chunks], 'cells': cells,
            'rationale_bindings': rationales,
            'source_binding_passed': passed, 'human_gold_count': 0, 'generation_runs': 0,
            'joint_source_candidate_coverage': 'not_evaluated',
            'limitations': ['Mechanical source replay is not independent human truth.',
                           'Currency normalization remains provisional.',
                           'Exact occurrence order and box association are required; ambiguity is never guessed.']}


def load_source_replay(path, bundle, artifact_dir, source_registry):
    """Read a frozen replay result, closing its structural and input identities.

    This verifies an operator-frozen producer receipt, not authenticity of an
    adversarially rewritten configuration/receipt or independent human review.
    """
    watched = {'replay': path, 'registry': source_registry,
               'corpus_manifest': artifact_dir / 'manifest.json',
               'corpus_chunks': artifact_dir / 'chunks.jsonl',
               'build_receipt': artifact_dir / 'build-receipt.json'}
    before = snapshot(watched)
    require(before['replay'] == bundle.experiment.data['input_sha256']['source_replay'],
            'source replay SHA-256 differs from frozen experiment')
    receipt = read_completed_replay(path)
    keys = {'schema_version', 'status', 'scope', 'experiment_binding_sha256', 'input_sha256', 'source_files',
            'ordered_case_ids', 'counts', 'corpus_profile', 'chunk_ids', 'cells', 'rationale_bindings', 'source_binding_passed',
            'human_gold_count', 'generation_runs', 'joint_source_candidate_coverage', 'limitations'}
    require(type(receipt) is dict and set(receipt) == keys and type(receipt['schema_version']) is int
            and receipt['schema_version'] == 2 and receipt['status'] == 'complete'
            and receipt['scope'] == 'operator-local-source-candidate-replay', 'source replay schema differs')
    require(receipt['experiment_binding_sha256'] == experiment_binding_sha256(bundle.experiment.data)
            and receipt['ordered_case_ids'] == [case.case_id for case in bundle.cases]
            and receipt['counts'] == plain(bundle.counts), 'source replay case/config binding differs')
    require(type(receipt['human_gold_count']) is int and receipt['human_gold_count'] == 0
            and type(receipt['generation_runs']) is int and receipt['generation_runs'] == 0
            and receipt['joint_source_candidate_coverage'] == 'not_evaluated', 'source replay evidence boundary differs')
    require(type(receipt['input_sha256']) is dict and set(receipt['input_sha256']) == {
        'registry', 'cases', 'case_freeze', 'corpus_manifest', 'corpus_chunks', 'build_receipt'},
        'source replay input identity schema differs')
    for name in ('registry', 'corpus_manifest', 'corpus_chunks', 'build_receipt'):
        require(receipt['input_sha256'][name] == before[name], 'source replay consumed input differs')
    require(receipt['input_sha256']['cases'] == bundle.cases_sha256
            and receipt['input_sha256']['case_freeze'] == bundle.freeze_sha256, 'source replay case bytes differ')
    configured = bundle.experiment.data['documents']
    require(type(receipt['source_files']) is list and len(receipt['source_files']) == len(configured),
            'source replay registered file count differs')
    for source, doc in zip(receipt['source_files'], configured, strict=True):
        require(type(source) is dict and set(source) == {'document_id', 'pdf_sha256', 'source_bytes'}
                and source['document_id'] == doc['document_id'] and source['pdf_sha256'] == doc['pdf_sha256']
                and type(source['source_bytes']) is int and 0 < source['source_bytes'] <= 256_000_000,
                'source replay registered PDF differs')
    corpus = load_corpus(artifact_dir=artifact_dir, source_registry=source_registry)
    require(receipt['chunk_ids'] == [chunk.chunk_id for chunk in corpus.chunks], 'source replay chunk inventory differs')
    referenced = {key for case in bundle.cases for key in case.expected['rationale_binding_ids']}
    expected_rationales = [{'binding_id': binding['binding_id'], 'kind': binding['kind'],
        'receipt_sha256': binding['receipt_sha256'], 'status': 'candidate-bound'}
        for binding in bundle.experiment.data['rationale_bindings'] if binding['binding_id'] in referenced]
    require(receipt['rationale_bindings'] == expected_rationales, 'source replay negative rationale bindings differ')
    build = strict_json(read_body(watched['build_receipt']))
    require(receipt['corpus_profile'] == {'library': 'PyMuPDF', 'version': build['runtime']['PyMuPDF'],
        'mode': 'text', 'sort': False, 'normalization': 'strip', 'text_encoding': 'utf-8',
        'offset_unit': 'unicode_code_points', 'interval': 'half_open'}, 'source replay corpus profile differs')
    expected_cells = [(case, cell) for case in bundle.cases for cell in case.source_cells]
    require(type(receipt['cells']) is list and len(receipt['cells']) == len(expected_cells), 'source replay cell count differs')
    row_fields = {'case_id', 'cell_id', 'document_id', 'pdf_sha256', 'physical_page', 'association_receipt_sha256',
                  'source_profile', 'source_page_text_sha256', 'corpus_page_text_sha256', 'corpus_page_text_chars',
                  'source_binding_passed', 'essential_roles', 'corpus_spans', 'chunk_mappings'}
    for row, (case, cell) in zip(receipt['cells'], expected_cells, strict=True):
        require(type(row) is dict and set(row) == row_fields and row['case_id'] == case.case_id,
                'source replay cell schema/order differs')
        for name in ('cell_id', 'document_id', 'pdf_sha256', 'physical_page'):
            require(type(row[name]) is type(cell[name]) and row[name] == cell[name], 'source replay cell identity differs')
        binding = cell['bindings']
        require(row['association_receipt_sha256'] == binding['association_receipt_sha256']
                and row['source_profile'] == plain(binding['extraction_profile'])
                and row['source_page_text_sha256'] == binding['page_text_sha256'], 'source replay coordinate binding differs')
        require(type(row['source_binding_passed']) is bool and row['source_binding_passed'] == cell['binding_complete'],
                'source replay completeness differs')
        size = row['corpus_page_text_chars']
        require(type(size) is int and 0 < size <= bundle.experiment.data['limits']['max_source_text_chars']
                and type(row['corpus_page_text_sha256']) is str
                and len(row['corpus_page_text_sha256']) == 64
                and all(c in '0123456789abcdef' for c in row['corpus_page_text_sha256']),
                'source replay corpus page identity differs')
        roles = {'statement', 'unit', 'date', 'row', 'value'}
        if cell['period_kind'] == 'duration':
            roles.add('duration')
        if cell['unit_kind'] in ('monetary', 'per_share'):
            roles.add('currency')
        expected_roles = sorted(roles) if row['source_binding_passed'] else []
        require(row['essential_roles'] == expected_roles, 'source replay essential roles differ')
        spans = row['corpus_spans']
        require(type(spans) is dict and spans.get('status') in ('complete', 'not_evaluated')
                and type(spans.get('ranges')) is dict, 'source replay corpus spans differ')
        if spans['status'] == 'complete':
            require(set(spans) == {'status', 'ranges'} and set(spans['ranges']) == set(expected_roles),
                    'source replay essential ranges are incomplete')
            for interval in spans['ranges'].values():
                require(type(interval) is list and len(interval) == 2
                        and all(type(v) is int for v in interval) and 0 <= interval[0] < interval[1] <= size,
                        'source replay corpus interval differs')
        else:
            require(set(spans) == {'status', 'ranges', 'reason'} and spans['ranges'] == {}
                    and type(spans['reason']) is str and len(spans['reason']) <= 200,
                    'source replay unresolved spans differ')
        actual_chunks = [chunk for chunk in corpus.chunks if chunk.document_id == cell['document_id']
                         and chunk.page_number == cell['physical_page']]
        require(type(row['chunk_mappings']) is list and len(row['chunk_mappings']) == len(actual_chunks),
                'source replay same-page chunk count differs')
        for mapping, chunk in zip(row['chunk_mappings'], actual_chunks, strict=True):
            require(type(mapping) is dict and set(mapping) == {'chunk_id', 'status', 'range', 'text_sha256'}
                    and mapping['chunk_id'] == chunk.chunk_id
                    and mapping['text_sha256'] == sha256(chunk.text.encode()).hexdigest()
                    and mapping['status'] in ('unique', 'ambiguous', 'absent'), 'source replay chunk mapping differs')
            interval = mapping['range']
            if mapping['status'] == 'unique':
                require(type(interval) is list and len(interval) == 2 and all(type(v) is int for v in interval)
                        and 0 <= interval[0] < interval[1] <= size and interval[1] - interval[0] == len(chunk.text),
                        'source replay chunk interval differs')
            else:
                require(interval is None, 'unresolved chunk must not acquire a guessed range')
    positive = {cell_id for case in bundle.cases for cell_id in case.expected['cell_ids']}
    passed = bool(positive) and all(row['source_binding_passed'] for row in receipt['cells'] if row['cell_id'] in positive)
    require(type(receipt['source_binding_passed']) is bool and receipt['source_binding_passed'] == passed,
            'source replay overall binding status differs')
    require(snapshot(watched) == before, 'source replay inputs changed while loading')
    require(not any(os.path.lexists(guard_path(path, marker)) for marker in ('pending', 'failure')),
            'source replay completion guard appeared while loading')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'source-registry', 'cases', 'case-freeze', 'artifact-dir',
                 'experiment', 'binding-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    require(output.is_absolute(), 'output requires an absolute local path')
    for key in ('source_dir', 'artifact_dir', 'binding_dir'):
        require(not output.resolve().is_relative_to(args[key].resolve()), 'output overlaps a read-only input directory')
    require(output.resolve() not in {path.resolve() for path in args.values()}, 'output overlaps a read-only input file')
    receipt = replay_sources(**args)
    # Recheck all consumed bodies immediately before and after writing, without
    # a second PDF parse or a second source-coverage computation.
    watch = {'registry': args['source_registry'], 'cases': args['cases'], 'case_freeze': args['case_freeze'],
             'corpus_manifest': args['artifact_dir'] / 'manifest.json',
             'corpus_chunks': args['artifact_dir'] / 'chunks.jsonl',
             'build_receipt': args['artifact_dir'] / 'build-receipt.json', 'experiment': args['experiment']}
    registry = strict_json(read_body(args['source_registry']))
    for source in registry['sources']:
        watch['pdf:' + source['document_id']] = args['source_dir'] / source['filename']
    for cell in receipt['cells']:
        digest = cell['association_receipt_sha256']
        if digest is not None:
            watch['association:' + digest] = args['binding_dir'] / (digest + '.json')
    for binding in receipt['rationale_bindings']:
        digest = binding['receipt_sha256']
        watch['rationale:' + digest] = args['binding_dir'] / (digest + '.json')
    before = snapshot(watch)
    cfg = load_experiment(args['experiment'])
    require(experiment_binding_sha256(cfg.data) == receipt['experiment_binding_sha256'], 'experiment changed after replay')
    for name, expected in receipt['input_sha256'].items():
        require(before[name] == expected, 'input changed after replay')
    for source in receipt['source_files']:
        require(before['pdf:' + source['document_id']] == source['pdf_sha256'], 'source changed after replay')
    for cell in receipt['cells']:
        digest = cell['association_receipt_sha256']
        if digest is not None:
            require(before['association:' + digest] == digest, 'association packet changed after replay')
    for binding in receipt['rationale_bindings']:
        digest = binding['receipt_sha256']
        require(before['rationale:' + digest] == digest, 'rationale packet changed after replay')
    def unchanged():
        require(snapshot(watch) == before, 'input identity changed while closing source replay')
    write_replay_exclusive(output, receipt, unchanged)
    print(json.dumps({'status': 'complete', 'case_count': receipt['counts']['cases'],
                      'source_cell_count': len(receipt['cells']), 'source_binding_passed': receipt['source_binding_passed'],
                      'human_gold_count': 0, 'generation_runs': 0}))


if __name__ == '__main__':
    main()
