"""Complete archives can exceed one body; bounds never truncate source evidence."""
from hashlib import sha256
import json

import pytest

from test_evaluation_run import case, library


def test_multiple_complete_bodies_can_exceed_legacy_64_mib_total(tmp_path):
    module = library()
    writer = module._Writer(tmp_path)
    body = b'complete-source-body\n' * (2 * 1024 * 1024)
    assert 32 * 1024 * 1024 < len(body) < 64 * 1024 * 1024
    for name in ('first.pdf', 'second.pdf'):
        identity = writer.body(name, body)
        assert identity == {'bytes': len(body), 'sha256': sha256(body).hexdigest()}
        assert (tmp_path / name).read_bytes() == body
    assert writer.written == 2 * len(body) > 64 * 1024 * 1024


def test_body_limits_stay_separate_for_reads_writes_and_attempt_updates(tmp_path, monkeypatch):
    module = library()
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', 32)
    monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', 1024)
    writer = module._Writer(tmp_path)
    writer.body('exact.bin', b'x' * 32)
    assert module._read(tmp_path / 'exact.bin') == b'x' * 32
    (tmp_path / 'over.bin').write_bytes(b'x' * 33)
    with pytest.raises(ValueError, match='budget'):
        module._read(tmp_path / 'over.bin')
    with pytest.raises(ValueError, match='budget'):
        writer.body('never-written.bin', b'x' * 33)
    assert not (tmp_path / 'never-written.bin').exists()
    writer.progress({'value': 'small'})
    before = (tmp_path / 'attempts.json').read_bytes()
    with pytest.raises(ValueError, match='budget'):
        writer.progress({'value': 'x' * 33})
    assert (tmp_path / 'attempts.json').read_bytes() == before


def test_complete_writer_budget_accepts_exact_boundary_then_refuses(tmp_path, monkeypatch):
    module = library()
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', 32)
    monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', 64)
    writer = module._Writer(tmp_path)
    writer.body('a', b'a' * 32)
    writer.body('b', b'b' * 32)
    with pytest.raises(ValueError, match='budget'):
        writer.body('c', b'c')
    assert writer.written == 64 and not (tmp_path / 'c').exists()


def test_snapshot_preflight_refuses_aggregate_before_reading_any_body(tmp_path, monkeypatch):
    module = library()
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', 32)
    monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', 63)
    paths = {}
    for name in ('a', 'b'):
        paths[name] = tmp_path / name
        paths[name].write_bytes(b'x' * 32)
    monkeypatch.setattr(module, '_read', lambda *a, **k: pytest.fail('oversized archive was read'))
    with pytest.raises(ValueError, match='budget'):
        module._capture_bodies(paths)


def test_completed_loader_refuses_aggregate_before_reading_run_bodies(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path)
    module.run_evaluation(**options)
    total = sum(path.stat().st_size for path in options['output_dir'].rglob('*') if path.is_file())
    monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', total - 1)
    monkeypatch.setattr(module, '_read', lambda *a, **k: pytest.fail('oversized run was read'))
    with pytest.raises(ValueError, match='budget'):
        module.load_completed_run(options['output_dir'])


def test_complete_graph_loads_when_total_is_larger_than_each_body_bound(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path)
    largest = max(path.stat().st_size for path in module._package_files().values())
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', largest + 1024)
    module.run_evaluation(**options)
    total = sum(path.stat().st_size for path in options['output_dir'].rglob('*') if path.is_file())
    assert total > module.MAX_BODY_BYTES
    loaded = module.load_completed_run(options['output_dir'])
    assert loaded['receipt']['status'] == 'complete'
    assert loaded['results']['calls'][0]['query'] == '  Revenue?  '


def test_larger_than_64_mib_graph_round_trips_and_imports_unknown_review(tmp_path, monkeypatch):
    from nvidia_rag.evaluation_review import import_review
    module = library()
    _, options = case(tmp_path)
    package = module._package_files()
    # Opaque Python comment snapshots exercise real large-file capture without
    # a native model, generated financial labels, or modifying installed code.
    padding = b'# synthetic captured source comment ' + b'x' * (34 * 1024 * 1024) + b'\n'
    for name in ('large_a.py', 'large_b.py'):
        path = tmp_path / name
        path.write_bytes(padding)
        package[name] = path
    monkeypatch.setattr(module, '_package_files', lambda: package)
    module.run_evaluation(**options)
    output = options['output_dir']
    assert sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) > 64 * 1024 * 1024
    loaded = module.load_completed_run(output)
    for name in ('large_a.py', 'large_b.py'):
        assert (output / 'inputs' / 'package' / name).read_bytes() == padding
    assert loaded['receipt']['status'] == 'complete'
    review = tmp_path / 'unknown-review.json'
    review.write_bytes((output / 'review-packet.json').read_bytes())
    destination = tmp_path / 'imported-unknown.json'
    import_review(run_dir=output, review_path=review, output_path=destination)
    summary = json.loads(destination.read_bytes())['summary']
    assert summary['unjudged_count'] == summary['retrieved_unique_count'] == 2
    assert summary['judgment_coverage'] == 0


def test_dynamic_oversize_packet_preserves_failure_and_complete_attempts(tmp_path, monkeypatch):
    from nvidia_rag import evaluation_review
    module = library()
    _, options = case(tmp_path)
    largest = max(path.stat().st_size for path in module._package_files().values())
    limit = largest + 1024
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', limit)
    monkeypatch.setattr(evaluation_review, 'build_review_packet', lambda *a, **k: {'oversize': 'x' * limit})
    with pytest.raises(ValueError, match='budget'):
        module.run_evaluation(**options)
    output = options['output_dir']
    assert (output / 'pending.json').is_file() and (output / 'failure.json').is_file()
    assert not (output / 'review-packet.json').exists()
    assert not (output / 'receipt.json').exists()
    attempts = json.loads((output / 'attempts.json').read_bytes())['attempts']
    assert attempts[0]['status'] == 'complete' and len(attempts[0]['candidates']) == 2
    assert [candidate['rank'] for candidate in attempts[0]['candidates']] == [1, 2]


def test_loader_refuses_an_oversize_body_before_reading_any_body(tmp_path, monkeypatch):
    module = library()
    _, options = case(tmp_path)
    module.run_evaluation(**options)
    largest = max(path.stat().st_size for path in options['output_dir'].rglob('*') if path.is_file())
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', largest - 1)
    monkeypatch.setattr(module, '_read', lambda *a, **k: pytest.fail('oversized run was read'))
    with pytest.raises(ValueError, match='budget'):
        module.load_completed_run(options['output_dir'])


@pytest.mark.parametrize('oversize', ['corpus', 'vectors', 'combined'])
def test_known_input_budget_is_checked_before_corpus_or_vector_loading(tmp_path, monkeypatch, oversize):
    from nvidia_rag import corpus, corpus_embeddings
    module = library()
    artifact, options = case(tmp_path, backend='cached-bge' if oversize == 'vectors' else 'lexical')
    largest = max(path.stat().st_size for path in module._package_files().values())
    limit = largest + 1024
    monkeypatch.setattr(module, 'MAX_BODY_BYTES', limit)
    if oversize == 'vectors':
        vectors, cache = tmp_path / 'vectors', tmp_path / 'cache'
        vectors.mkdir()
        cache.mkdir()
        for name in module.VECTOR_FILES:
            (vectors / name).write_bytes(b'x' * (limit + 1 if name.endswith('.npy') else 1))
        options.update(vector_dir=vectors, model_cache_dir=cache)
        monkeypatch.setattr(corpus_embeddings, 'load_local_bge_vectors',
            lambda **k: pytest.fail('oversized vectors were loaded'))
    else:
        if oversize == 'corpus':
            (artifact.directory / 'chunks.jsonl').write_bytes(b'x' * (limit + 1))
        else:
            monkeypatch.setattr(module, 'MAX_OUTPUT_BYTES', 1)
        monkeypatch.setattr(corpus, 'load_verified_corpus',
            lambda **k: pytest.fail('oversized known inputs were loaded'))
    with pytest.raises(ValueError, match='budget'):
        module.run_evaluation(**options)
    assert not options['output_dir'].exists()
