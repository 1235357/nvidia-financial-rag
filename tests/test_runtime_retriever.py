"""The evaluator and HTTP service share BGE conventions, not evidence gating."""
import numpy as np
import pytest

from nvidia_rag import runtime
from nvidia_rag.core import Chunk, ScoredChunk


def test_public_retriever_preserves_query_input_and_all_ranked_candidates():
    factory = getattr(runtime, 'build_bge_retriever', None)
    assert callable(factory), 'Evaluation needs a public raw BGE retriever factory'
    calls = []

    class Model:
        def encode(self, texts, **kwargs):
            calls.append((list(texts), kwargs))
            return np.array([[0., 1.]])

    chunks = [Chunk('one', 'report', 1, 'First passage'),
              Chunk('two', 'report', 2, 'Second passage')]
    retriever = factory(chunks=chunks, model=Model(),
                        passage_embeddings=np.array([[1., 0.], [0., 1.]]))
    query = '  exact original question\n'
    candidates = retriever.retrieve(query, top_k=2)
    assert [row.chunk.chunk_id for row in candidates] == ['two', 'one']
    assert [row.score for row in candidates] == [1., 0.]
    assert calls == [([runtime.BGE_QUERY_INSTRUCTION + query], {
        'normalize_embeddings': True, 'batch_size': 32, 'show_progress_bar': False})]


def test_service_delegates_construction_to_public_factory(monkeypatch):
    chunk = Chunk('one', 'report', 1, 'Evidence')
    construction = []
    queries = []

    class Retriever:
        def retrieve(self, query, *, top_k):
            queries.append((query, top_k))
            return [ScoredChunk(chunk, .7)]

    def factory(**kwargs):
        construction.append(kwargs)
        return Retriever()

    monkeypatch.setattr(runtime, 'build_bge_retriever', factory, raising=False)
    model = object()
    service = runtime.build_bge_service(chunks=[chunk], model=model, threshold=.78, top_k=3)
    result = service.query(' original ')
    assert len(construction) == 1, 'HTTP service must use the shared public construction boundary'
    assert construction[0]['chunks'] == [chunk]
    assert construction[0]['model'] is model
    assert 'threshold' not in construction[0] and 'top_k' not in construction[0]
    assert queries == [('original', 3)]
    assert result.evidence == () and not result.can_answer


@pytest.mark.parametrize('options', [
    {'source_task': 'annual', 'route_fiscal_year': True},
    {'source_task': 'annual'},
    {'report_catalog': object(), 'route_fiscal_year': True},
])
def test_routing_conflicts_fail_before_encoding(options):
    factory = getattr(runtime, 'build_bge_retriever', None)
    assert callable(factory), 'Evaluation needs public routing validation before model calls'

    class ForbiddenModel:
        def encode(self, *args, **kwargs):
            raise AssertionError('Invalid routing must fail before encoding')

    with pytest.raises(ValueError):
        factory(chunks=[Chunk('one', 'report', 1, 'Evidence')], model=ForbiddenModel(), **options)
