"""Synthetic, hand-checkable learning lab. No financial answer or model benchmark."""
from nvidia_rag.core import Chunk
from nvidia_rag.retrieval import DenseRetriever
from nvidia_rag.service import EvidenceService


def run_lab() -> dict:
    chunks = [
        Chunk('wrong-period', 'synthetic-fy2024', 1, 'Synthetic revenue row: FY2024.'),
        Chunk('requested-period', 'synthetic-fy2025', 2, 'Synthetic revenue row: FY2025.'),
    ]
    calls = []

    def encode_query(texts):
        calls.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    # Unit vectors: manual dot products with [1, 0] are 1 and 0.8.
    retriever = DenseRetriever(chunks, query_encoder=encode_query,
                               precomputed_embeddings=[[1, 0], [0.8, 0.6]])
    service = EvidenceService(retriever=retriever, threshold=0.9, top_k=2)
    try:
        service.query('   ')
    except ValueError:
        pass
    else:
        raise AssertionError('Blank input should be rejected')
    assert calls == [], 'Input rejection must happen before query encoding'
    blank_input_encoder_calls = len(calls)
    question = 'What was revenue in FY2025?'
    trace = service.inspect(question)
    assert [c.chunk.chunk_id for c in trace.retrieval.candidates] == [
        'wrong-period', 'requested-period']
    assert trace.context.can_answer is True
    assert trace.context.evidence[0].chunk.document_id == 'synthetic-fy2024'
    # A perpendicular query gives both scores below 0.9.
    refusing_retriever = DenseRetriever(chunks, query_encoder=lambda _: [[0, 1]],
                                       precomputed_embeddings=[[1, 0], [0.8, 0.6]])
    refusal = EvidenceService(retriever=refusing_retriever, threshold=0.9,
                              top_k=2).query(question)
    assert refusal.can_answer is False and refusal.evidence == ()
    return {'scores': [round(c.score, 3) for c in trace.retrieval.candidates],
            'selected_document': trace.context.evidence[0].chunk.document_id,
            'similarity_gate_passed': trace.context.can_answer,
            'requested_period_supported': all(
                item.chunk.document_id == 'synthetic-fy2025'
                for item in trace.context.evidence),
            'below_threshold_refused': not refusal.can_answer,
            'blank_input_encoder_calls': blank_input_encoder_calls}


if __name__ == '__main__':
    import json
    print('SYNTHETIC LEARNING LAB - not NVIDIA financial evidence')
    print(json.dumps(run_lab(), indent=2))
