"""Real loopback transport acceptance; optional cached CUDA index, no quality claim."""
import argparse
from pathlib import Path
import secrets
import socket
import threading
import time
import httpx
import uvicorn
import nvidia_rag
from nvidia_rag.api import create_app
from nvidia_rag.core import Chunk, ScoredChunk
from nvidia_rag.service import EvidenceService

assert 'site-packages' in Path(nvidia_rag.__file__).resolve().parts

class Retriever:
    def retrieve(self, query, *, top_k):
        return [ScoredChunk(Chunk('synthetic:1', 'synthetic', 1,
            'Synthetic transport fixture; not financial evidence.'), 0.9)]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-dir', type=Path, default=None)
args = parser.parse_args()
question = 'synthetic'
expected_document = 'synthetic'
service = EvidenceService(retriever=Retriever(), threshold=0.78, top_k=1)
if args.artifact_dir is not None:
    from sentence_transformers import SentenceTransformer
    from nvidia_rag.ingest import load_chunks
    from nvidia_rag.embedding_store import load_passage_embedding_artifact
    from nvidia_rag.runtime import BGE_MODEL_NAME, BGE_MODEL_REVISION, build_bge_service
    path = args.artifact_dir.resolve(strict=True)
    chunks = load_chunks(path / 'chunks.jsonl')
    vectors = load_passage_embedding_artifact(
        chunks=chunks, chunks_path=path / 'chunks.jsonl',
        embeddings_path=path / 'bge-small-en-v1.5.npy',
        manifest_path=path / 'embedding-manifest.json',
        expected_model_name=BGE_MODEL_NAME, expected_model_revision=BGE_MODEL_REVISION)
    model = SentenceTransformer(BGE_MODEL_NAME, revision=BGE_MODEL_REVISION,
                                device='cuda', local_files_only=True)
    service = build_bge_service(chunks=chunks, model=model, threshold=0.78, top_k=5,
                                route_fiscal_year=True, passage_embeddings=vectors)
    question = 'What was NVIDIA annual revenue in FY2025?'
    expected_document = 'nvidia-fy2025-annual'
token = secrets.token_urlsafe(32)  # ephemeral; never printed or persisted
app = create_app(service=service, api_token=token)
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
    assert server.started and worker.is_alive(), 'API did not start'
    with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5,
                      follow_redirects=False, trust_env=False) as client:
        assert client.get('/healthz').status_code == 200
        assert client.post('/v1/query', json={'question': 'synthetic'}).status_code == 401
        headers = {'Authorization': f'Bearer {token}'}
        assert client.post('/v1/query', headers=headers,
                           json={'question': '   '}).status_code == 422
        response = client.post('/v1/query', headers=headers,
                               json={'question': question})
        assert response.status_code == 200
        payload = response.json()
        assert payload['evidence'][0]['document_id'] == expected_document
        if args.artifact_dir is not None:
            assert payload['evidence'][0]['chunk_id'] == 'nvidia-fy2025-annual:p18:50cdb6ce95041a87'
            print('Observed real-index HTTP top evidence:', payload['evidence'][0]['chunk_id'])
        assert payload['answer'] == ''  # evidence-only response, not generated truth
        assert response.headers['X-Request-ID'] == payload['request_id']
    print('PASS: real loopback HTTP health, 401, 422, evidence response and request ID')
finally:
    server.should_exit = True
    worker.join(timeout=10)
    listener.close()
    assert not worker.is_alive(), 'Temporary API did not stop'
print('PASS: temporary server stopped; no generation, TLS or financial accuracy acceptance')
