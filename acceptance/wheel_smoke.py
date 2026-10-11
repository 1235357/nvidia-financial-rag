"""Consumer smoke check. Synthetic evidence is not financial validation."""
from pathlib import Path
import sys
import nvidia_rag
from nvidia_rag.core import Chunk, ScoredChunk
from nvidia_rag.service import EvidenceService

package_path = Path(nvidia_rag.__file__).resolve()
assert 'site-packages' in package_path.parts, package_path
assert sys.prefix != sys.base_prefix, 'Run in the isolated acceptance environment'

class ConsumerRetriever:
    def retrieve(self, query, *, top_k):
        assert query == 'synthetic question'
        return [ScoredChunk(Chunk('synthetic:1', 'synthetic', 1,
                                  'Synthetic passage; not a financial claim.'), 0.9)]

service = EvidenceService(retriever=ConsumerRetriever(), threshold=0.78, top_k=1)
result = service.query(' synthetic question ')
assert len(result.evidence) == 1
assert result.evidence[0].chunk.document_id == 'synthetic'
assert result.can_answer is True
try:
    service.query('   ')
except ValueError:
    pass
else:
    raise AssertionError('Blank query was not rejected')
print('Installed package:', package_path)
print('PASS: injected consumer retrieval and blank-query rejection')
