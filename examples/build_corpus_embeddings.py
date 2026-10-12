"""Build separate pinned BGE vectors using a prepared installed wheel and local cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nvidia_rag.corpus_embeddings import build_cached_local_bge_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("artifact-dir", "source-registry", "vector-dir", "wheel"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_cached_local_bge_vectors(**vars(args))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
