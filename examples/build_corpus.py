"""Build a new local corpus from approved, pinned operator PDFs; no acquisition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nvidia_rag.corpus import build_local_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source-registry", "source-dir", "output-dir", "tokenizer-cache-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_local_corpus(**vars(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
