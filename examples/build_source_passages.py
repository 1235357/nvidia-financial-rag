"""Build an operator-local source-only field sidecar using the installed package.

Prepare original pinned PDFs and reviewed fiscal declarations separately. This
script does not download, embed, query, generate, serve or rewrite raw corpora.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nvidia_rag.source_passages import build_source_passage_artifact
from nvidia_rag.statement_startup import load_statement_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-registry', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--fiscal-declarations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--document-id', action='append', default=None,
                        help='Optional declared report subset; repeat once per unique document ID.')
    args = parser.parse_args()
    if args.document_id is not None and len(args.document_id) != len(set(args.document_id)):
        parser.error('duplicate --document-id scope')
    snapshot = load_statement_snapshot(source_registry=args.source_registry,
        source_dir=args.source_dir, fiscal_declarations=args.fiscal_declarations,
        document_ids=None if args.document_id is None else frozenset(args.document_id))
    receipt = build_source_passage_artifact(snapshot=snapshot, output_dir=args.output_dir)
    print(json.dumps({'status':receipt['status'], 'passage_count':receipt['passage_count'],
        'passages_sha256':receipt['passages_sha256'], 'model_runs':0, 'embedding_runs':0,
        'human_acceptance':False}, sort_keys=True))


if __name__ == '__main__':
    main()
