"""Small consumer example. No model imports, credentials or automatic retries."""
from __future__ import annotations

import argparse
import json
import os
from urllib.parse import urlsplit

import httpx


def validate_base_url(value: str) -> str:
    """Permit local development HTTP; remote credentials require HTTPS."""
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError('use an HTTP(S) base URL without credentials, query or fragment')
    if parsed.scheme == 'http' and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('remote API connections require HTTPS; HTTP is loopback-only')
    return value


def query_api(client, *, token: str, question: str) -> dict:
    response = client.post('/v1/query',
        headers={'Authorization': f'Bearer {token}'}, json={'question': question})
    response.raise_for_status()
    return response.json()


def query_statement_candidates_api(
    client, *, token: str, question: str, document_ids: list[str] | None = None
) -> dict:
    """Inspect source candidates; preserve original wording and explicit scope."""
    payload = {'question': question}
    if document_ids is not None:
        payload['document_ids'] = document_ids
    response = client.post('/v1/statement-candidates',
        headers={'Authorization': f'Bearer {token}'}, json=payload)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--question', required=True)
    parser.add_argument('--statement-candidates', action='store_true',
                        help='Inspect source-bound income cells instead of the ordinary query route')
    parser.add_argument('--document-id', action='append', default=None,
                        help='Repeat to select registered report IDs in statement mode')
    args = parser.parse_args()
    try:
        base_url = validate_base_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    if args.document_id is not None and not args.statement_candidates:
        parser.error('document IDs require --statement-candidates')
    token = os.environ.get('NVIDIA_RAG_API_TOKEN', '')
    if not token:
        parser.error('configure NVIDIA_RAG_API_TOKEN privately; do not pass it as an argument')
    with httpx.Client(base_url=base_url, timeout=30.0, follow_redirects=False) as client:
        if args.statement_candidates:
            payload = query_statement_candidates_api(client, token=token,
                question=args.question, document_ids=args.document_id)
        else:
            payload = query_api(client, token=token, question=args.question)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
