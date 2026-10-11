FROM python:3.13-slim-bookworm AS build
WORKDIR /build
COPY pyproject.toml README.md LICENSE THIRD_PARTY.md ./
COPY src/ ./src/
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY --from=build /wheels/ /wheels/
COPY constraints/api-demo.txt /tmp/api-demo.txt
RUN python -m pip install --no-cache-dir -c /tmp/api-demo.txt \
    "/wheels/nvidia_financial_rag-0.1.0-py3-none-any.whl[api]" \
    && python -m pip check
WORKDIR /app
COPY --chown=10001:10001 examples/serve_demo.py /app/serve_demo.py
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=3).close()"
CMD ["python", "-I", "-B", "/app/serve_demo.py", "--host", "0.0.0.0", "--port", "8000"]
