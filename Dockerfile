FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code changes do not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY screengate/ ./screengate/
COPY bench/ ./bench/
COPY ui/ ./ui/

# Run as a non-root user. The gateway handles untrusted input by design,
# so the blast radius of a container escape is worth minimising.
RUN useradd --create-home --uid 10001 screengate \
    && chown -R screengate:screengate /app
USER screengate

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "screengate.app:app", "--host", "0.0.0.0", "--port", "8000"]
