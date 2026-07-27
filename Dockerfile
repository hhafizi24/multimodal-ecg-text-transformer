FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/appuser/.cache/huggingface

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchaudio

COPY pyproject.toml README.md ./
COPY configs/ ./configs/
COPY src/ ./src/

RUN pip install --no-cache-dir . \
    && mkdir -p "$HF_HOME" /app/artifacts /app/config \
    && chown -R appuser:appuser /home/appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).close()"

CMD ["uvicorn", "src.serving.serve:app", "--host", "0.0.0.0", "--port", "8000"]