# On-Premise RAG Document Assistant System - Docker image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so the layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the docrag package
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# Run as a non-root user; uploaded PDFs are stored on a mounted volume
RUN useradd --create-home appuser \
    && mkdir -p /data/uploads \
    && chown appuser /data/uploads
USER appuser

EXPOSE 8000
CMD ["uvicorn", "docrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
