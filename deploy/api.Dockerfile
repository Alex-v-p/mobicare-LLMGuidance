FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what we need for the API runtime
COPY services/shared/src /app/services/shared/src
COPY services/api/src /app/services/api/src

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/services/api/src:/app/services/shared/src

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2 \
    python-multipart==0.0.20 \
    "redis>=5,<6" \
    "minio>=7,<8"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
