FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/shared/src /app/services/shared/src
COPY services/inference/src /app/services/inference/src

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/services/inference/src:/app/services/shared/src

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2

EXPOSE 8001
CMD ["uvicorn", "inference.http.main:app", "--host", "0.0.0.0", "--port", "8001"]
