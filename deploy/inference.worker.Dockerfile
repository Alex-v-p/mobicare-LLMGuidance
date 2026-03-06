FROM python:3.12-slim

WORKDIR /app

COPY services/shared/src /app/services/shared/src
COPY services/inference/src /app/services/inference/src

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/services/inference/src:/app/services/shared/src

RUN pip install --no-cache-dir \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    pydantic-settings==2.5.2 \
    "redis>=5,<6" \
    "minio>=7,<8"

CMD ["python", "-m", "inference.worker.runtime.main"]
