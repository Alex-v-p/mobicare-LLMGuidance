# Deployment notes

## Reverse proxy layout

The Docker Compose stack now places an Nginx reverse proxy in front of the API service.

### Development

```bash
docker compose up --build
```

Public traffic:
- `http://localhost:8000` -> Nginx -> `api:8000`

Direct service ports remain exposed in development for debugging:
- inference: `http://localhost:8001`
- qdrant: `http://localhost:6333`
- minio console: `http://localhost:9001`
- ollama: `http://localhost:11434`

### Production-like

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

In this mode only the Nginx front door is published. The API, inference, Qdrant, MinIO, Redis, and Ollama containers are reachable only on the Docker network.

### Optional TLS with Nginx

Put certificates here:

- `deploy/nginx/certs/fullchain.pem`
- `deploy/nginx/certs/privkey.pem`

Then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.prod.tls.yml up -d --build
```

Default ports:
- `http://localhost:8080` redirects to HTTPS
- `https://localhost:8443` serves the gateway

## Ollama models

This project supports:
- `OLLAMA_MODEL` for the default generation model
- `OLLAMA_EMBEDDING_MODEL` for the default embedding model
- per-request `llm_model` and `embedding_model` overrides

Pull the models you want before using them:

```bash
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose exec ollama ollama pull qwen3-embedding:4b
```

After changing the embedding model, re-run ingestion so Qdrant contains embeddings produced by the same model.
