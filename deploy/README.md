# Deployment notes

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
