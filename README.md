# mobicare-LLMGuidance

A multi-service Python project for document ingestion, retrieval, and LLM-based guidance generation.

## What is in this repository

- **Gateway API**: document management and external-facing HTTP endpoints
- **Inference HTTP service**: guidance generation and retrieval endpoints
- **Inference worker**: asynchronous job execution and ingestion orchestration
- **Shared package**: common models and configuration used by multiple services
- **Infrastructure**: Docker Compose setup for MinIO, Qdrant, Ollama, and the app services

## Project structure

```text
services/
  api/                Gateway/API service
  inference/          Inference HTTP service and worker runtime
  shared/             Shared config and utilities
scripts/              Repository scripts
deploy/               Dockerfiles for deployable services
infra/                Supporting infrastructure assets
.github/workflows/    CI workflows
```

## Dependency management

This repository uses:

- `pyproject.toml` for package metadata and Python project configuration
- `requirements.lock` for locked runtime dependencies
- `requirements-dev.lock` for locked developer and CI dependencies

### Update lock files

When dependencies change in `pyproject.toml`, refresh the lock files intentionally.

Using `uv`:

```bash
uv pip compile pyproject.toml -o requirements.lock
uv pip compile pyproject.toml --extra dev -o requirements-dev.lock
```

Commit both lock files together with the dependency change.

## Run with Docker Compose

Build and start the stack:

```bash
docker compose up --build
```

Start in detached mode:

```bash
docker compose up -d --build
```

Stop the stack:

```bash
docker compose down
```

## Pull Ollama models

After the stack is up, pull the required models into the Ollama container:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull qwen2.5:0.5b
```

If you use a different embedding or chat model in requests, pull that model too.

## Health endpoints

- API: `http://localhost:8000/health`
- Inference: `http://localhost:8001/health`
- Qdrant: `http://localhost:6333`
- MinIO console: `http://localhost:9001`
- Ollama: `http://localhost:11434`

## Example ingestion requests

Naive chunking:

```json
{
  "options": {
    "cleaning_strategy": "deep",
    "cleaning_params": {},
    "chunking_strategy": "naive",
    "chunking_params": {
      "chunk_size": 300,
      "chunk_overlap": 100
    }
  }
}
```

Page-indexed chunking:

```json
{
  "options": {
    "cleaning_strategy": "deep",
    "chunking_strategy": "page_indexed",
    "chunking_params": {}
  }
}
```

Late chunking:

```json
{
  "options": {
    "cleaning_strategy": "deep",
    "chunking_strategy": "late",
    "chunking_params": {
      "chunk_size": 300,
      "chunk_overlap": 100
    }
  }
}
```

Medical-guideline-oriented ingestion:

```json
{
  "options": {
    "cleaning_strategy": "medical_guideline_deep",
    "cleaning_params": {},
    "chunking_strategy": "naive",
    "chunking_params": {
      "chunk_size": 300,
      "chunk_overlap": 100
    },
    "embedding_model": "qwen3-embedding:4b"
  }
}
```

## Example guidance requests

Basic guidance request:

```json
{
  "request_id": "case-2026-0001",
  "question": "According to the ESC heart failure guidelines, what is the recommended management for a patient with worsening chronic HFrEF who remains symptomatic despite treatment with ACE inhibitor, beta blocker, and MRA?",
  "patient": {
    "values": {
      "age": 67,
      "sex": "male",
      "diagnosis": "chronic heart failure with reduced ejection fraction",
      "ejection_fraction": 30,
      "nyha_class": "III",
      "current_medication": [
        "enalapril",
        "bisoprolol",
        "spironolactone"
      ],
      "blood_pressure_systolic": 110,
      "heart_rate": 78,
      "egfr": 58,
      "recent_hospitalization": true
    }
  },
  "options": {
    "use_retrieval": true,
    "top_k": 5,
    "temperature": 0.2,
    "max_tokens": 300,
    "use_example_response": false,
    "callback_url": "https://example.com/clinical-callback",
    "callback_headers": {
      "Authorization": "Bearer example-token",
      "Content-Type": "application/json"
    }
  }
}
```

Hybrid retrieval request:

```json
{
  "request_id": "case-2026-0002",
  "question": "What should be monitored after starting sacubitril/valsartan in symptomatic HFrEF?",
  "patient": {
    "values": {
      "age": 67,
      "diagnosis": "HFrEF"
    }
  },
  "options": {
    "use_retrieval": true,
    "retrieval_mode": "hybrid",
    "top_k": 4,
    "hybrid_dense_weight": 0.65,
    "hybrid_sparse_weight": 0.35,
    "use_graph_augmentation": true,
    "graph_max_extra_nodes": 2,
    "temperature": 0.2,
    "max_tokens": 300,
    "use_example_response": false
  }
}
```

Extended guidance request:

```json
{
  "request_id": "case-2026-0020",
  "question": "What escalation of therapy should be considered for symptomatic HFrEF despite ACE inhibitor and beta blocker therapy?",
  "patient": {
    "values": {
      "age": 72,
      "diagnosis": "HFrEF",
      "ejection_fraction": 32
    }
  },
  "options": {
    "use_retrieval": true,
    "top_k": 4,
    "temperature": 0.2,
    "max_tokens": 256,
    "retrieval_mode": "hybrid",
    "llm_model": "qwen2.5:3b-instruct",
    "embedding_model": "qwen3-embedding:4b",
    "enable_query_rewriting": true,
    "enable_response_verification": true,
    "enable_regeneration": true,
    "max_regeneration_attempts": 2
  }
}
```

## Running tests and checks

Run tests:

```bash
pytest
```

Run Pants lint:

```bash
pants lint ::
```

Run Pylint manually:

```bash
PYTHONPATH=services/api/src:services/inference/src:services/shared/src pylint $(git ls-files '*.py')
```

Validate Docker Compose:

```bash
docker compose config
```
