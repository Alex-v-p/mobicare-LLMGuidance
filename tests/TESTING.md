Run the full suite:
pytest -vv -l --tb=short

Run with coverage:
pytest --cov=services/api/src/api --cov=services/inference/src/inference --cov=services/shared/src/shared --cov-report=term-missing -vv

Run only fast unit and route tests:
pytest -m "unit or route" -vv

Run integration tests:
pytest -m integration -vv

Run end-to-end tests:
pytest -m e2e -vv

Run the optional real-container integration tests only:
pytest tests/integration/test_real_redis_job_store.py tests/integration/test_real_minio_document_repository.py -vv

The real-container tests require Docker and will skip automatically when Docker is unavailable.
