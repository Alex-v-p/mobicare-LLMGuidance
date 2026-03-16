# Testing

## Commands

Run everything:

```bash
pytest tests
```
```bash
pytest -vv
```

Run only unit tests:

```bash
pytest tests/unit
```

Run only route tests:

```bash
pytest tests/unit/api/routes
```

Run integration tests:

```bash
pytest tests/integration
```

Run E2E tests:

```bash
pytest tests/e2e
```

Run a single file with verbose output:

```bash
pytest -v tests/e2e/test_guidance_async_flow.py
```
