# Kubernetes deployment

This directory contains a small-cluster Kubernetes deployment for the full stack:

- API
- inference HTTP service
- inference worker
- Redis
- Qdrant
- MinIO
- Ollama

It is aimed at getting the project running on Kubernetes first. It is not a final production hardening pass yet.

## Images used

The manifests expect these application images to exist:

- `mobicare-llm/api:local`
- `mobicare-llm/inference-http:local`
- `mobicare-llm/inference-worker:local`

## Build images

From the repository root:

```bash
docker build -f deploy/api.Dockerfile -t mobicare-llm/api:local .
docker build -f deploy/inference.http.Dockerfile -t mobicare-llm/inference-http:local .
docker build -f deploy/inference.worker.Dockerfile -t mobicare-llm/inference-worker:local .
```

If your cluster cannot see your local Docker images directly, load or push them before deploying.

Example for `kind`:

```bash
kind load docker-image mobicare-llm/api:local
kind load docker-image mobicare-llm/inference-http:local
kind load docker-image mobicare-llm/inference-worker:local
```

## Deploy the stack

Apply the main manifests:

```bash
kubectl apply -k deploy/kubernetes
```

Wait for the stateful infrastructure:

```bash
kubectl rollout status statefulset/redis -n llmguidance
kubectl rollout status statefulset/qdrant -n llmguidance
kubectl rollout status statefulset/minio -n llmguidance
kubectl rollout status statefulset/ollama -n llmguidance
```

Run the one-shot bootstrap jobs:

```bash
kubectl apply -f deploy/kubernetes/bootstrap-jobs.yaml
kubectl wait --for=condition=complete job/inference-bootstrap-minio -n llmguidance --timeout=10m
kubectl wait --for=condition=complete job/ollama-pull-models -n llmguidance --timeout=20m
```

Then wait for the application workloads:

```bash
kubectl rollout status deployment/inference -n llmguidance
kubectl rollout status deployment/inference-worker -n llmguidance
kubectl rollout status deployment/api -n llmguidance
```

## Access the API locally

Port-forward the API service:

```bash
kubectl port-forward svc/api 8000:8000 -n llmguidance
```

The API will then be reachable at:

- `http://localhost:8000`
- `http://localhost:8000/docs` in `APP_ENV=dev`

Port-forward the MinIO console if needed:

```bash
kubectl port-forward svc/minio-console 9001:9001 -n llmguidance
```

## Optional ingress

If you already have an NGINX ingress controller installed, you can also expose the API with:

```bash
kubectl apply -f deploy/kubernetes/ingress.optional.yaml
```

That ingress carries over the important reverse-proxy behavior from the Docker Compose gateway:

- 64 MB request body limit
- 300 second proxy read timeout
- 300 second proxy send timeout

## Re-running bootstrap jobs

The jobs are intentionally separate from normal service startup.
If you want to run them again, delete them first and re-apply:

```bash
kubectl delete job inference-bootstrap-minio ollama-pull-models -n llmguidance --ignore-not-found
kubectl apply -f deploy/kubernetes/bootstrap-jobs.yaml
```

## Configuration

Edit these files before moving to a shared or production-like cluster:

- `deploy/kubernetes/configmap.yaml`
- `deploy/kubernetes/secret.yaml`

Important notes:

- the default manifests run with `APP_ENV=dev`
- `secret.yaml` contains placeholders for production-sensitive values
- the storage classes are left unspecified so your cluster default can be used
- the Ollama workload is CPU-only for now; add GPU scheduling later if you move it to GPU nodes
