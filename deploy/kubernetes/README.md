# Kubernetes production deployment

This directory now targets **production-style Kubernetes only**. Development stays on Docker Compose.

## What changed

The manifests are now organized by responsibility:

- `base/` contains reusable workload, storage, and network policy manifests grouped by component.
- `overlays/production/` contains the production config, pinned image tags, ingress, and bootstrap jobs.
- `kustomization.yaml` at the root points to the production overlay so `kubectl apply -k deploy/kubernetes` still works.

## Key production improvements

- static image tags are centralized in `overlays/production/kustomization.yaml`
- explicit CPU and memory requests/limits were added to every workload
- PodDisruptionBudgets were added for all long-running workloads
- namespace-wide default deny network policies were added, then opened back up only where needed
- a production ingress with TLS termination is included
- secrets and TLS certificates are no longer stored in the deploy tree
- service account tokens are no longer auto-mounted into the pods
- pod/container security settings now use runtime-default seccomp and drop Linux capabilities where practical

## Before first deploy

### 1. Review production config

Update the production ConfigMap values in:

```bash
deploy/kubernetes/overlays/production/config/app-config.yaml
```

At minimum, check:

- `AUTH_VALIDATION_URL`
- `OLLAMA_MODEL`
- `OLLAMA_EMBEDDING_MODEL`
- storage sizing
- image tags in `overlays/production/kustomization.yaml`

### 2. Create the runtime secret out of band

The deployment expects a pre-created secret named `llmguidance-secrets` in the `llmguidance` namespace.

Example:

```bash
kubectl create namespace llmguidance --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic llmguidance-secrets   -n llmguidance   --from-literal=MINIO_ROOT_USER='replace-me'   --from-literal=MINIO_ROOT_PASSWORD='replace-me'   --from-literal=JWT_SECRET_KEY='replace-me'   --from-literal=INTERNAL_SERVICE_TOKEN='replace-me'
```

### 3. Create the TLS secret out of band

The ingress expects a TLS secret named `llmguidance-api-tls` in the same namespace.

Example with existing cert/key files:

```bash
kubectl create secret tls llmguidance-api-tls   -n llmguidance   --cert=/path/to/fullchain.pem   --key=/path/to/privkey.pem
```

If you use cert-manager, create a `Certificate` for the same secret name instead.

## Deploy the stack

Apply the main production manifests:

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

Run the bootstrap jobs:

```bash
kubectl apply -k deploy/kubernetes/overlays/production/bootstrap
kubectl wait --for=condition=complete job/inference-bootstrap-minio -n llmguidance --timeout=10m
kubectl wait --for=condition=complete job/ollama-pull-models -n llmguidance --timeout=20m
```

Then wait for the application workloads:

```bash
kubectl rollout status deployment/inference -n llmguidance
kubectl rollout status deployment/inference-worker -n llmguidance
kubectl rollout status deployment/api -n llmguidance
```

## Access and operations

Port-forward the API if you need an admin tunnel:

```bash
kubectl port-forward svc/api 8000:8000 -n llmguidance
```

Port-forward the MinIO console when needed:

```bash
kubectl port-forward svc/minio-console 9001:9001 -n llmguidance
```

## Operational notes

- the PodDisruptionBudgets protect singleton workloads from voluntary evictions, which is good for production safety but can block node drains until you temporarily relax the PDB or scale out
- the network policies assume your workloads only need intra-namespace traffic, DNS, and outbound HTTP/HTTPS; add more egress rules if your environment needs extra ports
- the ingress host is currently `llmguidance.example.com`; replace it with your real production hostname
- the ingress class is currently `nginx`; change it if your cluster uses a different ingress controller
