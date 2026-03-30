#!/usr/bin/env bash
set -Eeuo pipefail

COMMAND="${1:-status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="llmguidance"
APP_IMAGES=(
  "deploy/api.Dockerfile|mobicare-llm/api:local"
  "deploy/inference.http.Dockerfile|mobicare-llm/inference-http:local"
  "deploy/inference.worker.Dockerfile|mobicare-llm/inference-worker:local"
)

cd "$ROOT_DIR"

kube_context() {
  kubectl config current-context
}

build_app_images() {
  local spec dockerfile image
  for spec in "${APP_IMAGES[@]}"; do
    dockerfile="${spec%%|*}"
    image="${spec##*|}"
    echo "Building ${image}..."
    docker build -f "$dockerfile" -t "$image" .
  done
}

load_app_images() {
  local context
  context="$(kube_context)"
  echo "Active Kubernetes context: ${context}"

  if [[ "$context" == "minikube" ]]; then
    local spec image
    for spec in "${APP_IMAGES[@]}"; do
      image="${spec##*|}"
      echo "Loading ${image} into minikube..."
      minikube image load "$image"
    done
    return
  fi

  if [[ "$context" == kind-* ]]; then
    local spec image
    for spec in "${APP_IMAGES[@]}"; do
      image="${spec##*|}"
      echo "Loading ${image} into kind..."
      kind load docker-image "$image"
    done
    return
  fi

  echo "No automatic image loader configured for context '${context}'."
  echo "If your cluster cannot see local Docker images directly, push them to a registry or load them manually."
}

wait_infra() {
  kubectl rollout status statefulset/redis -n "$NAMESPACE"
  kubectl rollout status statefulset/qdrant -n "$NAMESPACE"
  kubectl rollout status statefulset/minio -n "$NAMESPACE"
  kubectl rollout status statefulset/ollama -n "$NAMESPACE"
}

rebootstrap() {
  kubectl delete job inference-bootstrap-minio ollama-pull-models -n "$NAMESPACE" --ignore-not-found
  kubectl apply -f deploy/kubernetes/bootstrap-jobs.yaml
  kubectl wait --for=condition=complete job/inference-bootstrap-minio -n "$NAMESPACE" --timeout=10m
  kubectl wait --for=condition=complete job/ollama-pull-models -n "$NAMESPACE" --timeout=20m
}

restart_apps() {
  kubectl rollout restart deployment/inference -n "$NAMESPACE"
  kubectl rollout restart deployment/inference-worker -n "$NAMESPACE"
  kubectl rollout restart deployment/api -n "$NAMESPACE"
}

wait_apps() {
  kubectl rollout status deployment/inference -n "$NAMESPACE"
  kubectl rollout status deployment/inference-worker -n "$NAMESPACE"
  kubectl rollout status deployment/api -n "$NAMESPACE"
}

case "$COMMAND" in
  up)
    build_app_images
    load_app_images
    kubectl apply -k deploy/kubernetes
    wait_infra
    rebootstrap
    wait_apps
    kubectl get pods -n "$NAMESPACE"
    ;;
  redeploy)
    build_app_images
    load_app_images
    restart_apps
    wait_apps
    kubectl get pods -n "$NAMESPACE"
    ;;
  rebootstrap)
    rebootstrap
    kubectl get jobs -n "$NAMESPACE"
    ;;
  reset-redis)
    echo "Flushing Redis DB 0 for dev reset..."
    kubectl exec -n "$NAMESPACE" redis-0 -- redis-cli FLUSHDB
    restart_apps
    wait_apps
    kubectl get pods -n "$NAMESPACE"
    echo "Redis dev state cleared. This removes queued/running job records and retrieval state."
    ;;
  status)
    kubectl get pods -n "$NAMESPACE"
    kubectl get jobs -n "$NAMESPACE"
    ;;
  port-forward-api)
    kubectl port-forward svc/api 8000:8000 -n "$NAMESPACE"
    ;;
  port-forward-minio)
    kubectl port-forward svc/minio-console 9001:9001 -n "$NAMESPACE"
    ;;
  *)
    echo "Usage: $0 [up|redeploy|rebootstrap|reset-redis|status|port-forward-api|port-forward-minio]" >&2
    exit 1
    ;;
esac
