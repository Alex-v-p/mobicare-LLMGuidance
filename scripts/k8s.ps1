param(
    [ValidateSet('up', 'redeploy', 'rebootstrap', 'reset-redis', 'status', 'start', 'shutdown', 'port-forward-api', 'port-forward-minio')]
    [string]$Command = 'status'
)

$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $PSScriptRoot
$Namespace = 'llmguidance'
$AppImages = @(
    @{ Dockerfile = 'deploy/api.Dockerfile'; Image = 'mobicare-llm/api:1.0.0' },
    @{ Dockerfile = 'deploy/inference.http.Dockerfile'; Image = 'mobicare-llm/inference-http:1.0.0' },
    @{ Dockerfile = 'deploy/inference.worker.Dockerfile'; Image = 'mobicare-llm/inference-worker:1.0.0' }
)

Set-Location $RootDir

function Get-KubeContext {
    return (kubectl config current-context).Trim()
}

function Get-MinikubeProfileNames {
    $profiles = minikube profile list -o json | ConvertFrom-Json
    return @($profiles.valid | ForEach-Object { $_.Name })
}

function Build-AppImages {
    foreach ($spec in $AppImages) {
        Write-Host "Building $($spec.Image)..."
        docker build -f $spec.Dockerfile -t $spec.Image .
    }
}

function Load-AppImages {
    $context = Get-KubeContext
    Write-Host "Active Kubernetes context: $context"

    $profileNames = Get-MinikubeProfileNames

    if ($profileNames -contains $context) {
        foreach ($spec in $AppImages) {
            Write-Host "Loading $($spec.Image) into minikube profile '$context'..."
            minikube -p $context image load $spec.Image
        }
        return
    }

    if ($context -like 'kind-*') {
        foreach ($spec in $AppImages) {
            Write-Host "Loading $($spec.Image) into kind..."
            kind load docker-image $spec.Image
        }
        return
    }

    Write-Host "No automatic image loader configured for context '$context'."
    Write-Host 'If your cluster cannot see local Docker images directly, push them to a registry or load them manually.'
}

function Start-MinikubeGracefully {
    $context = Get-KubeContext
    Write-Host "Active Kubernetes context: $context"

    $profileNames = Get-MinikubeProfileNames

    if ($profileNames -contains $context) {
        Write-Host "Starting minikube profile '$context'..."
        minikube -p $context start
        kubectl config use-context $context | Out-Null
        Write-Host "Minikube profile '$context' has been started."
        return
    }

    throw "Current context '$context' is not a known minikube profile, so start was aborted."
}

function Shutdown-MinikubeGracefully {
    $context = Get-KubeContext
    Write-Host "Active Kubernetes context: $context"

    $profileNames = Get-MinikubeProfileNames

    if ($profileNames -contains $context) {
        Write-Host "Stopping minikube profile '$context' gracefully..."
        minikube -p $context stop
        Write-Host "Minikube profile '$context' has been stopped."
        return
    }

    throw "Current context '$context' is not a known minikube profile, so shutdown was aborted."
}

function Wait-Infra {
    kubectl rollout status statefulset/redis -n $Namespace
    kubectl rollout status statefulset/qdrant -n $Namespace
    kubectl rollout status statefulset/minio -n $Namespace
    kubectl rollout status statefulset/ollama -n $Namespace
}

function Rebootstrap {
    kubectl delete job inference-bootstrap-minio ollama-pull-models -n $Namespace --ignore-not-found
    kubectl apply -k deploy/kubernetes/overlays/production/bootstrap
    kubectl wait --for=condition=complete job/inference-bootstrap-minio -n $Namespace --timeout=10m
    kubectl wait --for=condition=complete job/ollama-pull-models -n $Namespace --timeout=20m
}

function Restart-Apps {
    kubectl rollout restart deployment/inference -n $Namespace
    kubectl rollout restart deployment/inference-worker -n $Namespace
    kubectl rollout restart deployment/api -n $Namespace
}

function Wait-Apps {
    kubectl rollout status deployment/inference -n $Namespace
    kubectl rollout status deployment/inference-worker -n $Namespace
    kubectl rollout status deployment/api -n $Namespace
}

switch ($Command) {
    'up' {
        Build-AppImages
        Load-AppImages
        kubectl apply -k deploy/kubernetes
        Wait-Infra
        Rebootstrap
        Wait-Apps
        kubectl get pods -n $Namespace
        break
    }
    'redeploy' {
        Build-AppImages
        Load-AppImages
        Restart-Apps
        Wait-Apps
        kubectl get pods -n $Namespace
        break
    }
    'rebootstrap' {
        Rebootstrap
        kubectl get jobs -n $Namespace
        break
    }
    'reset-redis' {
        Write-Host 'Flushing Redis DB 0 for dev reset...'
        kubectl exec -n $Namespace redis-0 -- redis-cli FLUSHDB
        Restart-Apps
        Wait-Apps
        kubectl get pods -n $Namespace
        Write-Host 'Redis dev state cleared. This removes queued/running job records and retrieval state.'
        break
    }
    'status' {
        kubectl get pods -n $Namespace
        kubectl get jobs -n $Namespace
        break
    }
    'start' {
        Start-MinikubeGracefully
        break
    }
    'shutdown' {
        Shutdown-MinikubeGracefully
        break
    }
    'port-forward-api' {
        kubectl port-forward svc/api 8000:8000 -n $Namespace
        break
    }
    'port-forward-minio' {
        kubectl port-forward svc/minio-console 9001:9001 -n $Namespace
        break
    }
}