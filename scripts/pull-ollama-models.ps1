$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (Test-Path '.env') {
    Get-Content '.env' |
        Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } |
        ForEach-Object {
            $parts = $_ -split '=', 2
            [Environment]::SetEnvironmentVariable($parts[0], $parts[1])
        }
}

$ollamaService = if ($env:OLLAMA_SERVICE) { $env:OLLAMA_SERVICE } else { 'ollama' }
$timeoutSeconds = if ($env:OLLAMA_START_TIMEOUT_SECONDS) { [int]$env:OLLAMA_START_TIMEOUT_SECONDS } else { 180 }
$ollamaModel = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { 'qwen2.5:0.5b' }
$embeddingModel = if ($env:OLLAMA_EMBEDDING_MODEL) { $env:OLLAMA_EMBEDDING_MODEL } else { 'nomic-embed-text' }

$models = [System.Collections.Generic.List[string]]::new()
foreach ($candidate in @($embeddingModel, $ollamaModel) + $args) {
    if (-not [string]::IsNullOrWhiteSpace($candidate) -and -not $models.Contains($candidate)) {
        $models.Add($candidate)
    }
}

if ($models.Count -eq 0) {
    Write-Host 'No Ollama models configured.'
    exit 0
}

Write-Host "Waiting for the '$ollamaService' container to accept Ollama commands..."
$deadline = (Get-Date).AddSeconds($timeoutSeconds)
while ((Get-Date) -lt $deadline) {
    try {
        docker compose exec -T $ollamaService ollama list *> $null
        break
    }
    catch {
        Start-Sleep -Seconds 2
    }
}

if ((Get-Date) -ge $deadline) {
    throw "Timed out while waiting for Ollama after $timeoutSeconds seconds."
}

foreach ($model in $models) {
    Write-Host "Pulling Ollama model: $model"
    docker compose exec -T $ollamaService ollama pull $model
}

Write-Host 'Ollama model pull complete.'
