$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host 'Building and starting the Docker Compose stack...'
docker compose up -d --build @args

Write-Host 'Installing Ollama models...'
& "$PSScriptRoot/pull-ollama-models.ps1"

Write-Host 'Startup complete.'
