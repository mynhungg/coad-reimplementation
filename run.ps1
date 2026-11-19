# CoAD demo, one command on Windows:  powershell -ExecutionPolicy Bypass -File run.ps1
# Installs uv (https://docs.astral.sh/uv/) if missing, then uv installs Python 3.11 + CPU PyTorch in its own cache
# and starts the demo server, which opens http://localhost:8765 in the browser.
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'uv not found, installing it (official installer from astral.sh)...'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Set-Location $PSScriptRoot
uv run demo/serve_coad.py @args
