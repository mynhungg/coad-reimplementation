#!/usr/bin/env sh
# CoAD demo, one command on macOS / Linux:  sh run.sh
# Installs uv (https://docs.astral.sh/uv/) if missing, then uv installs Python 3.11 + CPU PyTorch in its own cache
# and starts the demo server, which opens http://localhost:8765 in the browser.
set -e
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, installing it (official installer from astral.sh)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
fi
cd "$(dirname "$0")"
exec uv run demo/serve_coad.py "$@"
