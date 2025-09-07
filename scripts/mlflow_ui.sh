#!/usr/bin/env bash
set -euo pipefail

# Start MLflow UI using local sqlite store in mlflow.db under project root
# Usage: ./scripts/mlflow_ui.sh [HOST] [PORT]

HOST=${1:-127.0.0.1}
PORT=${2:-5000}
BACKEND_URI="sqlite:///$(pwd)/mlflow.db"
ARTIFACT_ROOT="$(pwd)/mlruns"

export MLFLOW_TRACKING_URI="http://${HOST}:${PORT}"

# Create artifact directory if missing
mkdir -p "${ARTIFACT_ROOT}"

echo "Starting MLflow UI on ${MLFLOW_TRACKING_URI}"

echo "Backend Store: ${BACKEND_URI}"

echo "Artifacts: ${ARTIFACT_ROOT}"

exec mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}"
