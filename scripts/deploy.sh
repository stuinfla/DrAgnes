#!/usr/bin/env bash
set -euo pipefail

# Mela Cloud Run Deployment Script
PROJECT_ID="${GCP_PROJECT_ID:-ruv-dev}"
REGION="${GCP_REGION:-us-central1}"
IMAGE="gcr.io/${PROJECT_ID}/mela:latest"

echo "Building Mela container..."
docker build -t "${IMAGE}" .

echo "Pushing to GCR..."
docker push "${IMAGE}"

echo "Deploying to Cloud Run..."
gcloud run services replace cloud-run.yaml \
  --project="${PROJECT_ID}" \
  --region="${REGION}"

echo "Done. Service URL:"
gcloud run services describe mela \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --format='value(status.url)'
