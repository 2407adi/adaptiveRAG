#!/usr/bin/env bash

# infra/deploy.sh — one-shot Azure Container Apps deployment for AdaptiveRAG.
# Idempotent: safe to re-run; each section creates-or-skips.
# Usage: bash infra/deploy.sh   (run from the repo root, .env must exist)

set -euo pipefail   # die on error / undefined var / mid-pipeline failure

# ── Config: every name in one place ─────────────────────────────
RESOURCE_GROUP="genai-rag-agents-rg"        # existing RG from Block 0.2
LOCATION="centralindia"                     # same region as the RG + OpenAI
STORAGE_ACCOUNT="adaptiveragstorage"        # existing storage account (reused)
SHARE_NAME="adaptiverag-data"               # Azure Files share we'll create
ACA_ENV="adaptiverag-env"                   # Container Apps environment
APP_NAME="adaptiverag"                      # the container app itself
IMAGE="ghcr.io/2407adi/adaptiverag:latest"  # public GHCR image from 4.5

echo "Deploying $APP_NAME to $RESOURCE_GROUP ($LOCATION)..."

# ── 1. Azure Files share — the app's persistent disk ────────────
# One network folder inside the existing storage account. Mounted
# later at /app/data so Chroma / SQLite / audit log survive revisions.
az storage share-rm create \
  --resource-group "$RESOURCE_GROUP" \
  --storage-account "$STORAGE_ACCOUNT" \
  --name "$SHARE_NAME" \
  --quota 5 \
  --output none
echo "✓ file share '$SHARE_NAME' ready (5 GiB cap)"

# ── 2. Container Apps environment — the building the app lives in ──
# Auto-creates a Log Analytics workspace for container logs.
az containerapp env create \
  --name "$ACA_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none
echo "✓ environment '$ACA_ENV' ready"

# ── 3. Teach the environment about the share ────────────────────
# Fetch the storage account's master key into a variable (never printed).
STORAGE_KEY=$(az storage account keys list \
  --resource-group "$RESOURCE_GROUP" \
  --account-name "$STORAGE_ACCOUNT" \
  --query "[0].value" --output tsv)

# Register the share under an alias; apps mount it by this name.
az containerapp env storage set \
  --name "$ACA_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --storage-name "$SHARE_NAME" \
  --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "$SHARE_NAME" \
  --access-mode ReadWrite \
  --output none
echo "✓ share registered with environment as '$SHARE_NAME'"

# ── 4. The container app — first deploy ─────────────────────────
source .env   # load the six values from the repo's .env into shell vars

az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ACA_ENV" \
  --image "$IMAGE" \
  --ingress external --target-port 8000 \
  --min-replicas 0 --max-replicas 1 \
  --cpu 1.0 --memory 2.0Gi \
  --secrets \
      azure-openai-api-key="$AZURE_OPENAI_API_KEY" \
      audit-hmac-key="$AUDIT_HMAC_KEY" \
      tavily-api-key="$TAVILY_API_KEY" \
      adaptiverag-api-keys="$ADAPTIVERAG_API_KEYS" \
  --env-vars \
      AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
      AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
      AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key \
      AUDIT_HMAC_KEY=secretref:audit-hmac-key \
      TAVILY_API_KEY=secretref:tavily-api-key \
      ADAPTIVERAG_API_KEYS=secretref:adaptiverag-api-keys \
  --output none

FQDN=$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn --output tsv)
echo "✓ app live at https://$FQDN"