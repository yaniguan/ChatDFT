#!/usr/bin/env bash
# Build Docker images and push to ECR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Resolve from Terraform outputs
TF_DIR="$SCRIPT_DIR/../terraform"
AWS_REGION=$(cd "$TF_DIR" && terraform output -raw aws_region)
ECR_SERVER=$(cd "$TF_DIR" && terraform output -raw ecr_server_url)
ECR_UI=$(cd "$TF_DIR" && terraform output -raw ecr_ui_url)
ACCOUNT_ID=$(echo "$ECR_SERVER" | cut -d. -f1)

TAG="${1:-latest}"

echo "==> Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "==> Building server image..."
docker build -t "$ECR_SERVER:$TAG" -f "$REPO_ROOT/Dockerfile" "$REPO_ROOT"

echo "==> Building UI image..."
docker build -t "$ECR_UI:$TAG" -f "$REPO_ROOT/Dockerfile.ui" "$REPO_ROOT"

echo "==> Pushing server image..."
docker push "$ECR_SERVER:$TAG"

echo "==> Pushing UI image..."
docker push "$ECR_UI:$TAG"

echo "==> Done. Images pushed:"
echo "    Server: $ECR_SERVER:$TAG"
echo "    UI:     $ECR_UI:$TAG"
