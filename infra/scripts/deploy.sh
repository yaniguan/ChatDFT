#!/usr/bin/env bash
# Full deployment pipeline: Terraform → Docker → K8s
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$SCRIPT_DIR/../terraform"
K8S_DIR="$SCRIPT_DIR/../k8s"

TAG="${1:-latest}"
NAMESPACE="chatdft"

echo "============================================"
echo " ChatDFT — AWS EKS Deployment"
echo "============================================"

# Step 1: Terraform
echo ""
echo "==> Step 1: Provisioning AWS infrastructure..."
cd "$TF_DIR"
terraform init
terraform apply -auto-approve
cd "$SCRIPT_DIR"

# Step 2: Cluster setup (kubeconfig + ALB controller)
echo ""
echo "==> Step 2: Configuring EKS cluster..."
bash "$SCRIPT_DIR/setup-cluster.sh"

# Step 3: Build and push Docker images
echo ""
echo "==> Step 3: Building and pushing Docker images..."
bash "$SCRIPT_DIR/build-push.sh" "$TAG"

# Step 4: Update image references in K8s manifests
ECR_SERVER=$(cd "$TF_DIR" && terraform output -raw ecr_server_url)
ECR_UI=$(cd "$TF_DIR" && terraform output -raw ecr_ui_url)

echo ""
echo "==> Step 4: Deploying to Kubernetes..."

# Apply namespace first
kubectl apply -f "$K8S_DIR/namespace.yaml"

# Check if secrets exist
if ! kubectl get secret chatdft-secrets -n "$NAMESPACE" &>/dev/null; then
    echo ""
    echo "WARNING: Secrets not found. Please create them first:"
    echo "  1. Copy infra/k8s/secrets.yaml.example to infra/k8s/secrets.yaml"
    echo "  2. Fill in real values (DATABASE_URL, OPENAI_API_KEY, etc.)"
    echo "  3. Run: kubectl apply -f infra/k8s/secrets.yaml"
    echo ""
    echo "RDS endpoint: $(cd "$TF_DIR" && terraform output -raw rds_endpoint)"
    echo "DB password secret ARN: $(cd "$TF_DIR" && terraform output -raw rds_password_secret_arn)"
    echo ""
    read -p "Press Enter after creating secrets to continue, or Ctrl+C to abort..."
fi

# Apply all K8s manifests (patch image references on the fly)
for f in "$K8S_DIR"/server-*.yaml "$K8S_DIR"/ui-*.yaml "$K8S_DIR"/ingress.yaml; do
    sed "s|ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/chatdft-server|$ECR_SERVER|g; \
         s|ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/chatdft-ui|$ECR_UI|g" "$f" | \
    kubectl apply -f - -n "$NAMESPACE"
done

# Step 5: Run DB migration
echo ""
echo "==> Step 5: Running database migration..."
MIGRATE_YAML=$(sed "s|ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/chatdft-server|$ECR_SERVER|g" "$K8S_DIR/jobs/db-migrate.yaml")

# Delete previous migration job if exists
kubectl delete job chatdft-db-migrate -n "$NAMESPACE" 2>/dev/null || true
echo "$MIGRATE_YAML" | kubectl apply -f - -n "$NAMESPACE"

echo "==> Waiting for migration to complete..."
kubectl wait --for=condition=complete job/chatdft-db-migrate -n "$NAMESPACE" --timeout=120s

# Step 6: Verify
echo ""
echo "==> Step 6: Verifying deployment..."
kubectl get pods -n "$NAMESPACE"
kubectl get ingress -n "$NAMESPACE"

echo ""
echo "============================================"
echo " Deployment complete!"
echo "============================================"
echo ""
echo "Check ALB URL:"
echo "  kubectl get ingress chatdft-ingress -n chatdft -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
echo ""
