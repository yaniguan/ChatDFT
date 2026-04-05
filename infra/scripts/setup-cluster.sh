#!/usr/bin/env bash
# Post-Terraform EKS setup: configure kubectl, install ALB controller
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TF_DIR="$SCRIPT_DIR/../terraform"

# Get Terraform outputs
AWS_REGION=$(cd "$TF_DIR" && terraform output -raw aws_region)
CLUSTER_NAME=$(cd "$TF_DIR" && terraform output -raw eks_cluster_name)
LB_ROLE_ARN=$(cd "$TF_DIR" && terraform output -raw lb_controller_role_arn)

echo "==> Updating kubeconfig for cluster: $CLUSTER_NAME"
aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"

echo "==> Verifying cluster access..."
kubectl cluster-info

echo "==> Installing AWS Load Balancer Controller..."

# Add Helm repo
helm repo add eks https://aws.github.io/eks-charts 2>/dev/null || true
helm repo update

# Install/upgrade the controller
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName="$CLUSTER_NAME" \
  --set serviceAccount.create=true \
  --set serviceAccount.name=aws-load-balancer-controller \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$LB_ROLE_ARN" \
  --set region="$AWS_REGION" \
  --set vpcId="$(cd "$TF_DIR" && terraform output -raw vpc_id)"

echo "==> Waiting for ALB controller to be ready..."
kubectl rollout status deployment/aws-load-balancer-controller -n kube-system --timeout=120s

echo "==> Creating chatdft namespace..."
kubectl apply -f "$SCRIPT_DIR/../k8s/namespace.yaml"

echo "==> Cluster setup complete!"
echo "    Next steps:"
echo "    1. Create secrets:  kubectl apply -f infra/k8s/secrets.yaml -n chatdft"
echo "    2. Deploy app:      kubectl apply -f infra/k8s/ -n chatdft"
