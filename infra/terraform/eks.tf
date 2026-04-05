# infra/terraform/eks.tf
# Amazon EKS cluster with managed node group

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-eks"
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Allow public access to API server (for kubectl from local machine)
  cluster_endpoint_public_access = true

  # EKS Addons
  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
  }

  # Managed node group
  eks_managed_node_groups = {
    default = {
      name           = "${var.project_name}-nodes"
      instance_types = [var.node_instance_type]
      capacity_type  = "ON_DEMAND"

      min_size     = var.node_min_count
      max_size     = var.node_max_count
      desired_size = var.node_desired_count

      # Disk
      disk_size = 30

      # Allow pods to pull from ECR
      iam_role_additional_policies = {
        ecr = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
      }
    }
  }

  # OIDC provider for IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Allow the current caller to manage the cluster
  enable_cluster_creator_admin_permissions = true
}
