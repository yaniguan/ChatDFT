# infra/terraform/vpc.tf
# VPC with public + private subnets across 2 AZs

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 2)
  public_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnets = ["10.0.10.0/24", "10.0.11.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true # Cost savings for small scale
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tags required for EKS ALB Ingress Controller
  public_subnet_tags = {
    "kubernetes.io/role/elb"                              = 1
    "kubernetes.io/cluster/${var.project_name}-eks"        = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"                      = 1
    "kubernetes.io/cluster/${var.project_name}-eks"        = "shared"
  }
}
