# infra/terraform/outputs.tf

output "aws_region" {
  value = var.aws_region
}

# VPC
output "vpc_id" {
  value = module.vpc.vpc_id
}

# EKS
output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "eks_update_kubeconfig_command" {
  value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# RDS
output "rds_endpoint" {
  value = aws_db_instance.main.endpoint
}

output "rds_database_url" {
  description = "DATABASE_URL for the application (password redacted)"
  value       = "postgresql+asyncpg://${var.db_username}:****@${aws_db_instance.main.endpoint}/${var.db_name}"
  sensitive   = true
}

output "rds_password_secret_arn" {
  value = aws_secretsmanager_secret.db_password.arn
}

# ECR
output "ecr_server_url" {
  value = aws_ecr_repository.server.repository_url
}

output "ecr_ui_url" {
  value = aws_ecr_repository.ui.repository_url
}

# IAM
output "lb_controller_role_arn" {
  value = module.lb_controller_irsa.iam_role_arn
}

output "secrets_role_arn" {
  value = module.secrets_irsa.iam_role_arn
}
