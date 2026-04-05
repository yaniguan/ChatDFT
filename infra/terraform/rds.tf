# infra/terraform/rds.tf
# RDS PostgreSQL 16 with pgvector support

resource "random_password" "db_password" {
  length  = 24
  special = false # Avoid URL-encoding issues in DATABASE_URL
}

resource "aws_secretsmanager_secret" "db_password" {
  name                    = "${var.project_name}-db-password"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db_password.result
}

# Security group: allow PostgreSQL from EKS nodes only
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
    description     = "PostgreSQL from EKS nodes"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Subnet group
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

# RDS instance
# pgvector is installed as a PostgreSQL extension (CREATE EXTENSION vector),
# NOT via shared_preload_libraries. No custom parameter group needed.
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-db"

  engine         = "postgres"
  engine_version = "16"
  instance_class = var.db_instance_class

  allocated_storage     = var.db_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  max_allocated_storage = 50 # Autoscaling ceiling

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db_password.result

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az            = false # Single AZ for small scale
  publicly_accessible = false
  skip_final_snapshot = true

  backup_retention_period = 1 # Free tier limit
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
}
