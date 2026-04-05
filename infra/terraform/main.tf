# infra/terraform/main.tf
# ChatDFT — AWS infrastructure root module

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state storage:
  # backend "s3" {
  #   bucket         = "chatdft-terraform-state"
  #   key            = "infra/terraform.tfstate"
  #   region         = "us-west-2"
  #   dynamodb_table = "chatdft-terraform-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "chatdft"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}
