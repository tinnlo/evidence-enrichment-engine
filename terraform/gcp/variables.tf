variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

# VPC Configuration
variable "create_vpc" {
  description = "Whether to create a new VPC (false = use default VPC)"
  type        = bool
  default     = false
}

variable "vpc_connector_cidr" {
  description = "CIDR range for VPC connector"
  type        = string
  default     = "10.8.0.0/28"
}

# Memorystore Redis Configuration
variable "redis_tier" {
  description = "Redis tier (BASIC or STANDARD)"
  type        = string
  default     = "BASIC"
  validation {
    condition     = contains(["BASIC", "STANDARD"], var.redis_tier)
    error_message = "Redis tier must be BASIC or STANDARD."
  }
}

variable "redis_memory_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
  validation {
    condition     = var.redis_memory_gb >= 1 && var.redis_memory_gb <= 300
    error_message = "Redis memory must be between 1 and 300 GB."
  }
}

variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "REDIS_7_0"
}

# GCS Configuration
variable "gcs_lifecycle_age" {
  description = "Age in days after which to delete objects from GCS"
  type        = number
  default     = 30
}

variable "gcs_versioning" {
  description = "Enable versioning for GCS bucket"
  type        = bool
  default     = false
}

# Cloud Run Configuration
variable "cloud_run_image" {
  description = "Container image for Cloud Run service"
  type        = string
}

variable "cloud_run_cpu" {
  description = "CPU allocation for Cloud Run (e.g., '1', '2', '4')"
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Memory allocation for Cloud Run (e.g., '512Mi', '1Gi', '2Gi')"
  type        = string
  default     = "2Gi"
}

variable "cloud_run_timeout" {
  description = "Task timeout in seconds"
  type        = number
  default     = 600
  validation {
    condition     = var.cloud_run_timeout >= 1 && var.cloud_run_timeout <= 3600
    error_message = "Timeout must be between 1 and 3600 seconds."
  }
}

variable "cloud_run_max_retries" {
  description = "Maximum number of retries for failed tasks"
  type        = number
  default     = 3
  validation {
    condition     = var.cloud_run_max_retries >= 0 && var.cloud_run_max_retries <= 10
    error_message = "Max retries must be between 0 and 10."
  }
}

# Observability Configuration
variable "observability_backend" {
  description = "Observability backend (langfuse, langsmith, dual, none)"
  type        = string
  default     = "langfuse"
  validation {
    condition     = contains(["langfuse", "langsmith", "dual", "none"], var.observability_backend)
    error_message = "Observability backend must be langfuse, langsmith, dual, or none."
  }
}

variable "langfuse_host" {
  description = "Langfuse host URL"
  type        = string
  default     = "https://cloud.langfuse.com"
}

# Execution Policy Configuration
variable "execution_policy_mode" {
  description = "Execution policy mode (off, audit, enforce)"
  type        = string
  default     = "audit"
  validation {
    condition     = contains(["off", "audit", "enforce"], var.execution_policy_mode)
    error_message = "Execution policy mode must be off, audit, or enforce."
  }
}

# Cloud Scheduler Configuration
variable "enable_scheduler" {
  description = "Enable Cloud Scheduler for periodic triggers"
  type        = bool
  default     = false
}

variable "scheduler_cron" {
  description = "Cron schedule for Cloud Scheduler (e.g., '0 * * * *' for hourly)"
  type        = string
  default     = "0 * * * *"
}

variable "scheduler_timezone" {
  description = "Timezone for Cloud Scheduler"
  type        = string
  default     = "UTC"
}
