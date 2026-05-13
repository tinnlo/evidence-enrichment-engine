variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "job_name" {
  description = "Cloud Run job name"
  type        = string
}

variable "image" {
  description = "Container image URL"
  type        = string
}

variable "service_account_email" {
  description = "Service account email for Cloud Run"
  type        = string
}

variable "vpc_connector_name" {
  description = "VPC connector name for private network access"
  type        = string
}

variable "env_vars" {
  description = "Environment variables"
  type        = map(string)
  default     = {}
}

variable "secrets" {
  description = "Secrets from Secret Manager"
  type = map(object({
    secret_name = string
    version     = string
  }))
  default = {}
}

variable "cpu_limit" {
  description = "CPU limit"
  type        = string
  default     = "2"
}

variable "memory_limit" {
  description = "Memory limit"
  type        = string
  default     = "2Gi"
}

variable "timeout" {
  description = "Task timeout in seconds"
  type        = number
  default     = 600
}

variable "max_retries" {
  description = "Maximum number of retries for failed tasks"
  type        = number
  default     = 3
}

variable "task_count" {
  description = "Number of tasks to run (default 1 for single execution)"
  type        = number
  default     = 1
}

variable "parallelism" {
  description = "Maximum number of tasks to run in parallel"
  type        = number
  default     = 1
}
