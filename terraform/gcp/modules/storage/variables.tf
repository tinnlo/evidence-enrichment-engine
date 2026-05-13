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

variable "bucket_name" {
  description = "GCS bucket name"
  type        = string
}

variable "lifecycle_age" {
  description = "Age in days after which to delete objects"
  type        = number
  default     = 30
}

variable "versioning" {
  description = "Enable versioning"
  type        = bool
  default     = false
}

variable "service_account_email" {
  description = "Service account email to grant access"
  type        = string
}
