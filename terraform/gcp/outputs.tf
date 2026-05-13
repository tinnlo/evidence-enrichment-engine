output "cloud_run_job_name" {
  description = "Name of the Cloud Run job"
  value       = module.cloud_run_job.job_name
}

output "cloud_run_job_id" {
  description = "ID of the Cloud Run job"
  value       = module.cloud_run_job.job_id
}

output "redis_host" {
  description = "Memorystore Redis host IP address"
  value       = module.redis.host
}

output "redis_port" {
  description = "Memorystore Redis port"
  value       = module.redis.port
}

output "gcs_bucket_name" {
  description = "GCS bucket name for trace artifacts"
  value       = module.storage.bucket_name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL"
  value       = module.storage.bucket_url
}

output "service_account_email" {
  description = "Service account email for Cloud Run"
  value       = google_service_account.cloud_run_sa.email
}

output "vpc_connector_name" {
  description = "VPC connector name"
  value       = google_vpc_access_connector.connector.name
}

output "secret_ids" {
  description = "Secret Manager secret IDs"
  value       = module.secrets.secret_ids
  sensitive   = true
}

output "scheduler_job_name" {
  description = "Cloud Scheduler job name (if enabled)"
  value       = var.enable_scheduler ? google_cloud_scheduler_job.enrichment_hourly[0].name : null
}

output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    environment           = var.environment
    region                = var.region
    cloud_run_job_name    = module.cloud_run_job.job_name
    redis_tier            = var.redis_tier
    redis_memory_gb       = var.redis_memory_gb
    execution_policy_mode = var.execution_policy_mode
    scheduler_enabled     = var.enable_scheduler
  }
}
