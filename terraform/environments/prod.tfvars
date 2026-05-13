# Production Environment Configuration

project_id  = "evidence-enrichment-prod"
region      = "us-central1"
environment = "prod"

# Container image (update after pushing to Artifact Registry)
cloud_run_image = "us-central1-docker.pkg.dev/evidence-enrichment-prod/evidence-enrichment/evidence-enrichment-engine:latest"

# VPC Configuration
create_vpc         = false
vpc_connector_cidr = "10.8.0.0/28"

# Redis Configuration (Standard tier with HA for prod)
redis_tier      = "STANDARD"
redis_memory_gb = 5
redis_version   = "REDIS_7_0"

# GCS Configuration
gcs_lifecycle_age = 90 # Keep traces longer in prod
gcs_versioning    = true

# Cloud Run Configuration (production-grade)
cloud_run_cpu         = "2"
cloud_run_memory      = "2Gi"
cloud_run_timeout     = 600
cloud_run_max_retries = 3

# Observability
observability_backend = "langfuse"
langfuse_host         = "https://cloud.langfuse.com"

# Execution Policy (enforce mode for prod)
execution_policy_mode = "enforce"

# Cloud Scheduler (enabled for prod)
enable_scheduler   = true
scheduler_cron     = "0 * * * *" # Hourly
scheduler_timezone = "UTC"
