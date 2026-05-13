# Staging Environment Configuration

project_id  = "evidence-enrichment-staging"
region      = "us-central1"
environment = "staging"

# Container image (update after pushing to Artifact Registry)
cloud_run_image = "us-central1-docker.pkg.dev/evidence-enrichment-staging/evidence-enrichment/evidence-enrichment-engine:latest"

# VPC Configuration
create_vpc         = false
vpc_connector_cidr = "10.8.0.0/28"

# Redis Configuration (Basic tier for staging)
redis_tier      = "BASIC"
redis_memory_gb = 2
redis_version   = "REDIS_7_0"

# GCS Configuration
gcs_lifecycle_age = 30
gcs_versioning    = true

# Cloud Run Configuration (moderate for staging)
cloud_run_cpu         = "2"
cloud_run_memory      = "2Gi"
cloud_run_timeout     = 600
cloud_run_max_retries = 3

# Observability
observability_backend = "langfuse"
langfuse_host         = "https://cloud.langfuse.com"

# Execution Policy (audit mode for staging)
execution_policy_mode = "audit"

# Cloud Scheduler (enabled for staging)
enable_scheduler   = true
scheduler_cron     = "0 */2 * * *" # Every 2 hours
scheduler_timezone = "UTC"
