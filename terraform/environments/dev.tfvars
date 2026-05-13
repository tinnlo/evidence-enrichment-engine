# Development Environment Configuration

project_id  = "evidence-enrichment-dev"
region      = "us-central1"
environment = "dev"

# Container image (update after pushing to Artifact Registry)
cloud_run_image = "us-central1-docker.pkg.dev/evidence-enrichment-dev/evidence-enrichment/evidence-enrichment-engine:latest"

# VPC Configuration
create_vpc         = false
vpc_connector_cidr = "10.8.0.0/28"

# Redis Configuration (Basic tier for dev)
redis_tier      = "BASIC"
redis_memory_gb = 1
redis_version   = "REDIS_7_0"

# GCS Configuration
gcs_lifecycle_age = 30
gcs_versioning    = false

# Cloud Run Configuration (minimal for dev)
cloud_run_cpu         = "2"
cloud_run_memory      = "2Gi"
cloud_run_timeout     = 600
cloud_run_max_retries = 3

# Observability
observability_backend = "langfuse"
langfuse_host         = "https://cloud.langfuse.com"

# Execution Policy (permissive for dev)
execution_policy_mode = "off"

# Cloud Scheduler (disabled for dev)
enable_scheduler   = false
scheduler_cron     = "0 * * * *"
scheduler_timezone = "UTC"
