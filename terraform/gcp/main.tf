terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state backend (GCS)
  # Before running terraform init, create the state bucket:
  #   export PROJECT_ID="your-project-id"
  #   export REGION="us-central1"
  #   gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${PROJECT_ID}-terraform-state
  #   gsutil versioning set on gs://${PROJECT_ID}-terraform-state
  #
  # Then configure the bucket name using -backend-config:
  #   terraform init -backend-config="bucket=YOUR_PROJECT_ID-terraform-state"
  #
  # The empty backend block is required for partial configuration to work.
  backend "gcs" {
    prefix = "evidence-enrichment/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "vpcaccess.googleapis.com",
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "pubsub.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# Service Account for Cloud Run
resource "google_service_account" "cloud_run_sa" {
  account_id   = "evidence-enrichment-sa"
  display_name = "Evidence Enrichment Cloud Run Service Account"
  description  = "Service account for evidence-enrichment-engine Cloud Run service"
}

# IAM roles for service account
resource "google_project_iam_member" "cloud_run_sa_roles" {
  for_each = toset([
    "roles/secretmanager.secretAccessor", # Read secrets
    "roles/logging.logWriter",            # Write logs
    "roles/monitoring.metricWriter",      # Write metrics
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# VPC Network (if not using default)
resource "google_compute_network" "vpc" {
  count                   = var.create_vpc ? 1 : 0
  name                    = "${var.environment}-evidence-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  count         = var.create_vpc ? 1 : 0
  name          = "${var.environment}-evidence-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc[0].id
}

# VPC Connector for Cloud Run to access Memorystore Redis
resource "google_vpc_access_connector" "connector" {
  name          = "${var.environment}-evidence-connector"
  region        = var.region
  network       = var.create_vpc ? google_compute_network.vpc[0].name : "default"
  ip_cidr_range = var.vpc_connector_cidr
  min_instances = 2
  max_instances = 3

  depends_on = [google_project_service.required_apis]
}

# Memorystore Redis Module
module "redis" {
  source = "./modules/memorystore"

  project_id         = var.project_id
  region             = var.region
  environment        = var.environment
  instance_name      = "${var.environment}-evidence-cache"
  tier               = var.redis_tier
  memory_size_gb     = var.redis_memory_gb
  redis_version      = var.redis_version
  authorized_network = var.create_vpc ? google_compute_network.vpc[0].id : "default"

  depends_on = [google_project_service.required_apis]
}

# Secret Manager Module
module "secrets" {
  source = "./modules/secrets"

  project_id  = var.project_id
  environment = var.environment
  secrets = {
    # Observability
    langfuse-api-key    = "Langfuse public API key"
    langfuse-secret-key = "Langfuse secret key"

    # Live analysis/synthesis providers (at least one required for live mode)
    openai-api-key    = "OpenAI API key for GPT models"
    anthropic-api-key = "Anthropic API key for Claude models"

    # Live search providers (at least one required for live mode)
    serper-api-key = "Serper API key for Google search"
    tavily-api-key = "Tavily API key for web search"
  }
  service_account_email = google_service_account.cloud_run_sa.email

  depends_on = [google_project_service.required_apis]
}

# GCS Bucket Module
module "storage" {
  source = "./modules/storage"

  project_id            = var.project_id
  region                = var.region
  environment           = var.environment
  bucket_name           = "${var.project_id}-${var.environment}-evidence-traces"
  lifecycle_age         = var.gcs_lifecycle_age
  versioning            = var.gcs_versioning
  service_account_email = google_service_account.cloud_run_sa.email

  depends_on = [google_project_service.required_apis]
}

# Cloud Run Job Module
module "cloud_run_job" {
  source = "./modules/cloud-run"

  project_id            = var.project_id
  region                = var.region
  environment           = var.environment
  job_name              = "${var.environment}-evidence-enrichment-job"
  image                 = var.cloud_run_image
  service_account_email = google_service_account.cloud_run_sa.email
  vpc_connector_name    = google_vpc_access_connector.connector.name

  # Environment variables
  env_vars = {
    OBSERVABILITY_BACKEND = var.observability_backend
    LANGFUSE_HOST         = var.langfuse_host
    REDIS_HOST            = module.redis.host
    REDIS_PORT            = tostring(module.redis.port)
    CACHE_ENABLED         = "true"
    GCS_BUCKET            = module.storage.bucket_name
    EXECUTION_POLICY_MODE = var.execution_policy_mode
    ENVIRONMENT           = var.environment
  }

  # Secrets from Secret Manager
  secrets = {
    # Observability
    LANGFUSE_PUBLIC_KEY = {
      secret_name = module.secrets.secret_ids["langfuse-api-key"]
      version     = "latest"
    }
    LANGFUSE_SECRET_KEY = {
      secret_name = module.secrets.secret_ids["langfuse-secret-key"]
      version     = "latest"
    }

    # Live analysis/synthesis providers
    OPENAI_API_KEY = {
      secret_name = module.secrets.secret_ids["openai-api-key"]
      version     = "latest"
    }
    ANTHROPIC_API_KEY = {
      secret_name = module.secrets.secret_ids["anthropic-api-key"]
      version     = "latest"
    }

    # Live search providers
    SERPER_API_KEY = {
      secret_name = module.secrets.secret_ids["serper-api-key"]
      version     = "latest"
    }
    TAVILY_API_KEY = {
      secret_name = module.secrets.secret_ids["tavily-api-key"]
      version     = "latest"
    }
  }

  # Resource limits
  cpu_limit    = var.cloud_run_cpu
  memory_limit = var.cloud_run_memory
  timeout      = var.cloud_run_timeout

  # Job configuration
  max_retries = var.cloud_run_max_retries
  task_count  = 1
  parallelism = 1

  depends_on = [
    google_project_service.required_apis,
    module.redis,
    module.secrets,
    module.storage,
  ]
}

# IAM binding for Cloud Scheduler to invoke the job
resource "google_cloud_run_v2_job_iam_member" "scheduler_invoker" {
  count = var.enable_scheduler ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = module.cloud_run_job.job_name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Cloud Scheduler (optional - for periodic triggers)
resource "google_cloud_scheduler_job" "enrichment_hourly" {
  count       = var.enable_scheduler ? 1 : 0
  name        = "${var.environment}-evidence-enrichment-hourly"
  description = "Trigger evidence enrichment job every hour"
  schedule    = var.scheduler_cron
  time_zone   = var.scheduler_timezone
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${module.cloud_run_job.job_name}:run"

    # Override default CMD with production workload
    body = base64encode(jsonencode({
      overrides = {
        containerOverrides = [{
          args = [
            "run",
            "--entity", "examples/microsoft.json",
            "--field", "hq_country",
            "--mode", "auto"
          ]
        }]
      }
    }))

    headers = {
      "Content-Type" = "application/json"
    }

    oauth_token {
      service_account_email = google_service_account.cloud_run_sa.email
    }
  }

  depends_on = [
    google_project_service.required_apis,
    module.cloud_run_job,
    google_cloud_run_v2_job_iam_member.scheduler_invoker,
  ]
}
