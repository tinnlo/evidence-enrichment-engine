# GCP Production Deployment Guide

This guide shows how to deploy the evidence-enrichment-engine to Google Cloud Platform (GCP) using Cloud Run Jobs, Memorystore Redis, Secret Manager, and Terraform.

## Architecture Overview

```
┌─────────────────┐
│ Cloud Scheduler │ (periodic triggers)
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────────┐
│ Cloud Run Job   │─────>│ Memorystore      │
│ (batch pipeline)│      │ Redis (cache)    │
└────────┬────────┘      └──────────────────┘
         │
         ├──────────────> Secret Manager (6 secrets: Langfuse + 4 provider API keys)
         │
         ├──────────────> GCS (trace artifacts)
         │
         └──────────────> Cloud Logging + Cloud Monitoring
```

## Prerequisites

- GCP project with billing enabled
- `gcloud` CLI installed and authenticated
- Terraform >= 1.5.0
- Docker installed locally

**Set up environment variables:**

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export ENVIRONMENT="dev"  # or staging, prod
```

## Deployment Options

### Option 1: Cloud Run Jobs (Batch processing) - **Recommended**

Best for: Scheduled batch enrichment runs, periodic processing.

**Characteristics:**
- Runs to completion then exits
- Triggered via Cloud Scheduler, Pub/Sub, or manual execution
- Suitable for scheduled batch processing
- Cost: Pay only for execution time

This guide uses **Cloud Run Jobs** as it matches the CLI-based batch processing architecture.

## Step 1: Prepare the Container Image

### 1.1 Build and Test Locally

```bash
# Build the Docker image
docker build -t evidence-enrichment-engine:latest .

# Test locally with replay mode (no credentials required)
docker run --rm evidence-enrichment-engine:latest \
  evidence-enrich demo --mode replay
```

### 1.2 Push to Google Artifact Registry

```bash
# Set your GCP project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export REPO_NAME="evidence-enrichment"

# Enable required APIs
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  redis.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Evidence enrichment engine container images"

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Tag and push the image
docker tag evidence-enrichment-engine:latest \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:latest

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:latest
```

## Step 2: Set Up Infrastructure with Terraform

### 2.1 Directory Structure

```
terraform/
├── gcp/
│   ├── main.tf              # Root module
│   ├── variables.tf         # Input variables
│   ├── outputs.tf           # Output values
│   ├── terraform.tfvars     # Variable values (gitignored)
│   └── modules/
│       ├── cloud-run/       # Cloud Run job module
│       ├── memorystore/     # Redis cache module
│       ├── secrets/         # Secret Manager module
│       └── storage/         # GCS bucket module
└── environments/
    ├── dev.tfvars
    ├── staging.tfvars
    └── prod.tfvars
```

### 2.2 Core Infrastructure Components

The Terraform configuration provisions:

1. **Cloud Run Job** — Runs the enrichment pipeline as a batch job
2. **Memorystore Redis** — Caching layer (24h document fetch, 7d evidence assessment)
3. **Secret Manager** — Stores API keys (Langfuse, OpenAI, Anthropic, Serper, Tavily)
4. **GCS Bucket** — Stores trace artifacts and resolved contexts
5. **Service Account** — IAM identity for the Cloud Run Job with least-privilege permissions
6. **VPC Connector** — Connects the Cloud Run Job to Memorystore Redis (private IP)
7. **Cloud Scheduler** — Optional periodic job triggers

### 2.3 Deploy Infrastructure

```bash
cd terraform/gcp

# Initialize Terraform
terraform init

# Review the plan
terraform plan -var-file=../environments/dev.tfvars

# Apply (creates all infrastructure)
terraform apply -var-file=../environments/dev.tfvars
```

**Expected resources created:**
- Cloud Run job: `${ENVIRONMENT}-evidence-enrichment-job`
- Memorystore Redis instance: `${ENVIRONMENT}-evidence-cache` (1GB Basic tier for dev, 5GB Standard for prod)
- Secret Manager secrets: `${ENVIRONMENT}-langfuse-api-key`, `${ENVIRONMENT}-langfuse-secret-key`, `${ENVIRONMENT}-openai-api-key`, `${ENVIRONMENT}-anthropic-api-key`, `${ENVIRONMENT}-serper-api-key`, `${ENVIRONMENT}-tavily-api-key`
- GCS bucket: `${PROJECT_ID}-${ENVIRONMENT}-evidence-traces`
- Service account: `evidence-enrichment-sa@${PROJECT_ID}.iam.gserviceaccount.com`
- VPC Connector: `${ENVIRONMENT}-evidence-connector`
- Cloud Scheduler job: `${ENVIRONMENT}-evidence-enrichment-hourly` (if enabled)

## Step 3: Configure Secrets

### 3.1 Store API Keys in Secret Manager

Terraform creates empty secrets. Add the actual values:

```bash
# Observability (required)
echo -n "pk-lf-..." | gcloud secrets versions add ${ENVIRONMENT}-langfuse-api-key --data-file=-
echo -n "sk-lf-..." | gcloud secrets versions add ${ENVIRONMENT}-langfuse-secret-key --data-file=-

# Live mode providers (at least one from each category required)
# Analysis/Synthesis: OpenAI or Anthropic
echo -n "sk-..." | gcloud secrets versions add ${ENVIRONMENT}-openai-api-key --data-file=-
echo -n "sk-ant-..." | gcloud secrets versions add ${ENVIRONMENT}-anthropic-api-key --data-file=-

# Search: Serper or Tavily
echo -n "..." | gcloud secrets versions add ${ENVIRONMENT}-serper-api-key --data-file=-
echo -n "tvly-..." | gcloud secrets versions add ${ENVIRONMENT}-tavily-api-key --data-file=-
```

**Note**: Terraform automatically grants the service account `secretAccessor` role on all secrets.

### 3.2 Environment Variables in Cloud Run Job

The Terraform configuration automatically sets these environment variables:

```bash
# Observability
OBSERVABILITY_BACKEND=langfuse
LANGFUSE_PUBLIC_KEY=<from Secret Manager>
LANGFUSE_SECRET_KEY=<from Secret Manager>
LANGFUSE_HOST=https://cloud.langfuse.com

# Live mode providers
OPENAI_API_KEY=<from Secret Manager>
ANTHROPIC_API_KEY=<from Secret Manager>
SERPER_API_KEY=<from Secret Manager>
TAVILY_API_KEY=<from Secret Manager>

# Infrastructure
REDIS_HOST=<Memorystore Redis IP>
REDIS_PORT=6379
CACHE_ENABLED=true
GCS_BUCKET=${PROJECT_ID}-${ENVIRONMENT}-evidence-traces
EXECUTION_POLICY_MODE=audit  # or "enforce" for prod
```

## Step 4: Deploy the Application

### 4.1 Manual Job Update (for testing)

After Terraform creates the job, you can update it manually:

```bash
gcloud run jobs update ${ENVIRONMENT}-evidence-enrichment-job \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:latest \
  --region=${REGION}
```

### 4.2 Automated Deployment (CI/CD)

For automated deployment, you can create a CI/CD workflow (e.g., `.github/workflows/deploy-gcp.yml`). A typical workflow would include:

```yaml
# Example workflow steps (not yet implemented)
- Build Docker image
- Push to Artifact Registry
- Deploy to Cloud Run Jobs (dev/staging/prod)
- Run smoke tests
- Notify on Slack
```

## Step 5: Set Up Scheduled Triggers

### 5.1 Cloud Scheduler for Periodic Job Execution

```bash
# Cloud Scheduler automatically created by Terraform if enable_scheduler = true
# To manually trigger the job:
gcloud run jobs execute ${ENVIRONMENT}-evidence-enrichment-job \
  --region=${REGION} \
  --wait

# To view scheduled executions:
gcloud scheduler jobs list --location=${REGION}

# To manually trigger the scheduler:
gcloud scheduler jobs run ${ENVIRONMENT}-evidence-enrichment-hourly \
  --location=${REGION}
```

### 5.2 Pub/Sub Event-Driven Triggers

```bash
# Create a Pub/Sub topic
gcloud pubsub topics create evidence-enrichment-requests

# Create a push subscription that triggers the job
# Note: Pub/Sub to Cloud Run Jobs requires Cloud Run Admin API
gcloud pubsub subscriptions create evidence-enrichment-sub \
  --topic=evidence-enrichment-requests \
  --push-endpoint=https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${ENVIRONMENT}-evidence-enrichment-job:run \
  --push-auth-service-account=evidence-enrichment-sa@${PROJECT_ID}.iam.gserviceaccount.com

# Publish a test message
gcloud pubsub topics publish evidence-enrichment-requests \
  --message='{"entity_id": "microsoft", "field": "hq_country", "mode": "auto"}'
```

## Step 6: Observability and Monitoring

### 6.1 Cloud Logging

View logs in Cloud Console or via CLI:

```bash
# Stream logs from the job
gcloud run jobs logs tail ${ENVIRONMENT}-evidence-enrichment-job --region=${REGION}

# Query specific errors
gcloud logging read "resource.type=cloud_run_job AND severity>=ERROR" \
  --limit=50 \
  --format=json
```

### 6.2 Cloud Monitoring

Key metrics to monitor:

- **Execution count** — `run.googleapis.com/job/completed_execution_count`
- **Execution time** — `run.googleapis.com/job/execution_time`
- **Container CPU utilization** — `run.googleapis.com/container/cpu/utilizations`
- **Container memory utilization** — `run.googleapis.com/container/memory/utilizations`

### 6.3 Langfuse Traces

All pipeline runs are traced to Langfuse (when `OBSERVABILITY_BACKEND=langfuse`):

- Navigate to https://cloud.langfuse.com
- View traces by `trace_id`, `entity_id`, or `field`
- Analyze span-level latency, cost, and cache hit rates

### 6.4 Custom Dashboards

Create a Cloud Monitoring dashboard with:

- Job execution count and success rate
- Execution duration (p50, p95, p99)
- Task failure rate and retry count
- Cache hit rate (from Langfuse span metadata)
- Cost per enrichment (from `finops_summary.json`)
- Execution policy violations (from `execution_policy.json`)

## Step 7: Cost Optimization

### 7.1 Cloud Run Job Configuration

```bash
# Configure task parallelism and retries
# For dev/staging: single task, 2 retries
task_count = 1
max_retries = 2

# For prod: consider parallel tasks if processing multiple entities
task_count = 1  # Increase if batch processing multiple entities
max_retries = 3
```

**Note:** Cloud Run Jobs don't autoscale like Services. Each execution runs a fixed number of tasks.

### 7.2 Memorystore Redis Tiers

- **Basic tier** (dev/staging): No HA, lower cost, 1GB-300GB
- **Standard tier** (prod): HA with automatic failover, 5GB-300GB

```bash
# Dev: Basic tier, 1GB
--tier=BASIC --memory-size-gb=1

# Prod: Standard tier, 5GB
--tier=STANDARD --memory-size-gb=5
```

### 7.3 GCS Lifecycle Policies

Automatically delete old trace artifacts:

```bash
# Delete traces older than 30 days
gcloud storage buckets update gs://${PROJECT_ID}-${ENVIRONMENT}-evidence-traces \
  --lifecycle-file=lifecycle.json
```

`lifecycle.json`:
```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
```

### 7.4 Budget Alerts

```bash
# Create a budget alert at 80% of monthly spend
gcloud billing budgets create \
  --billing-account=<BILLING_ACCOUNT_ID> \
  --display-name="Evidence Enrichment Budget" \
  --budget-amount=500 \
  --threshold-rule=percent=80
```

## Step 8: Multi-Environment Strategy

### 8.1 Environment Separation

Use separate GCP projects for isolation:

- **Dev**: `evidence-enrichment-dev`
- **Staging**: `evidence-enrichment-staging`
- **Prod**: `evidence-enrichment-prod`

Or use a single project with environment prefixes:

- Cloud Run jobs: `dev-evidence-enrichment-job`, `staging-evidence-enrichment-job`, `prod-evidence-enrichment-job`
- GCS buckets: `${PROJECT_ID}-dev-evidence-traces`, `${PROJECT_ID}-staging-evidence-traces`, `${PROJECT_ID}-prod-evidence-traces`

### 8.2 Terraform Workspaces

```bash
# Create workspaces
terraform workspace new dev
terraform workspace new staging
terraform workspace new prod

# Deploy to dev
terraform workspace select dev
terraform apply -var-file=../environments/dev.tfvars

# Deploy to prod
terraform workspace select prod
terraform apply -var-file=../environments/prod.tfvars
```

### 8.3 Environment-Specific Configs

`environments/dev.tfvars`:
```hcl
project_id = "evidence-enrichment-dev"
region = "us-central1"
redis_tier = "BASIC"
redis_memory_gb = 1
cloud_run_max_retries = 3
execution_policy_mode = "off"
```

`environments/prod.tfvars`:
```hcl
project_id = "evidence-enrichment-prod"
region = "us-central1"
redis_tier = "STANDARD"
redis_memory_gb = 5
cloud_run_max_retries = 3
execution_policy_mode = "enforce"
```

## Step 9: Rollback and Disaster Recovery

### 9.1 Cloud Run Job Rollback

Cloud Run Jobs don't have revisions like Services. To rollback:

```bash
# Update job to previous image
gcloud run jobs update ${ENVIRONMENT}-evidence-enrichment-job \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:previous-tag \
  --region=${REGION}

# Or use a specific image digest
gcloud run jobs update ${ENVIRONMENT}-evidence-enrichment-job \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine@sha256:abc123... \
  --region=${REGION}
```

### 9.2 Image Versioning Strategy

Use semantic versioning or commit SHAs for image tags:

```bash
# Tag with version
docker tag evidence-enrichment-engine:latest \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:v1.2.3

# Tag with commit SHA
docker tag evidence-enrichment-engine:latest \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:${GIT_SHA}

# Keep latest tag for convenience
docker tag evidence-enrichment-engine:latest \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/evidence-enrichment-engine:latest
```

### 9.3 Backup and Restore

**Memorystore Redis:**
- Automatic backups (Standard tier only)
- Manual export to GCS

```bash
gcloud redis instances export gs://${PROJECT_ID}-redis-backups/backup-$(date +%Y%m%d).rdb \
  --source=${ENVIRONMENT}-evidence-cache \
  --region=${REGION}
```

**GCS Trace Artifacts:**
- Versioning enabled by default
- Cross-region replication for prod

## Step 10: Security Hardening

### 10.1 IAM Least Privilege

Service account permissions:

```bash
# Cloud Run Job service account needs:
- roles/secretmanager.secretAccessor  # Read secrets
- roles/storage.objectAdmin           # Write to GCS
- roles/redis.editor                  # Connect to Memorystore
- roles/logging.logWriter             # Write logs
- roles/monitoring.metricWriter       # Write metrics
```

### 10.2 VPC Service Controls

For highly sensitive workloads, use VPC Service Controls to create a security perimeter:

```bash
gcloud access-context-manager perimeters create evidence-perimeter \
  --title="Evidence Enrichment Perimeter" \
  --resources=projects/<PROJECT_NUMBER> \
  --restricted-services=run.googleapis.com,redis.googleapis.com
```

### 10.3 Binary Authorization

Require signed container images:

```bash
gcloud container binauthz policy import policy.yaml
```

## Troubleshooting

### Issue: Cloud Run Job can't connect to Memorystore Redis

**Cause:** VPC Connector not configured or wrong subnet.

**Fix:**
```bash
# Verify VPC Connector exists
gcloud compute networks vpc-access connectors list --region=${REGION}

# Verify Cloud Run job uses the connector
gcloud run jobs describe ${ENVIRONMENT}-evidence-enrichment-job --region=${REGION} \
  | grep vpc-connector
```

### Issue: Secret Manager permission denied

**Cause:** Service account lacks `secretmanager.secretAccessor` role.

**Fix:**
```bash
gcloud secrets add-iam-policy-binding <SECRET_NAME> \
  --member="serviceAccount:evidence-enrichment-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Issue: Cold start latency

**Note:** Cloud Run Jobs don't have cold starts in the same way as Services. Each execution starts fresh.

**If execution time is too long:**
```bash
# Increase CPU/memory allocation
gcloud run jobs update ${ENVIRONMENT}-evidence-enrichment-job \
  --cpu=4 \
  --memory=4Gi \
  --region=${REGION}
```

### Issue: Out of memory errors

**Cause:** Default 2Gi memory insufficient for large document processing.

**Fix:**
```bash
# Increase memory allocation
gcloud run jobs update ${ENVIRONMENT}-evidence-enrichment-job \
  --memory=4Gi \
  --region=${REGION}
```

## Cost Estimation

**Monthly cost for dev environment (low traffic):**

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| Cloud Run Jobs | 1 execution/hour, 2 vCPU, 2Gi RAM, 5min/execution | $5-10 |
| Memorystore Redis | Basic tier, 1GB | $35 |
| Secret Manager | 6 secrets, 1000 accesses/month | $0.36 |
| GCS | 10GB storage, 1000 operations/month | $0.30 |
| Cloud Logging | 5GB logs/month | $2.50 |
| **Total** | | **~$43/month** |

**Monthly cost for prod environment (moderate traffic):**

| Service | Configuration | Estimated Cost |
|---------|--------------|----------------|
| Cloud Run Jobs | 1 execution/hour, 4 vCPU, 4Gi RAM, 10min/execution | $150-200 |
| Memorystore Redis | Standard tier, 5GB | $175 |
| Secret Manager | 6 secrets, 100K accesses/month | $0.54 |
| GCS | 100GB storage, 100K operations/month | $3.50 |
| Cloud Logging | 50GB logs/month | $25 |
| **Total** | | **~$354-404/month** |

## Next Steps

1. **Review Terraform modules** in `terraform/gcp/modules/`
2. **Set up CI/CD pipeline** (e.g., create `.github/workflows/deploy-gcp.yml`)
3. **Configure monitoring alerts** in Cloud Monitoring
4. **Test job execution** with different task counts and retry configurations
5. **Document runbooks** for common operational tasks

## References

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Memorystore for Redis](https://cloud.google.com/memorystore/docs/redis)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Langfuse Cloud](https://langfuse.com/docs/deployment/cloud)
