# Evidence Enrichment Engine - GCP Infrastructure

This Terraform configuration deploys the evidence-enrichment-engine to Google Cloud Platform.

## Architecture

- **Cloud Run Job** — Runs the enrichment pipeline as batch jobs
- **Memorystore Redis** — Caching layer (24h document fetch, 7d evidence assessment)
- **Secret Manager** — Stores Langfuse API keys and other secrets
- **GCS Bucket** — Stores trace artifacts and resolved contexts
- **VPC Connector** — Connects Cloud Run to Memorystore Redis (private IP)
- **Service Account** — IAM identity with least-privilege permissions
- **Cloud Scheduler** — Optional periodic job execution

## Prerequisites

- Terraform >= 1.5.0
- GCP project with billing enabled
- `gcloud` CLI authenticated

## Quick Start

```bash
# Initialize Terraform
terraform init

# Copy example variables
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your values
vim terraform.tfvars

# Review the plan
terraform plan

# Apply
terraform apply
```

## Multi-Environment Deployment

```bash
# Deploy to dev
terraform workspace new dev
terraform apply -var-file=../environments/dev.tfvars

# Deploy to staging
terraform workspace new staging
terraform apply -var-file=../environments/staging.tfvars

# Deploy to prod
terraform workspace new prod
terraform apply -var-file=../environments/prod.tfvars
```

## Outputs

After `terraform apply`, you'll get:

- `cloud_run_job_name` — Cloud Run job name
- `cloud_run_job_id` — Cloud Run job ID
- `redis_host` — Memorystore Redis IP address
- `gcs_bucket_name` — GCS bucket for trace artifacts
- `service_account_email` — Service account email

## Cost Estimation

Run `terraform plan` to see estimated costs, or use:

```bash
terraform plan -out=plan.tfplan
terraform show -json plan.tfplan | jq > plan.json
# Upload plan.json to https://cost.modules.tf/
```

## Cleanup

```bash
terraform destroy
```

**Warning:** This will delete all resources including data in GCS and Redis cache.
