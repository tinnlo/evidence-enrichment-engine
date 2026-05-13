resource "google_cloud_run_v2_job" "job" {
  name     = var.job_name
  location = var.region

  template {
    template {
      service_account = var.service_account_email

      vpc_access {
        connector = var.vpc_connector_name
        egress    = "PRIVATE_RANGES_ONLY"
      }

      max_retries = var.max_retries
      timeout     = "${var.timeout}s"

      containers {
        image = var.image

        resources {
          limits = {
            cpu    = var.cpu_limit
            memory = var.memory_limit
          }
        }

        # Environment variables
        dynamic "env" {
          for_each = var.env_vars
          content {
            name  = env.key
            value = env.value
          }
        }

        # Secrets from Secret Manager
        dynamic "env" {
          for_each = var.secrets
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = env.value.secret_name
                version = env.value.version
              }
            }
          }
        }
      }
    }

    task_count  = var.task_count
    parallelism = var.parallelism
  }

  lifecycle {
    ignore_changes = [
      template[0].template[0].containers[0].image, # Allow CI/CD to update image
    ]
  }
}
