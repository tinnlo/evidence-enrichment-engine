resource "google_redis_instance" "cache" {
  name               = var.instance_name
  tier               = var.tier
  memory_size_gb     = var.memory_size_gb
  region             = var.region
  redis_version      = var.redis_version
  authorized_network = var.authorized_network

  # Display name
  display_name = "${var.environment} Evidence Enrichment Cache"

  # Redis configuration
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }

  # Maintenance policy
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }

  # Labels
  labels = {
    environment = var.environment
    managed_by  = "terraform"
    service     = "evidence-enrichment"
  }

  lifecycle {
    prevent_destroy = false
  }
}
