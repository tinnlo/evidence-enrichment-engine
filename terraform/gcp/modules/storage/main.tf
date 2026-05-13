resource "google_storage_bucket" "traces" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = var.versioning
  }

  lifecycle_rule {
    condition {
      age = var.lifecycle_age
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    managed_by  = "terraform"
    service     = "evidence-enrichment"
  }
}

# Grant service account access to bucket
resource "google_storage_bucket_iam_member" "bucket_access" {
  bucket = google_storage_bucket.traces.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${var.service_account_email}"
}
