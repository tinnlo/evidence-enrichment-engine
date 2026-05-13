output "bucket_name" {
  description = "GCS bucket name"
  value       = google_storage_bucket.traces.name
}

output "bucket_url" {
  description = "GCS bucket URL"
  value       = google_storage_bucket.traces.url
}

output "bucket_self_link" {
  description = "GCS bucket self link"
  value       = google_storage_bucket.traces.self_link
}
