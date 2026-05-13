output "host" {
  description = "Redis instance host IP address"
  value       = google_redis_instance.cache.host
}

output "port" {
  description = "Redis instance port"
  value       = google_redis_instance.cache.port
}

output "instance_id" {
  description = "Redis instance ID"
  value       = google_redis_instance.cache.id
}

output "connection_string" {
  description = "Redis connection string"
  value       = "${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"
}
