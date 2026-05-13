# Architecture Decisions

## Cloud Run Jobs for GCP Deployment

### Context

The evidence-enrichment-engine is fundamentally a CLI-based batch processor designed for scheduled enrichment runs. When deploying to GCP, we needed to choose between:

1. **Cloud Run Service** - Long-running HTTP server with autoscaling
2. **Cloud Run Jobs** - Batch execution that runs to completion
3. **Compute Engine** - Traditional VMs with manual orchestration

### Decision

We chose **Cloud Run Jobs** as the deployment target.

### Rationale

**Architectural Fit:**
- The application is a CLI tool (`evidence-enrich run`), not an HTTP API
- Designed for batch processing with clear start/completion boundaries
- No need for request/response patterns or persistent connections
- Cloud Run Jobs match this execution model perfectly

**Cost Efficiency:**
- Pay only for execution time (no idle instance costs)
- No minimum instance requirements
- Automatic scale-to-zero between executions
- Dev environment: ~$5-10/month vs ~$50-100/month for always-on Service
- Prod environment: ~$150-200/month vs ~$300-400/month for Service with min instances

**Operational Simplicity:**
- No HTTP server code required
- No health check endpoints to maintain
- No traffic routing or revision management
- Simpler deployment model (update image, execute job)

**Integration:**
- Cloud Scheduler triggers jobs via OAuth token authentication
- Pub/Sub can trigger jobs for event-driven workflows
- Manual execution via `gcloud run jobs execute` for testing
- Same VPC connector, Secret Manager, and Redis access as Services

### Trade-offs

**What we gain:**
- ✅ Perfect architectural match for CLI batch processing
- ✅ Lower cost for scheduled workloads
- ✅ Simpler codebase (no HTTP layer)
- ✅ Automatic retry on failure
- ✅ Clear execution boundaries and logs

**What we lose:**
- ❌ No real-time API access (must use scheduled or event triggers)
- ❌ Cannot expose HTTP endpoints for external integrations
- ❌ No autoscaling based on traffic (fixed task count per execution)

### When to Reconsider

Consider switching to Cloud Run Service if:
- You need real-time API access for external systems
- You want to build a web UI that calls enrichment on-demand
- You need synchronous request/response patterns
- You plan to expose the pipeline as a general-purpose service

For the current use case (scheduled batch enrichment with observability), Cloud Run Jobs remain the optimal choice.

### Implementation

See [deployment_gcp.md](deployment_gcp.md) for complete deployment guide.

**Key components:**
- Terraform infrastructure in `terraform/gcp/`
- Cloud Run Job with VPC connector for Redis access
- Cloud Scheduler for periodic triggers
- Secret Manager for API keys (Langfuse + 4 provider keys)
- Memorystore Redis for caching (24h documents, 7d evidence)
- GCS bucket for trace artifacts

**Deployment status:** Production-ready, validated with `terraform validate`

---

## Langfuse as Primary Observability Backend

### Context

The pipeline supports multiple observability backends: Langfuse, LangSmith, dual (both), and none (local only).

### Decision

We position **Langfuse** as the primary documented remote tracing path.

### Rationale

**Privacy-first defaults:**
- `TRACE_REDACT_VALUES=true` by default
- Sensitive data stays local unless explicitly relaxed
- Clear privacy controls in documentation

**Feature completeness:**
- Full span hierarchy support
- Cost tracking and attribution
- Session grouping and filtering
- Public cloud offering with generous free tier

**Integration quality:**
- Native Python SDK with async support
- Automatic context propagation
- Metadata and tag support
- Works with LangGraph state graphs

**Documentation clarity:**
- Single primary path reduces decision paralysis
- LangSmith remains supported for existing users
- Dual mode available for migration scenarios

See [observability.md](observability.md) for runtime behavior and configuration.

---

## Redis Caching Strategy

### Context

The pipeline makes multiple external calls (search, fetch, LLM analysis) that are expensive and slow.

### Decision

Use **Memorystore Redis** with differentiated TTLs:
- Document fetch: 24 hours
- Evidence assessment: 7 days

### Rationale

**Document fetch (24h TTL):**
- Web content changes frequently
- Stale content can lead to incorrect enrichments
- 24h balances freshness with cost savings
- Cache hit rate: ~60-70% for repeated entities

**Evidence assessment (7d TTL):**
- LLM-generated assessments are expensive ($0.01-0.05 per call)
- Assessment quality is stable for a given document version
- 7d TTL captures most re-runs during development/testing
- Cache hit rate: ~80-90% for repeated fields

**Infrastructure choice:**
- Basic tier (1GB) for dev: $35/month, no HA
- Standard tier (5GB) for prod: $175/month, automatic failover
- VPC connector required for Cloud Run access

**Alternative considered:**
- Cloud Memorystore for Memcached: Lower cost but no persistence
- Redis on Compute Engine: More operational overhead
- No caching: 3-5x higher LLM costs, 2-3x slower execution

See cost analysis in [deployment_gcp.md](deployment_gcp.md#cost-estimation).

---

## Execution Policy as Capability Gate

### Context

The pipeline can perform live actions (search, fetch, retrieval, remote tracing) that have cost, privacy, and operational implications.

### Decision

Implement **Execution Policy** as a separate governance layer from FinOps:
- `off` - All live actions permitted (dev default)
- `audit` - Log policy violations but allow (staging default)
- `enforce` - Block disallowed actions (prod default)

### Rationale

**Separation of concerns:**
- FinOps controls cost budgets and spending
- Execution Policy controls which capabilities are permitted
- Independent configuration allows different governance models

**Use cases:**
- Disable live search in CI/CD (use replay only)
- Block remote tracing in privacy-sensitive environments
- Restrict retrieval to local mode only
- Audit live actions in staging before prod rollout

**Implementation:**
- Policy defined in `execution_policy.json` artifact
- Checked before each live action
- Violations logged to observability backend
- Clear error messages when blocked

See `tests/test_execution_policy.py` for behavior verification.
