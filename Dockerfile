FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[live,observability]"

# Cloud Run Jobs execute the container and expect it to exit
# Override CMD via Cloud Scheduler body or gcloud --args flag:
#   gcloud run jobs execute JOB --args="run,--entity,examples/microsoft.json,--field,hq_country,--mode,auto"
# Default: run demo in replay mode for testing
ENTRYPOINT ["evidence-enrich"]
CMD ["demo", "--mode", "replay"]
