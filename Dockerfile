FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[dev,live]"

CMD ["evidence-enrich", "demo", "--mode", "replay"]
