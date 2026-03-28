docker-up:
	docker compose up

docker-test:
	docker compose run --rm pipeline pytest tests/

docker-eval:
	docker compose run --rm pipeline evidence-enrich eval
