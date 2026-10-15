.PHONY: help start stop logs test test-llm

help:
	@echo "Available commands:"
	@echo "  make start           - Start all services"
	@echo "  make stop            - Stop all services"
	@echo "  make logs            - Follow logs"
	@echo "  make test            - Run offline unit tests"
	@echo "  make test-llm        - Send a tiny real prompt to the Anthropic API"

start:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

test:
	python3 -m pytest tests/

test-llm:
	python3 scripts/test_llm.py
