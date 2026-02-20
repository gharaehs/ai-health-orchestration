.PHONY: help start stop logs download-model test-llm

help:
	@echo "Available commands:"
	@echo "  make download-model  - Download Mistral 7B AWQ"
	@echo "  make start           - Start all services"
	@echo "  make stop            - Stop all services"
	@echo "  make logs            - Follow logs"
	@echo "  make test-llm        - Send a test prompt to the LLM"

download-model:
	bash scripts/download_model.sh

start:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

test-llm:
	curl -s http://localhost:8000/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{"model": "/models/mistral", "messages": [{"role": "user", "content": "Say hello!"}], "max_tokens": 50}' \
	  | python3 -m json.tool
