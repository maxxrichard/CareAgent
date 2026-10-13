.PHONY: install dev lint test index seed serve chat eval docker-build docker-up eval-set

install:
	pip install -e ".[embeddings,tracking]"

dev:
	pip install -e ".[all]"
	pre-commit install

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

test:
	pytest -q

index:
	python scripts/build_index.py

seed:
	careagent seed

eval-set:
	python scripts/generate_eval_set.py 500

serve:
	careagent serve --reload

chat:
	careagent chat

eval:
	python scripts/run_eval.py --backends anthropic,openai,ollama

docker-build:
	docker build -t careagent:latest .

docker-up:
	docker compose up --build
