# ── sec-rag-demo convenience targets ────────────────────────────────────────
.PHONY: help up down logs ps ingest ingest-clean venv install link-docs api ui ui-install

DOCS_SRC ?= /Users/manjusri/learning/generative_ai/KG-RAG-datasets/sec-10-q/data/v1/docs

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────────
up: ## Start Weaviate + Airflow (first run does DB init)
	docker compose up -d --wait

down: ## Stop all containers
	docker compose down

logs: ## Follow all container logs
	docker compose logs -f

ps: ## Show running containers
	docker compose ps

# ── Airflow ───────────────────────────────────────────────────────────────────
trigger: ## Trigger the ingestion DAG manually
	docker compose exec airflow-scheduler \
	  airflow dags trigger sec_10q_ingestion

# ── Local (no Docker) ─────────────────────────────────────────────────────────
venv: ## Create a Python virtual environment
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip

install: ## Install Python dependencies into .venv
	.venv/bin/pip install -r requirements.txt

ingest: ## Run ingestion pipeline locally (needs Weaviate running)
	.venv/bin/python scripts/run_ingestion.py

ingest-clean: ## Drop collection and re-ingest from scratch
	.venv/bin/python scripts/run_ingestion.py --recreate

api: ## Run the FastAPI server locally (uvicorn --reload, port 8000)
	.venv/bin/uvicorn src.api.main:app --reload --port 8000

ui: ## Run the Vite dev server for the evaluation frontend (port 5173)
	cd frontend && npm run dev

ui-install: ## Install frontend dependencies
	cd frontend && npm install

# ── Setup ─────────────────────────────────────────────────────────────────────
link-docs: ## Symlink the SEC corpus into ./data/docs
	mkdir -p data
	ln -sfn $(DOCS_SRC) data/docs
	@echo "Linked $(DOCS_SRC) → data/docs"

env: ## Copy .env.example to .env (edit it after)
	cp -n .env.example .env && echo "Created .env – fill in OPENAI_API_KEY"
