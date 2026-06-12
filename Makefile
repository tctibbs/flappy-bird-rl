.PHONY: help install lint format typecheck test check clean play train evaluate watch video

CONFIG ?= configs/default.yaml
MODEL ?= runs/latest/best_model
EPISODES ?= 5

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	uv sync --all-extras

lint: ## Run ruff lint
	uv run ruff check .

format: ## Format code and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Run ty type checking
	uv run ty check

test: ## Run pytest test suite
	uv run pytest

check: ## Run all quality gates
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check
	uv run pytest

clean: ## Remove build artifacts and cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/

play: ## Play the game yourself
	uv run flapper play

train: ## Train an agent (CONFIG=configs/default.yaml, SEED optional)
	uv run flapper train --config $(CONFIG) $(if $(SEED),--seed $(SEED))

evaluate: ## Evaluate a model (MODEL=path/to/model)
	uv run flapper evaluate --config $(CONFIG) --model $(MODEL)

watch: ## Watch a trained model play (MODEL=..., EPISODES=5)
	uv run flapper watch --model $(MODEL) --episodes $(EPISODES)

video: ## Record a model playing to mp4 (MODEL=...)
	uv run flapper video --model $(MODEL)
