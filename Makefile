.PHONY: help install lint format test check clean run train train-watch watch

# Default config for training
CONFIG ?= configs/default.yaml
# Default model path for watching
MODEL ?= models/dqn_flappybird
# Default number of episodes
EPISODES ?= 5

# Show available commands
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Install dependencies using uv
install: ## Install project dependencies
	uv sync --all-extras

# Run linting checks
lint: ## Run linting and static analysis
	uv run ruff check .

# Format code with ruff
format: ## Format code and fix linting issues
	uv run ruff format .
	uv run ruff check --fix .

# Run test suite (placeholder for when tests are added)
test: ## Run pytest test suite
	@echo "No tests configured yet"

# Run all quality gates
check: ## Run linting and tests
	$(MAKE) lint
	$(MAKE) test

# Clean build artifacts and cache files
clean: ## Remove build artifacts and cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/

# Default target - show help
default:
	@make help

# Run human playable mode
run: ## Run game in human playable mode
	uv run python src/main.py human

# Train a new agent
train: ## Train agent (CONFIG=configs/default.yaml)
	uv run python src/main.py agent_training --config $(CONFIG)

# Train with visualization
train-watch: ## Train agent with rendering (CONFIG=configs/default.yaml)
	uv run python src/main.py agent_training --config $(CONFIG) --render

# Watch a trained agent play
watch: ## Watch trained agent play (MODEL=models/dqn_flappybird EPISODES=5)
	uv run python src/main.py agent --model $(MODEL) --episodes $(EPISODES)
