setup:
	uv sync

lint:
	uv run ruff check .

.PHONY: setup lint
