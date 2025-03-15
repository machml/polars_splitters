.PHONY: lint format

lint:
	ruff check --fix .

format:
	ruff format .

build:
	rm -rf dist
	uv build

publish:
	uv publish