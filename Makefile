.PHONY: lint format clean build tag publish tidy

# extract current version from pyproject.toml
VERSION := $(shell sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)

# default tag message
TAG_MESSAGE ?= "Release v$(VERSION)"

lint:
	ruff check --fix . --exclude **/notebooks/*

format:
	ruff format .
	mdformat .

tidy:
	$(MAKE) format
	$(MAKE) lint

clean:
	rm -rf dist

build: clean
	uv build --no-sources


build-timestamped: clean
	uv build --no-sources
	@TIMESTAMP=$$(date +%Y%m%d%H%M%S) && \
	cd dist && \
	for wheel in *.whl; do \
		base=$$(echo "$$wheel" | sed 's/-py3-none-any\.whl$$//') && \
		mv "$$wheel" "$${base}-$${TIMESTAMP}-py3-none-any.whl"; \
	done


tag: # create a git tag with the current version
	@echo "Tagging version v$(VERSION)"
	git tag -a "v$(VERSION)" -m $(TAG_MESSAGE)
	git push origin "v$(VERSION)"

publish: tag
	uv publish