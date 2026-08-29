.PHONY: help install install-cpu install-gpu install-hf install-cpt install-dev test lint typecheck format clean

help:
	@echo "QwenDopamine src/Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install       - install core package"
	@echo "  install-cpu   - install with CPU extras"
	@echo "  install-gpu   - install with GPU extras (unsloth, fla, trl)"
	@echo "  install-hf    - install with HF extras (datasets, hf hub, tokenizers)"
	@echo "  install-cpt   - install with CPT notebook extras"
	@echo "  install-dev   - install with dev extras (pyrefly, pytest, ruff)"
	@echo "  test          - run pytest"
	@echo "  lint          - run ruff"
	@echo "  typecheck     - run pyrefly"
	@echo "  format        - format code with ruff"
	@echo "  clean         - remove build artifacts"

install:
	uv pip install -e .

install-cpu:
	uv pip install -e ".[cpu]"

install-gpu:
	uv pip install -e ".[gpu]"

install-hf:
	uv pip install -e ".[hf]"

install-cpt:
	uv pip install -e ".[cpt]"

install-dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest -v

test-slow:
	uv run pytest -v -m slow

test-fast:
	uv run pytest -v -m "not slow"

lint:
	uv run ruff check .

typecheck:
	uv run pyrefly check

format:
	uv run ruff format .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .pyrefly/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
