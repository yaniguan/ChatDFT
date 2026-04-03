.PHONY: demo benchmark showcase figures test ui install clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

demo:  ## Run interactive demo (no database needed)
	python demo.py

benchmark:  ## Run full benchmark suite (7 figures)
	python demo.py --benchmark

showcase:  ## Generate all showcase figures (15 figures)
	python -m science.benchmarks.showcase

figures: benchmark showcase  ## Generate ALL figures (22 total)
	@echo "All figures saved to figures/"

test:  ## Run all tests
	pytest tests/ -v --tb=short

ui:  ## Launch Streamlit web interface
	streamlit run client/app.py

install:  ## Install all dependencies
	pip install -e ".[dev,docs]"

clean:  ## Remove generated files
	rm -rf figures/ results/ __pycache__ .pytest_cache
	find . -name '*.pyc' -delete
