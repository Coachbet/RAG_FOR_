# PitWall AI: every step of the project is one `make` target. Run `make help`.
# Full pipeline from scratch:  make install data index eval-draft  (verify questions)  sweep eval train figures space

SHELL := /bin/bash
UV    ?= uv
RUN   := $(UV) run python
SPACE ?=                       # your Hugging Face Space id, e.g. make space-push SPACE=louis/pitwall-ai
Q     ?= What must an overtaking car achieve on the inside of a corner?
SEASONS ?=                     # e.g. make fetch SEASONS=2025 (default: enabled rows of data/sources/seasons.csv)

.DEFAULT_GOAL := help
.PHONY: help install env data fetch prepare index ask eval-draft eval-merge sweep retrieval eval eval-baseline \
        train train-curve figures app test lint format space space-push clean clean-all

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nPitWall AI targets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[33m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------- setup
install: ## Create the virtual env and install all dependencies (uv sync)
	$(UV) sync
	@test -f .env || (cp .env.example .env && echo "Created .env: put your GROQ_API_KEY in it")

env: ## Check the configuration (without printing secrets)
	@$(RUN) -c "from pitwall.config import settings as s; print({k: ('***' if 'key' in k and v else v) for k, v in vars(s).items()})"

# ---------------------------------------------------------------- data
fetch: ## Download the FIA rulebooks and stewards' decisions -> data/raw/fia/ (cached, 1 request / 10 s)
	$(RUN) scripts/fetch_fia.py $(if $(strip $(SEASONS)),--seasons $(strip $(SEASONS)),)

prepare: ## Parse, keep on-track incidents, anonymise, label, split -> data/processed/ and data/ft/ (offline)
	$(RUN) scripts/prepare_data.py

data: fetch prepare ## fetch + prepare

# ---------------------------------------------------------------- RAG

index: ## Chunk + embed + build the FAISS index -> data/index/
	$(RUN) scripts/build_index.py

ask: ## Ask one question: make ask Q="What is the penalty for causing a collision?"
	$(RUN) scripts/ask.py "$(Q)"

eval-draft: ## Draft candidate eval questions with the LLM (to VERIFY BY HAND)
	$(RUN) scripts/draft_eval_questions.py --n 30

eval-merge: ## Append the drafts marked "verified": true to eval/questions.jsonl
	$(RUN) scripts/draft_eval_questions.py --merge

sweep: ## Calibrate the refusal threshold MIN_SCORE on the DEV questions
	$(RUN) scripts/run_eval.py --sweep

retrieval: ## Retrieval metrics on TEST: hit rate@k, MRR, dense vs hybrid (no LLM calls)
	$(RUN) scripts/run_eval.py --retrieval-only

eval: retrieval ## Full RAG evaluation on the TEST split (retrieval + answers + refusals)
	$(RUN) scripts/run_eval.py

eval-baseline: ## Same, plus the LLM alone without RAG (what RAG adds)
	$(RUN) scripts/run_eval.py --baseline

# ---------------------------------------------------------------- fine-tuning
train: ## Fine-tune DistilBERT and compare with the baselines -> models/classifier/
	$(RUN) scripts/train_classifier.py

train-curve: ## Same + learning curve (50 / 150 / all training examples)
	$(RUN) scripts/train_classifier.py --curve 50,150

# ---------------------------------------------------------------- report, demo, quality
figures: ## Figures for the report -> reports/figures/
	$(RUN) scripts/make_figures.py

app: ## Run the Gradio demo locally (http://127.0.0.1:7860)
	$(RUN) app.py

test: ## Unit tests (offline, no model download, no API call)
	$(UV) run pytest -q

lint: ## Ruff linter
	$(UV) run ruff check src scripts tests app.py

format: ## Ruff formatter
	$(UV) run ruff format src scripts tests app.py

# ---------------------------------------------------------------- deployment
space: ## Build dist/space/ (app, index, classifier, requirements.txt) for the HF Space
	$(RUN) scripts/prepare_space.py

space-push: space ## Upload dist/space/ to the Hugging Face Space: make space-push SPACE=user/pitwall-ai
	@test -n "$(strip $(SPACE))" || (echo "Usage: make space-push SPACE=<hf-username>/pitwall-ai" && exit 1)
	$(UV) run hf upload $(strip $(SPACE)) dist/space . --repo-type space

clean: ## Remove caches and build outputs (keeps data, index and models)
	rm -rf dist .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-all: clean ## Also remove the downloaded data, the index and the trained model
	rm -rf data/raw/fia data/processed data/index/* models
	find data/ft -name '*.jsonl' -delete; rm -f data/ft/label_review.csv
