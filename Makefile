.PHONY: install dev data train eval serve test lint loadtest charts docker-up docker-down

install:        ## install everything (CPU torch)
	pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3"
	pip install -r requirements.txt -r requirements-dev.txt

data:           ## generate the synthetic workplace fine-tuning dataset
	python training/make_dataset.py --n 1200

train: data     ## LoRA fine-tune (CPU friendly, ~30-60 min)
	python training/train_lora.py --epochs 1

eval:           ## held-out hallucination evaluation (base vs LoRA, with/without RAG)
	python eval/hallucination_eval.py

serve:          ## run the API locally on :8000
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:           ## unit + API tests (no model download needed)
	LLM_BACKEND=echo REDIS_URL= pytest

lint:
	ruff check app tests training eval loadtest scripts

loadtest:       ## concurrent-user load test against a running API
	python loadtest/load_test.py --scenario cold  --users 1 2 4 --requests 5
	python loadtest/load_test.py --scenario mixed --users 1 4 8 16 --requests 10
	python loadtest/load_test.py --scenario stream --users 1 --requests 10

charts:         ## turn results/*.json into the PNG charts used in the README
	python scripts/make_charts.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
