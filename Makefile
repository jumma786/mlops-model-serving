.PHONY: install train test run docker-build docker-run clean

install:
	pip install -r requirements.txt

train:
	python src/model/train.py --data-path data/bank-additional-full.csv

train-synthetic:
	python src/model/train.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t mlops-model-serving:latest .

docker-run:
	docker run -p 8000:8000 mlops-model-serving:latest

clean:
	rm -rf artifacts/ __pycache__
	find . -name "*.pyc" -delete

help:
	@echo "Commands:"
	@echo "  make install      Install dependencies"
	@echo "  make train        Train model on real UCI data"
	@echo "  make test         Run 11 API tests"
	@echo "  make run          Start FastAPI server"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"
