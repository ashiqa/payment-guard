.PHONY: setup demo train test serve benchmark
setup:
	python -m pip install -r requirements.lock
	python -m pip install --no-deps --no-build-isolation -e .
demo:
	paymentguard generate
	paymentguard train --data-kind synthetic
train:
	paymentguard train --data-kind synthetic
test:
	python -m unittest discover -s tests -v
serve:
	uvicorn paymentguard.api:app --host 127.0.0.1 --port 8000
benchmark:
	python scripts/benchmark.py
