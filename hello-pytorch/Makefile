VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: setup train clean

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

train:
	$(PYTHON) train.py

clean:
	rm -rf $(VENV)
