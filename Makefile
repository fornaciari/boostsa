.PHONY: install test

default: test

install:
    pip install --upgrade .

test:
    PYTHONPATH=. pytest