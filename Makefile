# We can create commands to be used with `make` in a terminal

# Installing libraries
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Applying some tests
test:
	python -m pytest -vv test_*.py

# Verifying formatting
format:
	black *.py

# Verifying code standards
lint:
	pylint --disable=R,C,E1120 *.py

# Do everything at once
all: install lint test format