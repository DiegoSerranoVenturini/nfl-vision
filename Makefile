help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "build - package"

all: default

default: clean dev_deps deps build

.venv:
	if [ ! -e ".venv/bin/activate_this.py" ] ; then virtualenv --clear .venv ; fi

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr dist/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

deps: .venv
	. .venv/bin/activate && pip install -U -r requirements.txt -t ./src/libs

dev_deps: .venv
	. .venv/bin/activate && pip install -U -r dev_requirements.txt

build: clean
	mkdir ./dist
	cp ./src/project.py ./dist
	cp ./src/mlapp.py ./dist
	cd ./src && zip -x project.py -x mlapp.py -x \*libs\* -r ../dist/ml.zip ml
	cd ./src && zip -x project.py -x mlapp.py -x \*libs\* -r ../dist/config.zip config
	cd ./src && zip -x project.py -x mlapp.py -x \*libs\* -r ../dist/utils.zip utils
	cd ./libs && zip -r ../dist/libs.zip .

run:
    python mlapp.py