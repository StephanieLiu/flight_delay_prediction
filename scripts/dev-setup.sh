#!/usr/bin/env bash
set -e

# Make sure command is ran from the root of the repository
if [[ ! -f scripts/dev-setup.sh ]]; then
    echo "Run this command at the root of the repository" 2>&1
    exit 1
fi

# Make sure venv is activated
if [[ "${VIRTUAL_ENV+x}y" != "xy" ]]; then
    if [[ ! -f .venv/bin/activate ]]; then
        echo "Virtual environment not found. Creating it."
        python -m venv .venv --prompt ivamlops
    fi
    echo "Virtual environment found. Activating it."
    . .venv/bin/activate
fi

# There is a few dependencies that need to be installed before the rest of the requirements
# since they are not tracked in the requirements files:
# - pip-tools

# Install proper dependencies if not already installed
if ! pip show pip-tools &> /dev/null; then
    echo "Installing pip-tools"
    pip install "pip-tools~=7.4"
fi

## Check if requirements.in has been compiled
if [[ ! -f requirements.txt ]]; then
    echo "Compiling requirements"
    make generate-requirements
fi

## Install requirements for local dev
pip install -r requirements/local.txt
