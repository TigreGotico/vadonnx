#!/usr/bin/env python3
# Minimal shim so the shared `publish-alpha` automation can read the version
# via `python setup.py --version`. All real metadata lives in pyproject.toml.
from setuptools import setup

setup()
