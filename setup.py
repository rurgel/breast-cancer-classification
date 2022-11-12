#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    page_description = f.read()

# Package meta-data.
NAME = 'detect_breast_cancer'
DESCRIPTION = "Detect breast cancer from histological images."
URL = "https://github.com/rurgel/breast-cancer-image-classificator"
EMAIL = "rurgel@rurgel.io"
AUTHOR = "Rafael Gurgel"
REQUIRES_PYTHON = ">=3.7.0"

long_description = page_description
long_description_content_type="text/markdown",

# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'detect_breast_cancer'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"detect_breast_cancer": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    entry_points={
        'console_scripts': [
            'breast-cancer = detect_breast_cancer:call_prediction',
            ]
    }
)