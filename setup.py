#!/usr/bin/env python3
## begin license ##
#
# "selftest": a simpler test runner for python
#
# Copyright (C) 2022-2023 Seecr (Seek You Too B.V.) https://seecr.nl
#
# This file is part of "selftest"
#
# "selftest" is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "selftest" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with "selftest".  If not, see <http://www.gnu.org/licenses/>.
#
## end license ##

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.rst").read_text(encoding="utf-8")

version = "0.2.0"

setup(
    name="selftest",
    version=version,
    description="Python Testing Library",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    author="Erik Groeneveld",
    author_email="erik@seecr.nl",
    url="https://github.com/seecr/selftest",
    scripts=["bin/selftest"],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Testing :: Unit",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
)
