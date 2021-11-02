#!/usr/bin/env python3

from distutils.core import setup

setup(
        name='autotest',
        version='0.1.0',
        description='Python Testing Library',
        author='Erik Groeneveld',
        author_email='erik@seecr.nl',
        url='https://github.com/seecr/autotest',
        # files are included in MANIFEST.in
        scripts=["autotest.py"],
        )

