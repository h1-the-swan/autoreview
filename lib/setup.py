# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='autoreview',
    version='0.1.0',
    description='Library for ranking relevant papers based on a set of seed papers',
    long_description=readme,
    author='Jason Portenoy',
    author_email='jporteno@uw.edu',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

