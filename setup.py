# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

requirements = ['pandas', 'numpy', 'sklearn']

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='autoreview',
    version='0.2.0',
    install_requires=requirements,
    description='Library for ranking relevant papers based on a set of seed papers',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Jason Portenoy',
    author_email='jporteno@uw.edu',
    url='https://github.com/h1-the-swan/autoreview',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs'))
)

