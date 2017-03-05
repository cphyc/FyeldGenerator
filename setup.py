# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('Readme.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='FyeldGenerator',
    version='0.1.0',
    description='Simple package to generate random field.',
    long_description=readme,
    author='Corentin Cadiou',
    author_email='contact@cphyc.me',
    url='https://github.com/cphyc/FyeldGenerator',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'six'
    ]
)
