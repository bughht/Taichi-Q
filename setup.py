# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='taichi_q',
    version='0.0.8',
    description='Taichi-Q: A quantum circuit simulator for both CPU and GPU',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Haotian Hong',
    author_email='bughht@outlook.com',
    install_requires=['numpy', 'taichi>=1.2.2',
                      'matplotlib', 'scipy'],
    python_requires='>3.6',
    url='https://github.com/bughht/Taichi-Q',
    license='Apache License Version 2.0',
    packages=find_packages(),
    test_suite='tests'
)

print(find_packages())
