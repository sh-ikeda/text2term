from setuptools import setup, find_packages
import re

VERSION = '4.3.1'

description = 'a tool for mapping free-text descriptions of entities to ontology terms'
long_description = open('README.md').read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='text2term',
    version=VERSION,
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/rsgoncalves/text2term',
    license='MIT',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Rafael Gonçalves, Jason Payne',
    author_email='rsgoncalves@gmx.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Environment :: Console',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering'
    ],
    python_requires=">=3.9",
)
