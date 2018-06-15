"""
A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='feature_processing',  # Required
    version='1.1.0.dev1',  # Required
    description='Feature processing functions for machine learning and data science.',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/pypa/feature_processing',  # Optional
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Machine Learning :: Data Science',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
       'pandas >= 0.19.2',
       'numpy >= 1.12.1'
    ],
    author="Mihaela Mares",
    author_email="mihaela.andreea.mares@gmail.com",
    keywords='machine_learning data_science AI ML feature_processing',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
