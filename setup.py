
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='feature_stuff',
    version='0.0.dev5',
    description='Feature extraction, processing and interpretation algorithms and functions for machine learning and data science.',  
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hiflyin/Advanced-Feature-Stuff-Lib',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
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
       'numpy >= 1.12.1',
       'scikit-learn >= 0.18',
       'scipy >= 0.19.0',
       'xgboost >= 0.6'
    ],
    author="Mihaela Mares",
    author_email="mihaela.andreea.mares@gmail.com",
    keywords='machine_learning data_science AI ML feature_extraction',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
