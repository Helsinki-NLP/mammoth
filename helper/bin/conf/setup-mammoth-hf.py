#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mammoth-nlp',
    description='Massively Multilingual Modular Open Translation @ Helsinki',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.2.1',
    packages=find_packages(),
    project_urls={
        "Documentation": "https://helsinki-nlp.github.io/mammoth/",
        "Source": "https://github.com/Helsinki-NLP/mammoth",
    },
    python_requires=">=3.5",
    install_requires=[
        # discontinued support:
        # "pyonmttok>=1.32,<2",
        # "sentencepiece==0.1.97",
        # check if we can stop using also these:
        "configargparse",
        "flake8==4.0.1",
        "flask==2.0.3",
        "pytest-flake8==1.1.1",
        "pytest==7.0.1",
        "pyyaml==6.0.2",
        "scikit-learn==1.2.0",
        "tensorboard>=2.9",
        "timeout_decorator",
        "torch>=1.10.2",
        "waitress",
        "x-transformers==1.32.14",
        # feat/hf-integration: in site-packages
        "absl-py==2.3.1",
        "colorama==0.4.6",
        "filelock==3.19.1",
        "grpcio==1.74.0",
        "hf-xet==1.1.8",
        "huggingface-hub==0.34.4",
        "Jinja2==3.1.6",
        "lxml==6.0.1",
        "Markdown==3.8.2",
        "mpmath==1.3.0",
        "numpy==2.3.2",
        "pillow==11.3.0",
        "portalocker==3.2.0",
        "regex==2025.7.34",
        "requests==2.32.5",
        "safetensors==0.6.2",
        "tabulate==0.9.0",
        "tokenizers==0.21.4",
        "Werkzeug==3.1.3",
        # feat/hf-integration: to be added
        "certifi==2025.8.3",
        "charset-normalizer==3.4.3",
        "configargparse<=1.7.1",
        "einops>=0.8.0,<=0.8.1",
        "einx==0.3.0",
        "frozendict==2.4.6",
        "fsspec==2025.7.0",
        "idna==3.10",
        "loguru==0.7.3",
        "markupsafe==3.0.2",
        "networkx==3.5",
        "packaging==25.0",
        "protobuf==6.32.0",
        "sacrebleu>=2.3.1,<=2.5.1",
        "setuptools==80.9.0",
        "sympy==1.14.0",
        "tensorboard>=2.0,<=2.20.0",
        "tqdm>=4.66.2,<=4.67.1",
        "transformers==4.55.4",
        "typing_extensions==4.14.1",
        "urllib3==2.5.0",
    ],
    entry_points={
        "console_scripts": [
            # "onmt_server=mammoth.bin.server:main",
            "mammoth_train=mammoth.bin.train:main",
            "mammoth_translate=mammoth.bin.translate:main",
            "mammoth_config_config=mammoth.bin.config_config:main",
            "mammoth_iterate_tasks=mammoth.bin.iterate_tasks:main",
            "mammoth_generate_synth_data=mammoth.bin.generate_synth_data:main",
            # "onmt_release_model=mammoth.bin.release_model:main",
            # "onmt_average_models=mammoth.bin.average_models:main",
            # "onmt_build_vocab=mammoth.bin.build_vocab:main",
        ],
    },
)
