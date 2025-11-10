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
        # "Forum": "http://forum.opennmt.net/",
        # "Gitter": "https://gitter.im/OpenNMT/OpenNMT-py",
        "Source": "https://github.com/Helsinki-NLP/mammoth",
    },
    python_requires=">=3.9",
    install_requires=[
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
