#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /tests/conftest.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 05:23:36 am                                                 #
# Modified   : Monday August 25th 2025 06:29:04 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import os
import shutil

import pytest

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.extractor.config import TranslationDatasetExtractorConfig
from mini_transformer.data.extractor.extract import TranslationDatasetExtractor
from tests.test_data import TEST_DATA_ROOT

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HF tokenizers threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ------------------------------------------------------------------------------------------------ #
#                                  DEPENDENCY INJECTION                                            #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = MiniTransformerContainer()
    container.init_resources()
    container.wire(modules=[__name__, "mini_transformer.data.extractor.extract"])
    return container


# ------------------------------------------------------------------------------------------------ #
#                                        DATASET                                                   #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def raw_dataset():
    config = TranslationDatasetExtractorConfig(n=8)
    extractor = TranslationDatasetExtractor(extractor_config=config)
    return extractor.extract()


# ------------------------------------------------------------------------------------------------ #
#                                          REPO                                                    #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="function", autouse=False)
def datasetrepo(container, dataset):
    """Returns a repository with a single dataset"""
    shutil.rmtree(path=TEST_DATA_ROOT, ignore_errors=True)
    repo = container.data.repo()
    repo.add(dataset=dataset)
    return repo
