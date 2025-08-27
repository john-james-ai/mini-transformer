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
# Modified   : Wednesday August 27th 2025 01:55:08 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import os
import shutil

import pytest

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.builder.extractor import (
    TranslationDatasetBuilderRaw,
    TranslationDatasetBuilderRawConfig,
)
from mini_transformer.utils.io import YamlIO
from tests.test_data import TEST_DATA_ROOT

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # HF tokenizers threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ #
RAW_DATASET_SIZE = 64
CONFIG_PATH = "config.yaml"


# ------------------------------------------------------------------------------------------------ #
#                                  DEPENDENCY INJECTION                                            #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    """
    Fixture that temporarily modifies the config.yaml on disk to use a
    test-specific data repo, guaranteeing it's restored after the test.
    """
    # 1. Read the current config and save the original path
    config = YamlIO.read(CONFIG_PATH)
    original_repo_path = config["data"]["repo"]

    # 2. Write the temporary test path to the file
    config["data"]["repo"] = "tests/data/datasets/wmt14/"
    YamlIO.write(CONFIG_PATH, config)

    # 3. Now, initialize and yield the container
    container = MiniTransformerContainer()
    container.init_resources()
    container.wire(modules=[__name__])

    yield container

    # 4. TEARDOWN: Always restore the original path
    config["data"]["repo"] = original_repo_path
    YamlIO.write(CONFIG_PATH, config)
    container.unwire()


# ------------------------------------------------------------------------------------------------ #
#                                        DATASET                                                   #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def dataset():
    config = TranslationDatasetBuilderRawConfig(n=RAW_DATASET_SIZE, split="test")
    builder = TranslationDatasetBuilderRaw(config=config)
    return builder.build()


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
