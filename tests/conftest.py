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
# Modified   : Monday August 25th 2025 08:27:46 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import shutil

import pytest

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.dataset_builder.builder import (
    TranslationDatasetBuilder,
    TranslationDatasetBuilderConfig,
)
from tests.test_data import TEST_DATA_ROOT


# ------------------------------------------------------------------------------------------------ #
#                                  DEPENDENCY INJECTION                                            #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=True)
def container():
    container = MiniTransformerContainer()
    container.init_resources()
    return container


# ------------------------------------------------------------------------------------------------ #
#                                        DATASET                                                   #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def dataset():
    builder_config = TranslationDatasetBuilderConfig(
        split="test", dataset_target_size=16, oversample=3
    )
    builder = TranslationDatasetBuilder(builder_config=builder_config)
    return builder.build()


# ------------------------------------------------------------------------------------------------ #
#                                          REPO                                                    #
# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="function", autouse=False)
def datasetrepo(container, dataset):
    """Returns a repository with a single dataset"""
    shutil.rmtree(path=TEST_DATA_ROOT, ignore_errors=True)
    repo = container.repo.repo()
    repo.add(dataset=dataset)
    return repo


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="module", autouse=False)
def datasetrepo3(container, dataset):
    """Returns a repository with a single dataset"""
    shutil.rmtree(path=TEST_DATA_ROOT, ignore_errors=True)

    train_config = TranslationDatasetBuilderConfig(
        split="train", dataset_target_size=16, oversample=3
    )
    val_config = TranslationDatasetBuilderConfig(
        split="validation", dataset_target_size=12, oversample=3
    )
    test_config = TranslationDatasetBuilderConfig(
        split="test", dataset_target_size=8, oversample=3
    )

    builder_train = TranslationDatasetBuilder(builder_config=train_config)
    builder_val = TranslationDatasetBuilder(builder_config=val_config)
    builder_test = TranslationDatasetBuilder(builder_config=test_config)

    train = builder_train.build()
    val = builder_val.build()
    test = builder_test.build()

    repo = container.repo.repo()

    repo.add(dataset=train)
    repo.add(dataset=val)
    repo.add(dataset=test)
    return repo
