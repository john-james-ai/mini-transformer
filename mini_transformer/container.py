#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/container.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 07:59:27 pm                                                #
# Modified   : Wednesday August 27th 2025 01:59:51 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Framework Dependency Container"""

import logging
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from mini_transformer.data.repo import DatasetRepo
from mini_transformer.data.tokenize.bpe import BPETokenization
from mini_transformer.infra.dal.fal import FileAccessLayer
from mini_transformer.infra.dal.oal import ObjectAccessLayer


# ------------------------------------------------------------------------------------------------ #
#                                        LOGGING                                                   #
# ------------------------------------------------------------------------------------------------ #
class LoggingContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


# ------------------------------------------------------------------------------------------------ #
#                                        REPO                                                      #
# ------------------------------------------------------------------------------------------------ #
class DataContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    oal = providers.Singleton(ObjectAccessLayer, location=config.repo)

    fal = providers.Singleton(FileAccessLayer, location=config.repo)

    repo = providers.Singleton(DatasetRepo, fal=fal, oal=oal)

    tokenization = providers.Singleton(
        BPETokenization,
        filepath=config.tokenizer.filepath,
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        unk_token=config.tokenizer.unk_token,
        special_tokens=config.tokenizer.special_tokens,
    )


# ------------------------------------------------------------------------------------------------ #
#                                       FRAMEWORK                                                  #
# ------------------------------------------------------------------------------------------------ #
class MiniTransformerContainer(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yaml"])

    logs = providers.Container(LoggingContainer, config=config)

    data = providers.Container(DataContainer, config=config.data)
