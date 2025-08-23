#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /mini_transformer/container.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 07:59:27 pm                                                #
# Modified   : Saturday August 23rd 2025 12:35:10 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Framework Dependency Container"""

import logging
import logging.config  # pragma: no cover

import redis
from dependency_injector import containers, providers

from mini_transformer.data.repo.dataset_repo import DatasetRepo
from mini_transformer.infra.database.fal import FileAccessLayer
from mini_transformer.infra.database.oal import ObjectAccessLayer


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
class DatasetRepoContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # Redis Object Access Layer
    pool = providers.Singleton(
        redis.ConnectionPool,
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        decode_responses=config.redis.decode_responses,
    )

    oal = providers.Singleton(ObjectAccessLayer, location=config.location)

    fal = providers.Singleton(FileAccessLayer, location=config.location)

    repo = providers.Singleton(DatasetRepo, fal=fal, oal=oal)


# ------------------------------------------------------------------------------------------------ #
#                                       FRAMEWORK                                                  #
# ------------------------------------------------------------------------------------------------ #
class MiniTransformerContainer(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yaml"])

    logs = providers.Container(LoggingContainer, config=config)

    repo = providers.Container(DatasetRepoContainer, config=config.repo)
