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
# Modified   : Monday August 25th 2025 09:07:05 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Framework Dependency Container"""

import logging
import logging.config  # pragma: no cover

from dependency_injector import containers, providers

from mini_transformer.data.repo.dataset import DatasetRepo
from mini_transformer.infra.dal.fal import FileAccessLayer
from mini_transformer.infra.dal.oal import ObjectAccessLayer
from mini_transformer.infra.data_io.download import HFDatasetDownloader


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
#                                        INFRA                                                     #
# ------------------------------------------------------------------------------------------------ #
class InfraContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    hf_downloader = providers.Singleton(
        HFDatasetDownloader,
        dataset=config.hf_downloader.dataset,
        language=config.hf_downloader.language,
        shuffle=config.hf_downloader.shuffle,
        buffer_size=config.hf_downloader.buffer_size,
        seed=config.hf_downloader.seed,
    )


# ------------------------------------------------------------------------------------------------ #
#                                        REPO                                                      #
# ------------------------------------------------------------------------------------------------ #
class RepoContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    oal_dataset = providers.Singleton(ObjectAccessLayer, location=config.dataset)

    fal_dataset = providers.Singleton(FileAccessLayer, location=config.dataset)

    oal_datafile = providers.Singleton(ObjectAccessLayer, location=config.datafile)

    fal_datafile = providers.Singleton(FileAccessLayer, location=config.datafile)

    dataset = providers.Singleton(DatasetRepo, fal=fal_dataset, oal=oal_dataset)

    datafile = providers.Singleton(DataFileRepo, fal=fal_datafile, oal=oal_datafile)


# ------------------------------------------------------------------------------------------------ #
#                                       FRAMEWORK                                                  #
# ------------------------------------------------------------------------------------------------ #
class MiniTransformerContainer(containers.DeclarativeContainer):

    config = providers.Configuration(yaml_files=["config.yaml"])

    logs = providers.Container(LoggingContainer, config=config)

    repo = providers.Container(RepoContainer, config=config.repo)

    infra = providers.Container(InfraContainer, config=config.infra)
