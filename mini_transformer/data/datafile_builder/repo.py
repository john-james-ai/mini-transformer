#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/datafile_builder/repo.py                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 09:08:42 am                                                 #
# Modified   : Monday August 25th 2025 09:34:48 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from mini_transformer.data.base import DataObject
from mini_transformer.infra.dal.fal import FileAccessLayer
from mini_transformer.infra.dal.oal import ObjectAccessLayer
from mini_transformer.utils.exceptions import DataCorruptionError

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class Repo(ABC):
    def __init__(self, fal: FileAccessLayer, oal: ObjectAccessLayer) -> None:
        self._fal = fal
        self._oal = oal

    @abstractmethod
    def add(self, data: DataObject) -> None:
        """Add data to the repository."""


# ------------------------------------------------------------------------------------------------ #
class DataRepo(ABC):

    def __init__(self, fal: FileAccessLayer, oal: ObjectAccessLayer) -> None:

        self._fal = fal
        self._oal = oal

    def add(self, data: DataObject) -> None:

        key = getattr(data, "name", None)
        dsid = getattr(data, "id", None)
        if not isinstance(key, str) or not key:
            raise ValueError("Dataset is missing a valid 'name'.")
        if not isinstance(dsid, str) or not dsid:
            raise ValueError("Dataset is missing a valid 'id'.")

        # 1) Write rows first to avoid dangling meta on failure.
        self._fal.create(key=key, data=data.data)

        # 2) Persist slim manifest (data = []).
        self._oal.create(key=dsid, data=data.dematerialize())

        logger.info(
            f"Added data id: {data.id}, name: {data.name} to the Dataset repository."
        )

    def get(self, data_id: str) -> DataObject:

        try:
            data_meta = self._oal.read(key=data_id)  # raises KeyError if missing
        except KeyError:
            raise KeyError(f"No data manifest found for id '{data_id}'.") from None
        except Exception as e:
            raise OSError(f"Failed to read manifest for id '{data_id}'.") from e

        key = getattr(data_meta, "name", None)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Manifest for id '{data_id}' lacks a valid 'name'.")

        try:
            data = self._fal.read(key=key)  # should raise FileNotFoundError if missing
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found for id '{data_id}'."
            ) from None
        except Exception as e:
            raise OSError(f"Failed reading data file for id '{data_id}'.") from e

        try:
            return data_meta.materialize(data=data)
        except Exception as e:
            raise OSError(f"Failed to materialize data '{data_id}'.") from e

    def remove(self, data_id: str) -> bool:
        deleted_meta = False
        deleted_data = False

        # Resolve manifest → name
        try:
            meta = self._oal.read(key=data_id)
        except KeyError:
            logger.info("Nothing to remove for '%s' (metadata missing).", data_id)
            return False
        except Exception as e:
            logger.info(
                "Nothing to remove for '%s' (metadata store unavailable: %s).",
                data_id,
                e,
            )
            return False

        key = getattr(meta, "name", None)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Metadata for '{data_id}' lacks a valid 'name'.")

        # Delete data first, then manifest
        try:
            deleted_data = self._fal.delete(key=key)  # False if not present
        except Exception as e:
            raise OSError(f"Failed deleting data for '{data_id}'.") from e

        try:
            deleted_meta = self._oal.delete(key=data_id)  # False if not present
        except Exception as e:
            raise OSError(f"Failed deleting metadata for '{data_id}'.") from e

        if deleted_meta and deleted_data:
            logger.info("Removed data '%s' (metadata + data).", data_id)
        elif deleted_meta:
            logger.warning("Removed metadata only for '%s' (data missing).", data_id)
        elif deleted_data:
            logger.warning("Removed data only for '%s' (metadata missing).", data_id)
        else:
            logger.info("Nothing to remove for '%s'.", data_id)

        return deleted_meta or deleted_data

    def exists(self, data_id: str) -> bool:
        if self._oal.exists(key=data_id):
            data_meta = self._oal.read(key=data_id)
            key = data_meta.name
            if self._fal.exists(key=key):
                return True
            else:
                raise DataCorruptionError(
                    f"Dataset file for data_id: {data_id}, name: {data_meta.name} does not exist."
                )
        else:
            return False

    def show(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        # List all manifest keys
        try:
            ids = self._oal.get_all_names()
        except Exception as e:
            logger.debug("show(): manifest listing unavailable: %s", e)
            ids = []

        for data_id in ids:
            try:
                meta = self._oal.read(key=data_id)
            except KeyError:
                continue
            except Exception as e:
                logger.debug("show(): skipping '%s' (read error: %s)", data_id, e)
                continue

            try:
                info = dict(meta.info)  # property returning a dict
            except Exception as e:
                logger.debug("show(): manifest '%s' has invalid .info (%s)", data_id, e)
                info = {
                    "data_id": data_id,
                    "name": getattr(meta, "name", None),
                    "size": None,
                }

            rows.append(info)

        df = pd.DataFrame(rows)
        if not df.empty:
            order = [
                c for c in ("data_id", "name", "size", "created") if c in df.columns
            ]
            df = (
                df.loc[:, order + [c for c in df.columns if c not in order]]
                .sort_values(
                    order[:2] or ["data_id"], kind="stable", na_position="last"
                )
                .reset_index(drop=True)
            )
        return df
