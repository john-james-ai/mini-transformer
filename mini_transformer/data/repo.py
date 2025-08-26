#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/repo/dataset.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 06:28:04 pm                                                #
# Modified   : Monday August 25th 2025 08:27:46 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
from abc import ABC
from typing import Any

import pandas as pd

from mini_transformer.infra.dal.fal import FileAccessLayer
from mini_transformer.infra.dal.oal import ObjectAccessLayer
from mini_transformer.utils.exceptions import DataCorruptionError

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class DatasetRepo(ABC):
    """Lightweight local repository for dataset manifests + row files.

    Persists datasets using:
      - ObjectAccessLayer (OAL): stores a *slim* manifest (same class as the dataset) with `data=[]`.
      - FileAccessLayer (FAL): stores row data as JSONL at a deterministic path.

    Policy:
      - `add()` writes rows first, then the slim manifest (to avoid dangling meta on write failure).
      - `get()` rehydrates by reading manifest then the JSONL rows.
      - `remove()` is best-effort and idempotent: deletes whatever exists; returns True if anything was removed.
      - `show()` returns a minimal registry derived from each manifest's `.info` (dict).
    """

    def __init__(self, fal: FileAccessLayer, oal: ObjectAccessLayer) -> None:
        """Initialize the repository.

        Args:
            fal: File access layer providing create/read/delete for JSONL files.
            oal: Object access layer providing create/read/delete and listing of manifests.

        Notes:
            - `fal.create()` should create parent directories as needed.
            - `oal.read()` is expected to raise KeyError if the key is missing.
        """

        self._fal = fal
        self._oal = oal

    def add(self, dataset) -> None:
        """Persist a dataset (rows + slim manifest).

        Writes the materialized rows to JSONL, then stores a dematerialized
        manifest (same dataset object with `data=[]`) under `dataset.id`.

        Args:
            dataset: A Dataset (or subclass) instance with attributes
                `id`, `name`, and `data` (list of dicts).

        Raises:
            ValueError: If required attributes are missing or invalid.
            FileExistsError: If `fal`/`oal` enforce exclusive create and an object already exists.
            OSError: On underlying I/O failures (file write or object store issues).
        """
        key = getattr(dataset, "name", None)
        dsid = getattr(dataset, "id", None)
        if not isinstance(key, str) or not key:
            raise ValueError("Dataset is missing a valid 'name'.")
        if not isinstance(dsid, str) or not dsid:
            raise ValueError("Dataset is missing a valid 'id'.")

        # 1) Write rows first to avoid dangling meta on failure.
        self._fal.create(key=key, data=dataset.data)

        # 2) Persist slim manifest (data = []).
        self._oal.create(key=dsid, data=dataset.dematerialize())

        logger.info(
            f"Added dataset id: {dataset.id}, name: {dataset.name} to the Dataset repository."
        )

    def get(self, dataset_id: str):
        """Return the dataset for the given `dataset_id` (manifest + rows).

        Args:
            dataset_id: Identifier used as the manifest key in the object store.

        Returns:
            A materialized Dataset (or subclass) instance with rows populated.

        Raises:
            KeyError: If the manifest is not found.
            FileNotFoundError: If the JSONL data file is not found.
            OSError: For other I/O or deserialization errors.
        """
        try:
            dataset_meta = self._oal.read(key=dataset_id)  # raises KeyError if missing
        except KeyError:
            raise KeyError(
                f"No dataset manifest found for id '{dataset_id}'."
            ) from None
        except Exception as e:
            raise OSError(f"Failed to read manifest for id '{dataset_id}'.") from e

        key = getattr(dataset_meta, "name", None)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Manifest for id '{dataset_id}' lacks a valid 'name'.")

        try:
            data = self._fal.read(key=key)  # should raise FileNotFoundError if missing
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found for id '{dataset_id}'."
            ) from None
        except Exception as e:
            raise OSError(f"Failed reading data file for id '{dataset_id}'.") from e

        try:
            return dataset_meta.materialize(data=data)
        except Exception as e:
            raise OSError(f"Failed to materialize dataset '{dataset_id}'.") from e

    def remove(self, dataset_id: str) -> bool:
        """Remove both manifest and data for `dataset_id` (best-effort, idempotent).

        Returns:
            True if at least one of (manifest, data file) was deleted; False if nothing existed.

        Raises:
            ValueError: If the manifest exists but lacks a valid name.
            OSError: If underlying delete operations fail unexpectedly.
        """
        deleted_meta = False
        deleted_data = False

        # Resolve manifest → name
        try:
            meta = self._oal.read(key=dataset_id)
        except KeyError:
            logger.info("Nothing to remove for '%s' (metadata missing).", dataset_id)
            return False
        except Exception as e:
            logger.info(
                "Nothing to remove for '%s' (metadata store unavailable: %s).",
                dataset_id,
                e,
            )
            return False

        key = getattr(meta, "name", None)
        if not isinstance(key, str) or not key:
            raise ValueError(f"Metadata for '{dataset_id}' lacks a valid 'name'.")

        # Delete data first, then manifest
        try:
            deleted_data = self._fal.delete(key=key)  # False if not present
        except Exception as e:
            raise OSError(f"Failed deleting data for '{dataset_id}'.") from e

        try:
            deleted_meta = self._oal.delete(key=dataset_id)  # False if not present
        except Exception as e:
            raise OSError(f"Failed deleting metadata for '{dataset_id}'.") from e

        if deleted_meta and deleted_data:
            logger.info("Removed dataset '%s' (metadata + data).", dataset_id)
        elif deleted_meta:
            logger.warning("Removed metadata only for '%s' (data missing).", dataset_id)
        elif deleted_data:
            logger.warning("Removed data only for '%s' (metadata missing).", dataset_id)
        else:
            logger.info("Nothing to remove for '%s'.", dataset_id)

        return deleted_meta or deleted_data

    def exists(self, dataset_id: str) -> bool:
        """Check whether a dataset exists in both metadata and file storage.

        This method verifies the integrity of a dataset by ensuring two conditions:
        1. The dataset metadata exists in the object access layer (OAL).
        2. The corresponding dataset file exists in the file access layer (FAL).

        If the metadata exists but the file does not, a ``DataCorruptionError`` is raised
        to signal an inconsistent or corrupted state.

        Args:
            dataset_id (str): Unique identifier of the dataset to check.

        Returns:
            bool: ``True`` if both metadata and dataset file exist;
                ``False`` if the dataset metadata does not exist.

        Raises:
            DataCorruptionError: If metadata exists but the corresponding dataset file is missing.
        """
        if self._oal.exists(key=dataset_id):
            dataset_meta = self._oal.read(key=dataset_id)
            key = dataset_meta.name
            if self._fal.exists(key=key):
                return True
            else:
                raise DataCorruptionError(
                    f"Dataset file for dataset_id: {dataset_id}, name: {dataset_meta.name} does not exist."
                )
        else:
            return False

    def show(self) -> pd.DataFrame:
        """Return a minimal registry of datasets as a pandas DataFrame.

        Columns come from each manifest's `.info` dict (kept intentionally small
        and dataset-type-agnostic). Typical keys: ``dataset_id``, ``name``,
        ``size``, and optionally ``created`` when provided by a subclass.

        Returns:
            pandas.DataFrame: One row per dataset in the object store. If the
            object store is empty or unavailable, returns an empty DataFrame.

        Notes:
            - This method does *not* read JSONL rows; it relies only on manifests.
            - Any manifests that disappear between list/read are skipped.
        """
        rows: list[dict[str, Any]] = []

        # List all manifest keys
        try:
            ids = self._oal.get_all_names()
        except Exception as e:
            logger.debug("show(): manifest listing unavailable: %s", e)
            ids = []

        for dataset_id in ids:
            try:
                meta = self._oal.read(key=dataset_id)
            except KeyError:
                continue
            except Exception as e:
                logger.debug("show(): skipping '%s' (read error: %s)", dataset_id, e)
                continue

            try:
                info = dict(meta.info)  # property returning a dict
            except Exception as e:
                logger.debug(
                    "show(): manifest '%s' has invalid .info (%s)", dataset_id, e
                )
                info = {
                    "dataset_id": dataset_id,
                    "name": getattr(meta, "name", None),
                    "size": None,
                }

            rows.append(info)

        df = pd.DataFrame(rows)
        if not df.empty:
            order = [
                c for c in ("dataset_id", "name", "size", "created") if c in df.columns
            ]
            df = (
                df.loc[:, order + [c for c in df.columns if c not in order]]
                .sort_values(
                    order[:2] or ["dataset_id"], kind="stable", na_position="last"
                )
                .reset_index(drop=True)
            )
        return df
