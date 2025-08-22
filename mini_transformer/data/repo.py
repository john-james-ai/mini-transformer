#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /repo.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 06:28:04 pm                                                #
# Modified   : Friday August 22nd 2025 06:43:45 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
from abc import ABC
from typing import Any

import pandas as pd

from mini_transformer.infra.database.fal import FileAccessLayer
from mini_transformer.infra.database.oal import ObjectAccessLayer

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
        manifest (same dataset object with `data=[]`) under `dataset.dataset_id`.

        Args:
            dataset: A Dataset (or subclass) instance with attributes
                `dataset_id`, `dataset_name`, and `data` (list of dicts).

        Raises:
            ValueError: If required attributes are missing or invalid.
            FileExistsError: If `fal`/`oal` enforce exclusive create and an object already exists.
            OSError: On underlying I/O failures (file write or object store issues).
        """
        name = getattr(dataset, "dataset_name", None)
        dsid = getattr(dataset, "dataset_id", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Dataset is missing a valid 'dataset_name'.")
        if not isinstance(dsid, str) or not dsid:
            raise ValueError("Dataset is missing a valid 'dataset_id'.")

        # 1) Write rows first to avoid dangling meta on failure.
        self._fal.create(name=name, data=dataset.data)

        # 2) Persist slim manifest (data = []).
        self._oal.create(name=dsid, data=dataset.dematerialize())

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
            dataset_meta = self._oal.read(name=dataset_id)  # raises KeyError if missing
        except KeyError:
            raise KeyError(
                f"No dataset manifest found for id '{dataset_id}'."
            ) from None
        except Exception as e:
            raise OSError(f"Failed to read manifest for id '{dataset_id}'.") from e

        name = getattr(dataset_meta, "dataset_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Manifest for id '{dataset_id}' lacks a valid 'dataset_name'."
            )

        try:
            data = self._fal.read(
                name=name
            )  # should raise FileNotFoundError if missing
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
            ValueError: If the manifest exists but lacks a valid dataset_name.
            OSError: If underlying delete operations fail unexpectedly.
        """
        deleted_meta = False
        deleted_data = False

        # Resolve manifest → dataset_name
        try:
            meta = self._oal.read(name=dataset_id)
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

        name = getattr(meta, "dataset_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Metadata for '{dataset_id}' lacks a valid 'dataset_name'."
            )

        # Delete data first, then manifest
        try:
            deleted_data = self._fal.delete(name=name)  # False if not present
        except Exception as e:
            raise OSError(f"Failed deleting data for '{dataset_id}'.") from e

        try:
            deleted_meta = self._oal.delete(name=dataset_id)  # False if not present
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
        return self._oal.exists(name=dataset_id)

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
                meta = self._oal.read(name=dataset_id)
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
                    "name": getattr(meta, "dataset_name", None),
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
