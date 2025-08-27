#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/utils/io.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 27th 2025 01:23:46 am                                              #
# Modified   : Wednesday August 27th 2025 01:24:12 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path
from typing import Any, Dict, Union

import yaml


# ------------------------------------------------------------------------------------------------ #
class YamlIO:
    """A utility class for reading from and writing to YAML files.

    This class provides a simple, stateless interface with static methods for
    common YAML file operations, abstracting away the underlying `PyYAML`
    library calls.
    """

    @staticmethod
    def read(path: Union[str, Path]) -> Dict[str, Any]:
        """Reads and parses a YAML file from the given path.

        Args:
            path: The file path to the YAML file.

        Returns:
            A dictionary containing the parsed contents of the YAML file.
        """
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def write(path: Union[str, Path], data: Dict[str, Any]) -> None:
        """Writes a dictionary to a YAML file at the given path.

        If the file already exists, it will be overwritten.

        Args:
            path: The file path where the YAML file will be saved.
            data: The dictionary to write to the file.
        """
        with open(path, "w") as f:
            yaml.dump(data, f, indent=4, sort_keys=False)
