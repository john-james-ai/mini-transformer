#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /format.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 21st 2025 08:12:53 pm                                               #
# Modified   : Thursday August 21st 2025 10:07:23 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Dict

from .dtypes import IMMUTABLE_TYPES


# ------------------------------------------------------------------------------------------------ #
def dict_as_string(name: str, data: Dict, width: int = 80) -> str:
    breadth = int(width / 2)
    s = f"\n\n{name.center(width, ' ')}"
    for k, v in data.items():
        # if isinstance(v, datetime):
        #   v = v.isoformat(timespec="seconds")
        if isinstance(v, IMMUTABLE_TYPES):
            s += f"\n{k.rjust(breadth, ' ')} | {v}"
    s += "\n\n"
    return s
