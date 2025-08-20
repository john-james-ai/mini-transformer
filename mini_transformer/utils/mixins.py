#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /mixins.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 02:45:28 pm                                                #
# Modified   : Tuesday August 19th 2025 08:19:16 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import json
import logging
from dataclasses import fields, is_dataclass
from hashlib import sha1

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class FingerprintMixin:
    _fp_version = 1

    @property
    def fingerprint(self) -> str:
        if not is_dataclass(self):
            msg = "FingerprintMixin requires a dataclass subclass."
            logger.error(msg)
            raise TypeError(msg)
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)  # type: ignore[reportArgumentType]
            if f.metadata.get("stable")
        }
        payload["_v"] = self._fp_version
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha1(s.encode("utf-8")).hexdigest()[:8]
