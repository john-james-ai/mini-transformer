#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /observer.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 21st 2025 06:39:14 pm                                               #
# Modified   : Friday August 22nd 2025 07:34:18 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from mini_transformer.utils.mixins import FreezableMixin, ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Observer(ABC, FreezableMixin, ObjectRepresentationMixin):
    started_at: datetime = None
    ended_at: datetime = None
    duration: float = 0.0

    def start(self) -> None:
        self.started_at = datetime.now()

    def end(self) -> None:
        if self.started_at is None:
            raise TypeError("Attempted to end without first calling start().")
        self.ended_at = datetime.now()
        self.duration = round((self.ended_at - self.started_at).total_seconds(), 3)

    def __enter__(self) -> Observer:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()
        self.log_summary()

    @abstractmethod
    def log_summary(self) -> None:
        """Logs the observer."""
