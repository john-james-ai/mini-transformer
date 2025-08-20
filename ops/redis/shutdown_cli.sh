#!/usr/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /shutdown_cli.sh                                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 20th 2025 12:10:34 am                                              #
# Modified   : Wednesday August 20th 2025 12:10:53 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# Graceful shutdown with default persistence behavior
redis-cli shutdown

# Force an RDB/AOF save before exit (fail shutdown if save fails)
redis-cli shutdown SAVE

# Don’t save (discard unpersisted changes)
redis-cli shutdown NOSAVE
