#!/usr/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /shutdown_systemd.sh                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 20th 2025 12:09:46 am                                              #
# Modified   : Wednesday August 20th 2025 12:10:18 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# See which unit you have
systemctl list-unit-files | grep -i redis

# Then stop it gracefully
sudo systemctl stop redis-server          # or: redis-stack-server
# optional: start again
sudo systemctl start redis-server
