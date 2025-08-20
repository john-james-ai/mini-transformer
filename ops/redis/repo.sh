#!/usr/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /repo.sh                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 11:47:37 pm                                                #
# Modified   : Tuesday August 19th 2025 11:49:13 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# Prereqs
sudo apt-get install -y lsb-release curl gpg

# Add Redis official repo (use your codename; if unsupported, use jammy)
CODENAME=$(lsb_release -cs || echo jammy)
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $CODENAME main" \
  | sudo tee /etc/apt/sources.list.d/redis.list

sudo apt-get update
sudo apt-get install -y redis-stack-server

# Enable and start
sudo systemctl enable --now redis-stack-server
