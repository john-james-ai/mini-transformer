#!/usr/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /reset.sh                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 11:43:05 pm                                                #
# Modified   : Tuesday August 19th 2025 11:45:46 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# See what’s installed
dpkg -l | grep -E 'redis|redis-stack' || true
systemctl list-unit-files | grep -i redis || true

# Stop any services (ignore if not present)
sudo systemctl stop redis-stack-server 2>/dev/null || true
sudo systemctl stop redis-server 2>/dev/null || true

# OPTIONAL: backup data/config before purging
sudo mkdir -p ~/redis-backup
sudo cp -a /etc/redis* ~/redis-backup/ 2>/dev/null || true
sudo cp -a /var/lib/redis* ~/redis-backup/ 2>/dev/null || true
sudo cp -a /var/log/redis* ~/redis-backup/ 2>/dev/null || true

# Purge packages + residual config
sudo apt-get purge -y redis-stack-server redis redis-server redis-tools redis-sentinel || true
sudo apt-get autoremove -y
sudo apt-get autoclean

# If you want a truly fresh repo setup, remove old Redis repo + keyring
sudo rm -f /etc/apt/sources.list.d/redis.list
sudo rm -f /usr/share/keyrings/redis-archive-keyring.gpg
sudo apt-get update
