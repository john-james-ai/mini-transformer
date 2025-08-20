#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /__main__.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 18th 2025 11:29:48 pm                                                 #
# Modified   : Tuesday August 19th 2025 03:34:24 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
import logging.handlers
import os
import sys

import typer

from mini_transformer.config import PathsConfig

# ------------------------------------------------------------------------------------------------ #
# --- Typer App Initialization ---
# This creates the main application object.
app = typer.Typer(
    name="Mini-Transformer",
    help="A small custom Transformer implemented from scratch in Numpy.",
    add_completion=False,
)


# ------------------------------------------------------------------------------------------------ #
def setup_logging(log_filepath: str) -> None:
    """
    Configures a time-rotating logger.

    Log files will rotate daily, and up to 7 old log files will be kept.
    This function configures the root logger, so any module using
    logging.getLogger(__name__) will inherit this configuration.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_filepath)
    os.makedirs(log_dir, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent handlers from being added multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler for rotating files
    handler = logging.handlers.TimedRotatingFileHandler(
        log_filepath, when="d", interval=1, backupCount=7
    )

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    logger.addHandler(handler)

    # Also log to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging has been configured successfully.")


# ------------------------------------------------------------------------------------------------ #
@app.command()
def main():
    # Setup Logging
    log_filepath = os.path.join(PathsConfig().logs_dir, "mini-transformer.log")
    setup_logging(log_filepath)


# ------------------------------------------------------------------------------------------------ #
if __name__ == "__main__":
    app()
