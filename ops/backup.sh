#!/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /ops/backup.sh                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 27th 2025 12:32:39 pm                                              #
# Modified   : Wednesday August 27th 2025 12:56:02 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
##############################################################################
#
# backup.sh
#
# Creates a timestamped .tar.gz archive of a source directory. It uses
# flags to specify the source, destination, and an optional commit message.
#
# USAGE:
#   ./backup.sh [-s <source>] [-d <destination>] [-m <message>]
#
# OPTIONS:
#   -s <source>       (Optional) The directory to back up.
#                     Defaults to the value set in the CONFIGURATION section.
#
#   -d <destination>  (Optional) The directory where the backup is saved.
#                     Defaults to the value set in the CONFIGURATION section.
#
#   -m <message>      (Optional) A short, descriptive message (no spaces)
#                     to append to the filename.
#
# EXAMPLES:
#   # Run with all defaults
#   ./backup.sh
#
#   # Specify only a message, using defaults for source and destination
#   ./backup.sh -m "pre-refactor-snapshot"
#
#   # Specify a source and destination in any order
#   ./backup.sh -d /mnt/d/Archives -s /home/user/another_project
#
##############################################################################


# --- ⚙️ CONFIGURATION: SET YOUR DEFAULTS HERE ---
# IMPORTANT: Use WSL2 paths (e.g., /home/user, /mnt/c/...).
DEFAULT_SOURCE_DIR="/home/john/projects/mini-transformer/data"
DEFAULT_DEST_DIR="/mnt/c/Users/John/Documents/StudioAI/Projects/Mini-Transformer/backups/data"
# --- END CONFIGURATION ---



# --- ASSIGN DEFAULTS INITIALLY ---
SOURCE_DIR="$DEFAULT_SOURCE_DIR"
DEST_DIR="$DEFAULT_DEST_DIR"
COMMIT_MSG="" # Default to no message


# --- PARSE COMMAND-LINE OPTIONS ---
while [ "$#" -gt 0 ]; do
    case "$1" in
        -s) SOURCE_DIR="$2"; shift 2;;
        -d) DEST_DIR="$2"; shift 2;;
        -m) COMMIT_MSG="$2"; shift 2;;
        *)
            echo "❌ Error: Unknown option: $1"
            echo "   Usage: $0 [-s <source>] [-d <destination>] [-m <message>]"
            exit 1
            ;;
    esac
done


# --- DISPLAY PLANNED ACTION ---
echo "▶️  Source:      $SOURCE_DIR"
echo "▶️  Destination: $DEST_DIR"
[ -n "$COMMIT_MSG" ] && echo "▶️  Message:     $COMMIT_MSG"
echo "--------------------------------------------------"


# --- VALIDATE PATHS ---
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

# --- PREPARE FILENAME AND DESTINATION ---
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
BASENAME=$(basename "$SOURCE_DIR")
ARCHIVE_NAME="backup-${BASENAME}-${TIMESTAMP}"
if [ -n "$COMMIT_MSG" ]; then
    ARCHIVE_NAME+="-${COMMIT_MSG}"
fi
ARCHIVE_NAME+=".tar.gz"
FULL_DEST_PATH="$DEST_DIR/$ARCHIVE_NAME"

# Ensure destination directory exists
mkdir -p "$DEST_DIR"

# --- CREATE THE BACKUP ---
echo "🚀 Starting backup..."
tar -czf "$FULL_DEST_PATH" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"
echo "✅ Backup complete! Archive saved to: $FULL_DEST_PATH"