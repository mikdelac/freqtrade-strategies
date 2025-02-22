#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
  echo ".env file not found!"
  exit 1
fi

# Export variables from .env file
export $(grep -v '^#' .env | xargs)

echo "Environment variables set from .env file."
