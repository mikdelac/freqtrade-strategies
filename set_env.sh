#!/bin/bash

# Check if .env file exists
if [ ! -f .env ]; then
  echo ".env file not found!"
  exit 1
fi

# Export variables from .env file, handling both commented and uncommented lines
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ ! "$line" =~ ^[[:space:]]*# && -n "$line" ]]; then
        # Remove any inline comments
        line=${line%%#*}
        # Export the variable
        export "$line"
    fi
done < .env

echo "Environment variables set from .env file."
