#!/bin/bash

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found."
    exit 1
fi

# Read dependencies, ignoring comments and empty lines
deps=()
while IFS= read -r line; do
    # Trim leading/trailing spaces
    line=$(echo "$line" | xargs)
    
    # Ignore empty lines and lines starting with #
    if [[ -n "$line" && "$line" != \#* ]]; then
        deps+=("$line")
    fi
done < requirements.txt

# Check if there are dependencies to add
if [ ${#deps[@]} -eq 0 ]; then
    echo "Error: No valid dependencies found in requirements.txt."
    exit 1
fi

# Run uv add command with valid dependencies
echo "Running: uv add ${deps[*]}"
uv add "${deps[@]}"

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error: Failed to add dependencies to pyproject.toml."
    exit 1
fi

echo "Dependencies successfully added to pyproject.toml."
exit 0
