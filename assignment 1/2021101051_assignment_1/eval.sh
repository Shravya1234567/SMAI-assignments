#!/bin/bash

# Check if a file path is provided
if [ $# -eq 0 ]
  then
    echo "No file path provided"
    exit 1
fi

# Check if the file exists
if [ ! -f $1 ]; then
    echo "File path not found"
    exit 1
fi

# Run the Python script
python3 test.py $1