#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -i input_txt"
    exit 1
}

# Parse command-line arguments
while getopts ":i:" opt; do
  case $opt in
    i)
      input_txt=$OPTARG
      ;;
    *)
      usage
      ;;
  esac
done

# Check if input_txt is set
if [ -z "$input_txt" ]; then
    usage
fi

# Calculate the average of the numbers in the text file, ignoring blank rows
average=$(awk 'NF > 0 {sum += $1; count += 1} END {if (count > 0) print sum / count}' "$input_txt")

# Print the average
echo "Average: $average"
