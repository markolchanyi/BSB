#!/bin/bash

# Function to split comma-separated numbers and iterate through them

input_string="$1"
bvalpath="$2"
bvecpath="$3"
datapath="$4"
outputpath="$5"


#!/bin/bash

# Define the directory name
DIR_NAME=$outputpath
echo "working with $outputpath"

# Safety checks
if [ "$DIR_NAME" = "/" ]; then
    echo "Dangerous operation! You cannot delete the base directory."
    exit 1
fi

if [[ "$DIR_NAME" != *shell* ]]; then
    echo "Safety check failed! The directory name does not contain the keyphrase 'shell'."
    exit 1
fi

# Check if directory exists
if [ -d "$DIR_NAME" ]; then
    echo "Directory $DIR_NAME exists. Deleting and remaking it."
    rm -r "$DIR_NAME"
else
    echo "Directory $DIR_NAME does not exist. Creating it."
fi

# Create the directory
mkdir "$DIR_NAME"



# Convert comma-separated string into array
IFS=',' read -ra num_array <<< "$input_string"

# Iterate over the array
for num in "${num_array[@]}"; do
    # Process each number (in this case, just print it)
    echo "Processing shell: $num"
    echo "Performing operation: dwiextract -shell 0,$num datapath $outputpath/data_shell$num.mif -fslgrad $bvecpath $bvalpth --force"
    dwiextract -shell 0,$num $datapath $outputpath/data_shell$num.mif -fslgrad $bvecpath $bvalpath --force
    mrconvert $outputpath/data_shell$num.mif $outputpath/data_shell$num.nii.gz -export_grad_fsl $outputpath/bvecs_shell$num $outputpath/bvals_shell$num
done


