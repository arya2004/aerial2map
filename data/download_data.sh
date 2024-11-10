#!/bin/bash


# Define dataset name and URL
FILE="maps"
URL="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz"
TAR_FILE="./$FILE.tar.gz"

echo "Starting download of [$FILE] dataset..."

# Download the maps dataset
wget -N $URL -O $TAR_FILE

# Extract the dataset to the current directory
tar -zxvf $TAR_FILE --strip-components=1 -C ./

# Remove the tar file after extraction
rm $TAR_FILE

echo "Download and extraction of [$FILE] dataset completed in the current directory."
