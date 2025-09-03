#!/bin/bash

# Script to process stereo pairs from all hash folders in a batch and sync results
# Usage: ./run_demo_folder.sh <batch_folder_path>
# Example: ./run_demo_folder.sh "/cluster/work/cvg/jiaqchen/obj_art/data/droid_fstereo_pairs/svo_batch_0"

# source the necessary env
source /cluster/project/cvg/jiaqchen/obj_ego/activate_articulate_cluster.sh

set -e  # Exit on any error

# Check if batch folder is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <batch_folder_path>"
    echo "Example: $0 /cluster/work/cvg/jiaqchen/obj_art/data/droid_fstereo_pairs/svo_batch_0"
    exit 1
fi

BATCH_FOLDER="$1"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="/cluster/project/cvg/jiaqchen/obj_ego_fstereo/FoundationStereo/scripts/run_demo_folder.py"

# Check if batch folder exists
if [ ! -d "$BATCH_FOLDER" ]; then
    echo "Error: Batch folder '$BATCH_FOLDER' does not exist"
    exit 1
fi

# Check if TMPDIR is set
if [ -z "$TMPDIR" ]; then
    echo "Error: TMPDIR environment variable is not set"
    exit 1
fi

# Extract batch name from path
BATCH_NAME=$(basename "$BATCH_FOLDER")

echo "Processing batch folder: $BATCH_FOLDER"
echo "Batch name: $BATCH_NAME"

# Remote server settings
REMOTE_HOST="julia@129.132.245.9"
REMOTE_BASE_DIR="/mnt/smolboi/egomimic_droid/fstereo_outputs"

# Create remote batch directory if it doesn't exist
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_BASE_DIR/$BATCH_NAME'"

# Function to process a single hash folder
process_hash_folder() {
    local HASH_FOLDER="$1"
    local HASH_NAME=$(basename "$HASH_FOLDER")
    
    echo ""
    echo "========================================="
    echo "Processing hash folder: $HASH_NAME"
    echo "========================================="
    
    # Create output directory structure in TMPDIR
    local OUTPUT_DIR="$TMPDIR/fstereo_outputs/$BATCH_NAME/$HASH_NAME"
    mkdir -p "$OUTPUT_DIR"
    
    echo "Input folder: $HASH_FOLDER"
    echo "Output directory: $OUTPUT_DIR"
    
    # Check if intrinsic file exists in the hash folder
    local INTRINSIC_FILE="$HASH_FOLDER/K.txt"
    if [ ! -f "$INTRINSIC_FILE" ]; then
        echo "Error: Intrinsic file not found at $INTRINSIC_FILE"
        return 1
    fi
    
    # Run the stereo processing
    echo "Starting stereo depth processing for $HASH_NAME..."
    echo "Using intrinsic file: $INTRINSIC_FILE"
    if python "$PYTHON_SCRIPT" \
        --input_folder "$HASH_FOLDER" \
        --out_dir "$OUTPUT_DIR" \
        --intrinsic_file "$INTRINSIC_FILE"; then
        # --save_individual_clouds 1; then
        echo "Stereo processing completed successfully for $HASH_NAME"
    else
        echo "Error: Stereo processing failed for $HASH_NAME"
        return 1
    fi
    
    # Sync results to remote server
    local REMOTE_DIR="$REMOTE_BASE_DIR/$BATCH_NAME/$HASH_NAME"
    
    echo "Syncing results to remote server..."
    echo "Local: $OUTPUT_DIR"
    echo "Remote: $REMOTE_HOST:$REMOTE_DIR"
    
    # Rsync the results
    if rsync -avz --progress "$OUTPUT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"; then
        echo "Successfully synced results for $HASH_NAME"
        echo "Remote location: $REMOTE_HOST:$REMOTE_DIR"
    else
        echo "Error: Failed to sync results for $HASH_NAME"
        return 1
    fi
    
    echo "Processing and sync completed successfully for $HASH_NAME"
}

# Count total hash folders
TOTAL_FOLDERS=$(find "$BATCH_FOLDER" -maxdepth 1 -type d ! -path "$BATCH_FOLDER" | wc -l)
echo "Found $TOTAL_FOLDERS hash folders to process"

# Process all hash folders in the batch
PROCESSED=0
FAILED=0

for HASH_FOLDER in "$BATCH_FOLDER"/*; do
    if [ -d "$HASH_FOLDER" ]; then
        HASH_NAME=$(basename "$HASH_FOLDER")
        PROCESSED=$((PROCESSED + 1))
        
        echo ""
        echo "Progress: $PROCESSED/$TOTAL_FOLDERS"
        
        if process_hash_folder "$HASH_FOLDER"; then
            echo "✓ Successfully processed $HASH_NAME"
        else
            echo "✗ Failed to process $HASH_NAME"
            FAILED=$((FAILED + 1))
            # Continue with next folder instead of exiting
        fi
    else
        # Skip non-directories (files, symlinks, etc.)
        ITEM_NAME=$(basename "$HASH_FOLDER")
        echo "Skipping non-directory: $ITEM_NAME"
    fi
done

echo ""
echo "========================================="
echo "BATCH PROCESSING SUMMARY"
echo "========================================="
echo "Total folders: $TOTAL_FOLDERS"
echo "Processed: $PROCESSED"
echo "Successful: $((PROCESSED - FAILED))"
echo "Failed: $FAILED"
echo "Batch: $BATCH_NAME"

if [ $FAILED -eq 0 ]; then
    echo "All folders processed successfully!"
    exit 0
else
    echo "Some folders failed to process. Check the logs above for details."
    exit 1
fi
