#!/bin/bash
#SBATCH --job-name=fstereo
#SBATCH --account=ls_polle
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32000
#SBATCH --tmp=2000000
#SBATCH --gres=gpumem:11000m
#SBATCH --gpus=1
#SBATCH --output=fstereo_%j.out
#SBATCH --error=fstereo_%j.err

# Set up environment
set -e  # Exit on any error

# Check if batch folder is provided as argument
if [ $# -eq 0 ]; then
    echo "Error: No batch folder provided"
    echo "Usage: sbatch run_demo_folder.sbatch <batch_folder_path>"
    exit 1
fi

BATCH_FOLDER="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASH_SCRIPT="$SCRIPT_DIR/run_demo_folder.sh"

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Batch folder: $BATCH_FOLDER"
echo "TMPDIR: $TMPDIR"

# Run the main processing script
echo "Starting stereo processing batch..."
bash "$BASH_SCRIPT" "$BATCH_FOLDER"

echo "Job completed at: $(date)"
