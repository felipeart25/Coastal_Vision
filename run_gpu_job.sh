#!/bin/bash

#SBATCH --job-name=convLSTM        # Job name
#SBATCH --mail-user=daniel.arteagagutierrez@deltares.nl
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=output_%j.log         # Log file (%j = job ID)
#SBATCH --error=error_%j.log           # Error log file
#SBATCH --partition=gpu               # Adjust based on your cluster
#SBATCH --time=5-00:00:00                 # Max run time (2 hours)

# Load CUDA and necessary modules
module load python/3.11

# Activate virtual environment
source ~/Coastal_Vision/venv/bin/activate

# Run Python script
~/Coastal_Vision/venv/bin/python ./src/ConvLSTM.py
