#!/bin/bash

# Slurm sbatch options
#SBATCH -o script.sh.log-%j

# Loading the required modules
source /etc/profile
module load proxy-mitll
module load anaconda/2023b

# Run
python main.py