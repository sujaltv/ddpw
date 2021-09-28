#!/bin/sh

#SBATCH --output=ddp.out
#SBATCH --error=ddp.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --partition=Extended
#SBATCH --time=0-1

source activate ddp
python main.py
conda deactivate
