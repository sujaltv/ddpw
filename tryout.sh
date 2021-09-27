#!/bin/sh

#SBATCH --output=ddp.out
#SBATCH --error=ddp.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --partition=Quick

source activate ddp
python main.py
conda deactivate
