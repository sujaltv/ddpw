#!/bin/sh

#SBATCH --output=ddp.out
#SBATCH --error=ddp.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --partition=Extended
#SBATCH --time=1-4

source activate ddp
python main.py train -slurm=true -log=true -ckpt-freq=5 --epochs=20 -b 64 -val 20
# python main.py resume -slurm=true -log=true -ckpt-freq=5 --epochs=20 -b 64 -val 20 -ckpt 19
conda deactivate
