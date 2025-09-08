#!/bin/bash

#SBATCH --account=chenjun3
#SBATCH --partition=a100x4

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.error
#SBATCH --output=SBATCH.out


cd $SLURM_SUBMIT_DIR
module load nvidia/cuda/12.2
python -u  GRPO_demo.py  > GRPO_demo.out
