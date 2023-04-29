#!/bin/bash
#SBATCH -A m3792
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --time=06:00:00
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

module load python
conda activate myPyTorch
export HDF5_USE_FILE_LOCKING=FALSE
srun python MainExp.py 

